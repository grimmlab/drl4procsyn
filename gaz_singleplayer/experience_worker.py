import copy
import random
import sys
import psutil
import os

import torch
import numpy as np
import ray
import time
from gaz_singleplayer.config_syngame import Config
from gaz_singleplayer.syn_game import Game
from local_inferencer import LocalInferencer
from environment.env_config import EnvConfig
from gaz_singleplayer.rollout_obtainer import RolloutObtainer
from gaz_singleplayer.synthesis_network import SynthesisNetwork
from gumbel_mcts_single import SingleplayerGumbelMCTS
from inferencer import ModelInferencer
from shared_storage import SharedStorage

from typing import Dict, Optional, Type


@ray.remote
class ExperienceWorker:
    """
    Instances of this class run in separate processes and continuously play singleplayer matches.
    The game history is saved to the global replay buffer, which is accessed by the network trainer process.
    """

    def __init__(self, actor_id: int, config: Config, env_config: EnvConfig, shared_storage: SharedStorage, model_inference_worker: ModelInferencer,
                 game_class: Type[Game], network_class: Type[SynthesisNetwork], random_seed: int = 42, cpu_core: int = None):
        """
        actor_id [int]: Unique id to identify the self play process. Is used for querying the inference models, which
            send back the results to the actor.
        config [Config]: Config object
        shared_storage [SharedStorage]: Shared storage worker.
        model_inference_worker [ModelInferencer]: Instance of model inferencer to which the actor sends states to evaluate
        game_class: Subclass of Game from which instances of games are constructed
        random_seed [int]: Random seed for this actor
        """
        if config.pin_workers_to_core and sys.platform == "linux" and cpu_core is not None:
            os.sched_setaffinity(0, {cpu_core})
            psutil.Process().cpu_affinity([cpu_core])

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        self.actor_id = actor_id
        self.config = config
        self.env_config = env_config
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.network_class = network_class
        self.n_games_played = 0

        if isinstance(model_inference_worker, str):
            # Infer locally with device given by `model_inference_worker`
            model_inference_worker = LocalInferencer(
                config=self.config, env_config=env_config,
                shared_storage=shared_storage,
                network_class=network_class,
                model_named_keys=["newcomer", "best"],
                initial_checkpoint=None,  # is set in local inferencer
                device=torch.device(model_inference_worker),
            )
        self.model_inference_worker = model_inference_worker

        self.rollout_obtainer = RolloutObtainer(
            config=self.config, env_config=self.env_config,
            actor_id=self.actor_id,
            game_class=self.game_class, inferencer=self.model_inference_worker
        )

        # Stores MCTS tree which is persisted over the full game
        self.mcts: Optional[SingleplayerGumbelMCTS] = None
        # Set the random seed for the worker
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def play_game(self, problem_instance=None, deterministic: bool = False,
                  model_to_use_for_learning="newcomer", model_to_use_for_baseline_rollout="best",
                  best_moves_first=False, eval_mode=False):
        """
        Performs one match of the singleplayer game based on singleplayer GAZ MCTS.
        """
        game_time = time.perf_counter()  # track how long the worker needs for a game
        # initialize game and game history
        if problem_instance is None:
            problem_instance = self.game_class.generate_random_instance(self.env_config)

        # Obtain trajectory against which the learning actor competes and which is used as baseline for singleplayer
        env_config = copy.deepcopy(self.env_config)
        if eval_mode:
            env_config.max_num_root_finding_interactions = 0  # let root finding fsolve perform as needed
        baseline_trajectory = self.rollout_obtainer.get_trajectory(instance=problem_instance, model_type_for_policy=model_to_use_for_baseline_rollout,
                                                                   env_config=env_config)

        if baseline_trajectory is None:
            # no trajectory could be found which is feasible. In reality, should not happen
            return None, "broken"

        baseline_states = baseline_trajectory.raw_state_sequence
        baseline_objective = baseline_trajectory.objective
        baseline_action_sequence = baseline_trajectory.current_game_state.get_sequence(0)

        game: Game = self.game_class(config=self.config,
                                     flowsheet_simulation_config=env_config,
                                     problem_instance=problem_instance,
                                     opponent_outcome=None)
        game_history = GameHistory()

        # keep the baseline/objective rollout timestep synchronous to our learning actor
        baseline_state_index = 0 if not best_moves_first else 1
        game_history.observation_history.append(game.get_current_state())
        game_history.level_history.append(game.get_current_level())

        game_done = False

        game_stats = {
            "objective": float("-inf"),
            "sequence": None,
            "num_level_0_moves": 0,
            "max_search_depth": 0,
            "policies_for_selected_moves": {},
            "baseline_objective": baseline_objective,
            "baseline_sequence": baseline_action_sequence,
            "baseline_num_moves": sum([state["action_level"] == 0 for state in baseline_states]) - 1
        }

        move_counter = 0

        if not self.config.inference_on_experience_workers:
            ray.get(self.model_inference_worker.register_actor.remote(self.actor_id))
        with torch.no_grad():
            self.mcts = tree = SingleplayerGumbelMCTS(actor_id=self.actor_id, config=self.config,
                                                      model_inference_worker=self.model_inference_worker,
                                                      deterministic=deterministic,
                                                      min_max_normalization=True,
                                                      model_to_use=model_to_use_for_learning)

            game_broken = False
            while not game_done:
                level = game.get_current_level()

                root, mcts_info = tree.run_at_root(game)

                # Store maximum search depth for inspection
                if "max_search_depth" in mcts_info and game_stats["max_search_depth"] < mcts_info["max_search_depth"]:
                    game_stats["max_search_depth"] = mcts_info["max_search_depth"]

                if root.num_feasible_actions() == 1:
                    # auto-choose single possible action
                    action = root.get_feasible_actions()[0]
                    root.sequential_halving_chosen_action = action  # set as chosen action in node
                else:
                    action = root.sequential_halving_chosen_action

                if not root.has_feasible_actions():
                    print("No feasible actions left! Game broken!")
                    game_broken = True
                    break

                # Make the chosen move
                try:
                    game_done, reward, move_worked = game.make_move(action)
                except Exception as e:
                    print(e)
                    print(root.children_prior_logits)
                    move_worked = False
                if not move_worked:
                    print("Game broken!")
                    game_broken = True
                    break

                if level == 0:
                    game_stats["num_level_0_moves"] += 1
                # store statistics in the history, as well as the next observations/player/level
                game_history.action_history.append(action)
                game_history.reward_history.append(reward)
                # if this is True, then the policy is stored as a OHE of the actions
                store_ohe_policy = self.config.gumbel_simple_loss
                game_history.store_gumbel_search_statistics(tree, store_ohe_policy)

                move_counter += 1
                if move_counter in self.config.log_policies_for_moves:
                    policy = [child.prior for child in root.children.values()]
                    game_stats["policies_for_selected_moves"][move_counter] = policy

                # important: shift must happen after storing search statistics
                tree.shift(action)

                # store next observation
                # If the game is done, we store the final states of the actor
                baseline_state_index = min(len(baseline_states) - 1, baseline_state_index + 1)
                game_history.observation_history.append(game.get_current_state())
                game_history.level_history.append(game.get_current_level())

                if game_done:
                    game_history.game_outcome = reward
                    game_time = time.perf_counter() - game_time
                    game_stats["id"] = self.actor_id  # identify from which actor this game came from
                    game_stats["objective"] = game.get_objective(0)
                    game_stats["explicit_npv"] = game.get_explicit_npv(0)
                    game_stats["sequence"] = game.get_sequence(0)
                    game_stats["game_time"] = game_time
                    game_stats["waiting_time"] = tree.waiting_time

        if not game_broken:
            self.n_games_played += 1
        else:
            game_history = None
            game_stats = "broken"

        if not self.config.inference_on_experience_workers:
            ray.get(self.model_inference_worker.unregister_actor.remote(self.actor_id))

        return game_history, game_stats

    def add_query_results(self, results):
        if f"rollout_{self.actor_id}" in results[0][0]:
            self.rollout_obtainer.temp_query_results = (results[1], results[2], results[3])
        else:
            self.mcts.add_query_results(results)

    def eval_mode(self):
        """
        In evaluation mode, data to evaluate is pulled from shared storage until evaluation mode is unlocked.
        """
        while ray.get(self.shared_storage.in_evaluation_mode.remote()):
            to_evaluate = ray.get(self.shared_storage.get_to_evaluate.remote())

            if to_evaluate is not None:
                # We have something to evaluate
                eval_index, instance, eval_type = to_evaluate

                env_config = copy.deepcopy(self.env_config)
                env_config.max_num_root_finding_interactions = 0
                if eval_type == "test":
                    if self.config.gumbel_test_greedy_rollout:
                        trajectory = self.rollout_obtainer.get_trajectory(instance=instance, model_type_for_policy="newcomer",
                                                                          env_config=env_config)

                        game_stats = {
                            "objective": trajectory.objective,
                            "baseline_objective": trajectory.objective
                        }
                    else:
                        _, game_stats = self.play_game(
                            problem_instance=instance, deterministic=True,
                            model_to_use_for_learning="newcomer",
                            model_to_use_for_baseline_rollout="newcomer",
                            eval_mode=True
                        )
                    self.shared_storage.push_evaluation_result.remote((eval_index, copy.deepcopy(game_stats)))
                else:
                    raise ValueError(f"Unknown eval_type {eval_type}.")
            else:
                time.sleep(1)

    def continuous_play(self, replay_buffer, logger=None):
        while not ray.get(self.shared_storage.get_info.remote("terminate")):

            if ray.get(self.shared_storage.in_evaluation_mode.remote()):
                self.eval_mode()

            game_history, game_stats = self.play_game(
                problem_instance=None,
                deterministic=False,
                model_to_use_for_learning="newcomer",
                model_to_use_for_baseline_rollout="newcomer",
                best_moves_first=False
            )

            if game_history is None:
                continue

            game_history.analyze_level_policies()

            # save game to the replay buffer and notify logger
            replay_buffer.save_game.remote(game_history, self.shared_storage)
            if logger is not None:
                logger.played_game.remote(game_stats, "train")

            if self.config.ratio_range:
                infos: Dict = ray.get(
                    self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"]))
                num_played_games = infos["num_played_games"]
                num_games_in_replay_buffer = ray.get(replay_buffer.get_length.remote())
                ratio = infos["training_step"] / max(1, num_played_games - self.config.start_train_after_episodes)

                while (ratio < self.config.ratio_range[0] and num_games_in_replay_buffer > self.config.start_train_after_episodes
                       and not infos["terminate"] and not ray.get(self.shared_storage.in_evaluation_mode.remote())
                ):
                    infos: Dict = ray.get(
                        self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                    )
                    num_games_in_replay_buffer = ray.get(replay_buffer.get_length.remote())
                    ratio = infos["training_step"] / max(1, infos["num_played_games"] - self.config.start_train_after_episodes)
                    time.sleep(0.010)  # wait for 10ms


class GameHistory:
    """
    Stores information about the moves in a game.
    """
    def __init__(self):
        # Observation is a np.array containing the state of the env of the current player and the waiting player
        self.observation_history = []
        self.level_history = []
        # i-th entry corresponds to the action the player took who was on move in i-th observation. For simultaenous
        # this is a tuple of actions.
        self.action_history = []
        # stores estimated values for root states obtained from tree search
        self.root_values = []
        # stores the action policy of the root node at i-th observation after the tree search, depending on visit
        # counts of children. Each element is a list of length number of actions on level, and sums to 1.
        self.root_policies = []
        self.reward_history = []
        self.game_outcome: Optional[float] = None  # stores the final objective

        self.learning_policies_for_level_present = [False, False, False, False, False]

    def store_gumbel_search_statistics(self, mcts: SingleplayerGumbelMCTS,
                                       for_simple_loss: bool = False):
        """
        Stores the improved policy of the root node.
        """
        root = mcts.root

        if for_simple_loss:
            # simple loss is where we assign probability one to the chosen action
            action = root.sequential_halving_chosen_action
            policy = [0.] * len(root.children)
            policy[action] = 1.
        else:
            policy = mcts.get_improved_policy(root).numpy().tolist()

        self.root_values.append(root.value())
        self.root_policies.append(policy)

    def analyze_level_policies(self):
        for level in self.level_history:
            self.learning_policies_for_level_present[level] = True