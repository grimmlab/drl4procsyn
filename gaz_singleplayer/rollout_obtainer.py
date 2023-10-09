import time
from typing import Type, Union, List, Optional, Tuple

import numpy as np
import ray
import torch
from gaz_singleplayer.config_syngame import Config
from gaz_singleplayer.syn_game import Game
from local_inferencer import LocalInferencer
from environment.env_config import EnvConfig
from inferencer import ModelInferencer


class Trajectory:
    def __init__(self):
        self.current_game_state: Optional[Game] = None
        self.current_action_level: int = 0
        self.current_policy_logits: Optional[np.array] = None
        self.current_value = None
        self.raw_state_sequence: List = []
        self.latent_state_sequence: List = []
        self.log_sequence_probability: float = 0.
        self.accumulated_reward = 0.
        self.objective = None
        self.finished = False
        self.broken = False

        # if this is True, first policy and latent state must be set before once can transition with
        # the game state
        self.needs_policy_and_latent: bool = True

    @classmethod
    def from_initial_game_state(cls, game: Game):
        trajectory = cls()
        trajectory.current_game_state = game.copy()
        trajectory.current_action_level = game.get_current_level()
        trajectory.raw_state_sequence = [game.get_current_state()]
        return trajectory

    def create_by_action_transition(self, action: int):
        """
        Returns a new trajectory representing `self` plus action transition when taking `action`.
        """
        # make sure that we have correctly updated logits and can perform an
        # action on this trajectory
        assert self.needs_policy_and_latent is False
        assert self.broken is False
        assert self.finished is False
        assert self.current_policy_logits[action] != float("-inf")

        trajectory = Trajectory()
        trajectory.current_game_state = self.current_game_state.copy()
        game_done, reward, move_worked = trajectory.current_game_state.make_move(action)
        if not move_worked:
            # very rare case that the move converges in some conditions, and when tried again, it no longer does.
            # we simply skip it due to instability
            return None

        assert move_worked is True

        trajectory.accumulated_reward = self.accumulated_reward + reward
        trajectory.finished = game_done
        if trajectory.finished:
            trajectory.objective = trajectory.current_game_state.get_objective(0)
        trajectory.raw_state_sequence = self.raw_state_sequence + [trajectory.current_game_state.get_current_state()]
        trajectory.latent_state_sequence = self.latent_state_sequence.copy()
        trajectory.current_action_level = trajectory.current_game_state.get_current_level()

        policy_prob_logs = torch.log_softmax(torch.from_numpy(self.current_policy_logits), dim=0)

        trajectory.log_sequence_probability = self.log_sequence_probability + policy_prob_logs[action].item()
        return trajectory

    def set_state_evaluation_results(self, policy_logits: Optional[np.array], latent_state: Optional[torch.Tensor], value: Optional[float], do_copy: bool = True):
        if policy_logits is not None:
            self.current_policy_logits = policy_logits.copy() if do_copy else policy_logits

        self.latent_state_sequence.append(latent_state)
        self.current_value = value
        self.needs_policy_and_latent = False

    def get_current_number_of_actions(self):
        return len(self.current_game_state.get_feasible_actions_ohe_vector())

    def select_k_highest_feasible_actions(self, k: int, sampled: bool = False):
        """
        Obtains a maximum of k feasible actions in the policy logits. Pre-tests the actions, and if an action is not
        feasible, the next highest ones are tried until either all actions turn out to be infeasible (in this case, `None`
        is returned), or a minimum of 1 action remains (with maximum of k).

        Returns:
            action_idcs: List[int]. If empty list, all actions are found to be infeasible
        """
        feasible_actions_ohe_vector = self.current_game_state.get_feasible_actions_ohe_vector()
        num_all_actions = len(feasible_actions_ohe_vector)
        # Set known infeasible actions to -inf
        self.current_policy_logits[feasible_actions_ohe_vector == 0] = float("-inf")
        # make a copy of policy logits in which we later set all visited actions to -inf
        temp_logits = self.current_policy_logits.copy()
        if sampled:
            # Add Gumbels for sampling without replacement
            temp_logits += np.random.gumbel(size=len(temp_logits))

        action_idcs = []

        while True:
            if len(action_idcs) == min(k, num_all_actions):
                break

            # Get the action with highest logit. If it is infeasible, we break, because it means
            # that all actions are infeasible.
            highest_action_idx = np.argmax(temp_logits)
            if temp_logits[highest_action_idx] == float("-inf"):
                break

            # otherwise we make a copy of the current game and try the move.
            temp_game = self.current_game_state.copy()
            _, _, move_worked = temp_game.make_move(highest_action_idx)
            if not move_worked:
                self.current_policy_logits[highest_action_idx] = float("-inf")
            else:
                action_idcs.append(highest_action_idx)
            # in all cases set the visited action to -inf, indicating that we already tried it.
            temp_logits[highest_action_idx] = float("-inf")

        if not len(action_idcs):
            self.broken = True

        return action_idcs

    def selfcheck_finished_consistency(self):
        return self.finished and not self.broken \
               and len(self.latent_state_sequence) == len(self.raw_state_sequence) \
               and not self.needs_policy_and_latent and self.objective is not None


class RolloutObtainer:
    def __init__(self, config: Config, env_config: EnvConfig, actor_id: int,
                 game_class: Type[Game], inferencer: Union[LocalInferencer, ModelInferencer]):
        self.config = config
        self.env_config = env_config
        self.actor_id = actor_id
        self.game_class = game_class
        self.inferencer = inferencer

        # stores query results from inferencer
        self.temp_query_results: Optional[Tuple] = None

    def get_trajectory(self, instance, model_type_for_policy: str, config=None, env_config=None) -> Trajectory:
        """
        Parameters:
            instance: Problem instance
            model_type_for_policy: [str] Which model to use for obtaining policy. Value and latent
                states are always obtained with "newcomer" model.

        Returns:
            states: List of raw states of the trajectory. (t-th item indicating state at timestep t)
            latent_states: List of latent representation of the states of the trajectory.
            reward: [float] Final reward from this trajectory
            objective: [float] The objective coming from this trajectory (may be different to reward)
        """
        if config is None:
            config = self.config
        if env_config is None:
            env_config = self.env_config
        game = self.game_class(config=config, flowsheet_simulation_config=env_config, problem_instance=instance)

        if not self.config.inference_on_experience_workers:
            ray.get(self.inferencer.register_actor.remote(self.actor_id))

        with torch.no_grad():
            # Greedy or sampled trajectory.
            trajectory = simple_trajectory = \
                self._beam_search_trajectory(initial_game=game.copy(),
                                             model_type_for_policy=model_type_for_policy,
                                             beam_width=1,
                                             beam_selection_method="sequence_probability",
                                             max_finished_trajectories=1,
                                             action_selection_sampled=not self.config.simple_rollout_greedy)
            if trajectory is not None:
                assert simple_trajectory.selfcheck_finished_consistency()

            if self.config.include_probability_based_beam_search:
                # Beam search based on sequence probability
                prob_based_trajectory = \
                    self._beam_search_trajectory(initial_game=game.copy(),
                                                 model_type_for_policy=model_type_for_policy,
                                                 beam_width=self.config.beam_search_width,
                                                 beam_selection_method="sequence_probability",
                                                 max_finished_trajectories=self.config.max_num_finished_trajectories_in_beam_search)

                if prob_based_trajectory is None:
                    print("Prob based beam search is None! Can this really happen?!")
                if prob_based_trajectory is not None:
                    assert prob_based_trajectory.selfcheck_finished_consistency()
                    if (trajectory is None) or (
                            trajectory is not None and prob_based_trajectory.objective > trajectory.objective):
                        trajectory = prob_based_trajectory

            # we can use simple value based beam-search
            if self.config.include_value_based_beam_search:
                value_based_trajectory = \
                    self._beam_search_trajectory(initial_game=game.copy(),
                                                 model_type_for_policy=model_type_for_policy,
                                                 beam_width=self.config.beam_search_width,
                                                 beam_selection_method="value",
                                                 max_finished_trajectories=self.config.max_num_finished_trajectories_in_beam_search)
    #
                if value_based_trajectory is not None:
                    assert value_based_trajectory.selfcheck_finished_consistency()
                    if (trajectory is None) or (trajectory is not None and value_based_trajectory.objective > trajectory.objective):
                        trajectory = value_based_trajectory

        if not self.config.inference_on_experience_workers:
            ray.get(self.inferencer.unregister_actor.remote(self.actor_id))

        return trajectory

    def _beam_search_trajectory(self, initial_game: Game, model_type_for_policy: str,
                                beam_width: int,
                                beam_selection_method: str,
                                max_finished_trajectories: int,
                                action_selection_sampled: bool = False):
        """
        Performs beam search with a policy network and chooses sequences which maximize the sequence probability
        (as in NLP). As in general, trajectories do not have identical length and
        an episode can be arbitrarily long, `max_trajectories` specifies number
        of maximum finished trajectories until the best one is returned.

        Parameters:
            initial_game [Game]: Game object with initial state
            model_type_for_policy [str]: "best" or "newcomer" indicating which policy network should be used.
            beam_width [int]: Number of sequences to consider in beam search.
            beam_selection_method [str]: Way to choose the best beams from a set of beams. Possibilities:
                "sequence_probability", "value"
            max_finished_trajectories [int]: Maximum number of finished trajectories until beam search finishes early.
        """
        trajectory = Trajectory.from_initial_game_state(game=initial_game)

        considered_trajectories = [trajectory]  # keeps the currently active and considered trajectories
        finished_trajectories = []  # keeps finished trajectories

        while True:
            # Step 1: Get current state evaluation for each considered trajectory in beam

            temp_considered_trajectories = []  # store temporarily all trajectories here
            temp_finished_trajectories = []  # store temporarily finished trajectories which still need a final state evaluation

            # prune trajectories if their number of moves is exceeded
            _c = []
            for trajectory in considered_trajectories:
                num_lvl_zero = trajectory.current_game_state.get_number_of_lvl_zero_moves()
                if num_lvl_zero > self.env_config.max_steps_for_flowsheet_synthesis:
                    continue
                _c.append(trajectory)

            considered_trajectories = _c

            self.evaluate_trajectories(considered_trajectories, model_type_for_policy)
            if beam_selection_method == "value":
                # If we do value based beam search, we trim the number of trajectories right after the evaluation
                considered_trajectories = self.select_best_trajectories(considered_trajectories,
                                                                        selection_type=beam_selection_method,
                                                                        k=beam_width,
                                                                        )

            for i, trajectory in enumerate(considered_trajectories):
                actions = trajectory.select_k_highest_feasible_actions(k=beam_width, sampled=action_selection_sampled)
                if not len(actions):
                    # trajectory is broken, and we simply do not consider it anymore
                    continue

                # otherwise, for each action, we obtain a new temp considered trajectory
                for action in actions:
                    new_trajectory = trajectory.create_by_action_transition(action=action)
                    if new_trajectory is None:
                        continue
                    # if the trajectory is finished, we add it to the temp finished ones
                    if new_trajectory.finished:
                        temp_finished_trajectories.append(new_trajectory)
                    else:
                        temp_considered_trajectories.append(new_trajectory)

            # First, for all newly finished trajectories, obtain final state evaluation
            if len(temp_finished_trajectories):
                self.evaluate_trajectories(temp_finished_trajectories, model_type_for_policy, is_finished_only=True)
                finished_trajectories += temp_finished_trajectories  # add newly finished trajectories

            # Check if we have enough finished trajectories or there are no more trajectories to consider in next step
            if len(finished_trajectories) >= max_finished_trajectories or len(temp_considered_trajectories) == 0:
                break

            # Otherwise we reduce the number of considered trajectories down to beam_width
            if beam_selection_method in ["value"]:
                considered_trajectories = temp_considered_trajectories
            else:
                # probability based, we select select best trajectories directly here
                considered_trajectories = self.select_best_trajectories(temp_considered_trajectories,
                                                                        selection_type=beam_selection_method,
                                                                        k=beam_width)

        if not len(finished_trajectories):
            return None

        return self.select_best_trajectories(finished_trajectories, selection_type="objective", k=1)[0]

    def select_best_trajectories(self, trajectories: List[Trajectory], selection_type: str, k: int) -> List[Trajectory]:
        """
        Returns from a list of trajectories a maximum of k trajectories with the highest attribute given by
        `selection_type`, which can be "sequence_probability" or "value" or "objective"
        """
        def prob_based(traj: Trajectory):
            return traj.log_sequence_probability

        def value_based(traj: Trajectory):
            return traj.current_value

        def objective_based(traj: Trajectory):
            return traj.objective

        method_map = {
            "sequence_probability": prob_based,
            "value": value_based,
            "objective": objective_based
        }

        sorted_trajs = sorted(trajectories,
                              reverse=True,
                              key=method_map[selection_type])
        return sorted_trajs[:k]

    def evaluate_trajectories(self, trajectories: List[Trajectory], model_type_for_policy: str, is_finished_only: bool = False):
        batch = {
            "states": [trajectory.raw_state_sequence[-1] for trajectory in trajectories],
            "num_actions": [trajectory.get_current_number_of_actions() for trajectory in trajectories],
            "levels": [trajectory.current_action_level for trajectory in trajectories]
        }
        if not is_finished_only:
            policy_logits_list, values, latent_states = self._evaluate_states(states=batch["states"],
                                                                                  num_actions=batch["num_actions"],
                                                                                  policy_levels=batch["levels"],
                                                                                  model_type=model_type_for_policy)
        if model_type_for_policy != "newcomer" or is_finished_only:
            _, values, latent_states = self._evaluate_states(states=batch["states"],
                                                             num_actions=batch["num_actions"],
                                                             policy_levels=batch["levels"],
                                                             model_type="newcomer")

        for i, trajectory in enumerate(trajectories):
            trajectory.set_state_evaluation_results(
                policy_logits=policy_logits_list[i] if not is_finished_only else None,
                latent_state=latent_states[i],
                value=values[i],
                do_copy=not self.config.inference_on_experience_workers
            )

    def _evaluate_states(self, states: List, num_actions: List[int], policy_levels: List[int], model_type: str):
        """
        Parameters:
            states: List of states, where each element is a tuple
            num_actions: List[int] Number of actions for each state. Used to unpad policies.
            policy_levels: List[int] Action level of syn game for each state
            model_type: [str] "Best" or "newcomer", indicates which model to use.

        Returns:
            policies: List[[np.array]] Unpadded policy logits
            values: List[float] Values of states
            latent_states: List[torch.Tensor] Latent representation of states.
        """
        if not self.config.inference_on_experience_workers:
            # send to remote inferencer
            ray.get(self.inferencer.add_list_to_queue.remote(
                actor_id=self.actor_id,
                query_ids={model_type: [f"rollout_{self.actor_id}_{i}" for i in range(len(states))]},
                # add `None` if no statewise comparison is needed
                query_states={model_type: states},
                model_keys=[model_type]
            ))

            while self.temp_query_results is None:
                time.sleep(0)
        else:
            self.temp_query_results = self.inferencer.infer_batch(states, model_key=model_type)

        situation_vector_tensor: torch.Tensor = self.temp_query_results[0]
        policy_logits_for_levels: List[np.array] = self.temp_query_results[1]
        values: Optional[np.array] = self.temp_query_results[2]
        self.temp_query_results = None

        situation_vectors = [situation_vector_tensor[i] for i in range(len(states))]
        policy_logits = []
        for i in range(len(states)):
            level = policy_levels[i]
            if level == 0:
                # logits are given as [logit stream 1, ..., logit stream k, padding 1, ..., padding l, logit finished]
                logits = np.concatenate((policy_logits_for_levels[level][i, :num_actions[i] - 1], policy_logits_for_levels[level][i, -1:]))
            else:
                logits = policy_logits_for_levels[level][i, :num_actions[i]]
            policy_logits.append(logits)

        values = [None] * len(states) if values is None else list(values)

        return policy_logits, values, situation_vectors



