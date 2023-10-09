import copy
import collections

import numpy as np
import ray
import torch

from gaz_singleplayer.config_syngame import Config
from gaz_singleplayer.syn_game import Game
from environment.env_config import EnvConfig
from gaz_singleplayer.synthesis_network import SynthesisNetwork

from typing import Dict, Type


@ray.remote
class ReplayBuffer:
    """
    Stores played episodes and generates batches for training the network.
    Runs in separate process, workers store their games in it asynchronously, while the
    trainer pulls batches from it.
    """
    def __init__(self, initial_checkpoint: Dict, config: Config, env_config: EnvConfig, network_class: Type[SynthesisNetwork],
                 game_class: Type[Game], prefilled_buffer: collections.deque = None):
        self.config = config
        self.env_config = env_config
        self.network_class = network_class
        self.game_class = game_class
        # copy buffer if it has been provided
        if prefilled_buffer is not None:
            self.buffer = copy.deepcopy(prefilled_buffer)
        else:
            self.buffer = collections.deque([], maxlen=self.config.replay_buffer_size)

        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]

        # total samples keeps track of number of "available" total samples in the buffer (i.e. regarding only games
        # in buffer
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer]
        )

        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with prefilled buffer: {self.total_samples} samples ({self.num_played_games} games)"
            )

        # Minimum number of available games where some level is present
        # in order for a batch so be sampled
        self.min_num_games_available = [self.config.level_based_game_stepsize] * self.env_config.num_levels

        # Fix random seed
        np.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=None):
        # Store an episode in the buffer.
        # As we are using `collections.deque, older entries get thrown out of the buffer
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if len(self.buffer) == self.config.replay_buffer_size:
            self.total_samples -= len(self.buffer[0].root_values)

        self.buffer.append(copy.deepcopy(game_history))

        if shared_storage is not None:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

        return self.num_played_games, self.num_played_steps, self.total_samples

    def get_batch(self, for_level: int = 0, for_value=False):
        value_batch = []
        policy_batch = []
        states = []

        possible_histories = [history for history in self.buffer if history.learning_policies_for_level_present[for_level]]

        if len(possible_histories) < self.min_num_games_available[for_level]:
            # not enough histories for this level. Skip.
            return None
        else:
            # Enough histories for this level. Increase the minimum number of games which need to be available for next
            # batch. Only do this when obtaining batch for value, as network trainer first calls policy, then value batch.
            if for_value:
                self.min_num_games_available[for_level] = min(
                    self.min_num_games_available[for_level] + self.config.level_based_game_stepsize,
                    self.config.level_based_game_step_upper_limit
                )

        game_histories = np.random.choice(possible_histories, size=self.config.batch_size)

        for batch_idx, game_history in enumerate(game_histories):
            max_index = len(game_history.action_history) if not for_value else len(game_history.observation_history)
            possible_positions = [i for i in range(max_index)
                                  if game_history.level_history[i] == for_level and (for_value or len(game_history.root_policies[i]) >= 2)]

            game_position = np.random.choice(possible_positions)
            target_value, target_policy = self.make_target(game_history, game_position, for_value)

            state = copy.deepcopy(game_history.observation_history[game_position])
            states.append(state)

            value_batch.append(target_value)
            policy_batch.append(target_policy)

        states_batch = self.network_class.states_to_batch(states, config=self.config, env_config=self.env_config)

        if for_value:
            value_batch_tensor = torch.cat(value_batch, dim=0)
            return (
                states_batch,
                None,
                value_batch_tensor,  # (batch_size, 1)
                None,  # Padded policies of shape (batch_size, <max policy length in batch>)
                None
            )

        # pad the policies to maximum length in batch
        policy_lengths = [policy.shape[1] for policy in policy_batch]
        policy_averaging_tensor = torch.tensor(policy_lengths).float().unsqueeze(-1)
        policy_batch_tensor = torch.cat(policy_batch, dim=0)

        return (
            states_batch,  # List of canonical boards
            None,
            None,  # (batch_size, 1)
            policy_batch_tensor,  # Padded policies of shape (batch_size, <max policy length in batch>)
            policy_averaging_tensor
        )

    def get_length(self):
        return len(self.buffer)

    def make_target(self, game_history, state_index: int, for_value: bool):
        """
        Generates targets (value and policy) for each observation.

        Parameters
            game_history: Episode history
            state_index [int]: Position in game to sample
        Returns:
            target_value: Float Tensor of shape (1, 1)
            target_policy: Tensor of shape (1, policy length)
        """
        if for_value:
            value = self.singleplayer_value(game_history, state_index)
            target_value = torch.FloatTensor([value]).unsqueeze(0)
            return target_value, None

        policy = copy.deepcopy(game_history.root_policies[state_index])
        target_policy = torch.FloatTensor(policy).unsqueeze(0)

        return None, target_policy

    def singleplayer_value(self, game_history, state_index: int):
        if self.config.singleplayer_options["bootstrap_final_objective"]:
            return game_history.game_outcome
        else:
            bootstrap_n_steps = self.config.singleplayer_options["bootstrap_n_steps"]
            # The value target is the discounted root value of the search tree td_steps into
            # the future, plus the discounted sum of all rewards until then.
            if bootstrap_n_steps == -1:
                # sum up rewards until end
                bootstrap_index = len(game_history.root_values)
            else:
                bootstrap_index = min(state_index + self.config.singleplayer_options["bootstrap_n_steps"], len(game_history.root_values))

            value = game_history.root_values[bootstrap_index] if bootstrap_index < len(game_history.root_values) else 0
            value += sum(game_history.reward_history[state_index: bootstrap_index])
            return value

    def get_buffer(self):
        return self.buffer
