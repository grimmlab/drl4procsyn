import torch
from torch import nn
from gaz_singleplayer.config_syngame import Config
from environment.env_config import EnvConfig
from typing import List, Dict
from model.mlp_mixer_modules import MixerBlock, MixerBlockFinish


class FeedForward(nn.Module):
    """
    Simple MLP Network
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SynthesisNetwork(nn.Module):
    def __init__(self, config: Config, env_config: EnvConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.env_config = env_config

        self.device = device

        self.latent_dimension = self.config.problem_specifics["latent_dimension"]
        self.num_actions_per_level = self.env_config.num_actions_per_level

        self.situation_net_policy = SituationNetwork(config, env_config, device)
        self.situation_net_value = SituationNetwork(config, env_config, device, return_situation_vec=True)
        self.value_head = ValueHead(config, env_config, device)
        # Choose stream or terminate
        self.policy_lvl_0 = PolicyHead(config, env_config,
                                       mlp_input_dim=self.latent_dimension,
                                       mlp_hidden_dim=self.latent_dimension,
                                       num_logits=env_config.num_lines_flowsheet_matrix + 1)
        # Choose unit
        self.policy_lvl_1 = PolicyHead(config, env_config,
                                       mlp_input_dim=2 * self.latent_dimension,
                                       mlp_hidden_dim=self.latent_dimension,
                                       num_logits=env_config.num_units)
        # Choose target for mixer or recycle
        self.policy_lvl_2 = PolicyHead(config, env_config,
                                       mlp_input_dim=2 * self.latent_dimension + env_config.num_units,
                                       mlp_hidden_dim=self.latent_dimension,
                                       num_logits=env_config.num_lines_flowsheet_matrix)
        # Choose continuous parameter
        self.policy_lvl_3_and_4 = PolicyHead(config, env_config,
                                       mlp_input_dim=2 * self.latent_dimension + env_config.num_units + env_config.num_disc_steps_2b_1,
                                       mlp_hidden_dim=self.latent_dimension,
                                       num_logits=env_config.num_disc_steps_2b_1 + env_config.num_disc_steps_2b_2)

    def forward(self, x: Dict):
        transformed_sequence, _ = self.situation_net_policy(x)
        _, situation_vector = self.situation_net_value(x)

        value = self.value_head(situation_vector)

        policy_logits_list = [
            # stream
            self.policy_lvl_0(x=transformed_sequence,
                              chosen_stream_idx=None,
                              logits_mask=x["mask_lvl_zero_logits"],
                              additional_vector=None),
            # unit
            self.policy_lvl_1(x=transformed_sequence,
                              chosen_stream_idx=x["chosen_stream_idcs"],
                              logits_mask=x["mask_lvl_one_logits"],
                              additional_vector=None),
            # destination
            self.policy_lvl_2(x=transformed_sequence,
                              chosen_stream_idx=x["chosen_stream_idcs"],
                              logits_mask=x["mask_lvl_two_logits"],
                              additional_vector=x["chosen_unit"])
        ]

        # Continuous policy is factored into two levels (e.g., 0.1 * x + 0.01 * y),
        # but is predicted from the same head.
        logits_continuous = self.policy_lvl_3_and_4(
            x=transformed_sequence,
            chosen_stream_idx=x["chosen_stream_idcs"],
            logits_mask=None,
            additional_vector=torch.concat((x["chosen_unit"], x["chosen_continuous"]), dim=1)
        )

        # append logits separately for level 3 and 4 (2b_1, 2b_2)
        policy_logits_list.append(logits_continuous[:, :self.env_config.num_disc_steps_2b_1])
        policy_logits_list.append(logits_continuous[:, self.env_config.num_disc_steps_2b_1:])

        return situation_vector, value, policy_logits_list

    def set_weights(self, weights):
        if weights is not None:
            self.load_state_dict(weights)

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    @staticmethod
    def stack_batches_of_states(batch_1, batch_2):
        combined_board = dict()
        for key in batch_1:
            if torch.is_tensor(batch_1[key]):
                combined_board[key] = torch.cat(
                    (batch_1[key], batch_2[key]),
                    dim=0
                )

        return combined_board

    @staticmethod
    def states_to_batch(states: List, config: Config, env_config: EnvConfig, to_device: torch.device = None):
        if not to_device:
            to_device = torch.device("cpu")
        batch_size = len(states)
        # OHE of chosen unit (zero vector for no unit chosen): (batch_size, env_config.num_units)
        chosen_unit_batch_tensor = torch.stack(
            [torch.from_numpy(state["chosen_unit"]) for state in states], dim=0
        ).float().to(to_device)

        # OHE of first chosen continuous parameter (zero vector for not chosen yet): (batch_size, env_config.num_disc_steps_2b_1)
        chosen_continuous_batch_tensor = torch.stack(
            [torch.from_numpy(state["chosen_first_cont_interval"]) for state in states], dim=0
        ).float().to(to_device)

        # Number of stream elements to compute the average for the graph embedding
        num_streams_tensor = torch.FloatTensor([state["num_lines"] for state in states])\
            .unsqueeze(-1).to(to_device)

        # Flowsheet matrix for MLP-Mixer
        flowsheet_matrix = torch.stack(
            [torch.from_numpy(state["flowsheet_matrix"]) for state in states], dim=0
        ).float().to(to_device)

        # Indices for the chosen stream. (batch, 1) This is later used to torch.gather the corresponding sequence element from the
        # transformed sequence. If a stream has not yet been chosen, we simply set it to 0 (as the corresponding output
        # is not used)
        chosen_stream_idcs = torch.LongTensor([state["chosen_stream"] if state["chosen_stream"] is not None else 0
                                               for state in states]).unsqueeze(1).to(to_device)
        chosen_stream_idcs_for_lookup = torch.LongTensor([state["chosen_stream"] + 1 if state["chosen_stream"] is not None else 0
                                               for state in states]).to(to_device)

        max_num_lines = flowsheet_matrix.shape[1]
        # Create masks. We need
        # 1) mask for the MLP mixer, i.e. the stream sequence, as we mask all unused lines to 0.
        mask_graph_batch = torch.ones((batch_size, max_num_lines), device=to_device).float()
        # 2) mask for level 0 glimpse and final policy. This is the stream sequence + finish token
        mask_lvl_zero_policy_glimpse_batch = torch.ones((batch_size, max_num_lines + 1), device=to_device).float()
        # 3) mask for level 1. This corresponds to number of units
        mask_lvl_one_policy_batch = torch.ones((batch_size, env_config.num_units), device=to_device).float()
        # 4) mask for level 2 glimpse and final policy. This is choice of destination, so only stream sequence
        mask_lvl_two_policy_glimpse_batch = torch.ones((batch_size, max_num_lines), device=to_device).float()
        for i, state in enumerate(states):
            # 1) only set additional padded sequence elements to 0
            num_streams = state["num_lines"]
            mask_graph_batch[i, num_streams:] = 0

            # 2) set additional padded sequence elements to 0, and also infeasible stream elements
            # finally, mask the finish-token depending on the last element in feasible action
            mask_lvl_zero_policy_glimpse_batch[i, num_streams:] = 0  # also set finish-token to 0, as it gets reset anyway
            if state["action_level"] == 0:
                feasible = state["feasible_actions"]  # this is a numpy array of length num streams + 1 (for finish token)
                mask_lvl_zero_policy_glimpse_batch[i, :len(feasible) - 1] = torch.from_numpy(feasible[:-1])
                mask_lvl_zero_policy_glimpse_batch[i, -1] = feasible[-1]

            # 3) set additional padded sequence elements to 0
            mask_lvl_two_policy_glimpse_batch[i, num_streams:] = 0
            if state["action_level"] == 2:
                feasible = state["feasible_actions"]  # this is a numpy array of length num streams
                mask_lvl_two_policy_glimpse_batch[i] = torch.from_numpy(feasible)  # whoooot

            if state["action_level"] == 1:
                mask_lvl_one_policy_batch[i] = torch.from_numpy(state["feasible_actions"])

        # Current Net present value is used for the value network
        current_npv_batch = torch.FloatTensor([state["current_npv"] for state in states]).unsqueeze(1).to(to_device)

        return {
            "chosen_unit": chosen_unit_batch_tensor,  # tensor (batch, env_config.num_units)
            "chosen_continuous": chosen_continuous_batch_tensor,  # tensor (batch, env_config.num_disc_steps_2b_1)
            "num_streams": num_streams_tensor,  # tensor (batch, 1)
            "chosen_stream_idcs": chosen_stream_idcs,  # long tensor (batch, 1)
            "chosen_stream_idcs_for_lookup": chosen_stream_idcs_for_lookup,  # long tensor (batch,)
            "flowsheet_matrix": flowsheet_matrix,  # tensor (batch, max stream sequence len, 19)
            "mask_graph": mask_graph_batch.unsqueeze(-1),  # tensor (batch, max stream seq len, 1)
            "mask_lvl_zero_logits": mask_lvl_zero_policy_glimpse_batch,  # tensor (batch, max_stream seq len + 1)
            "mask_lvl_one_logits": mask_lvl_one_policy_batch,  # tensor (batch, env_config.num_units)
            "mask_lvl_two_logits": mask_lvl_two_policy_glimpse_batch,  # tensor (batch, max stream seq len)
            "current_npv": current_npv_batch,  # tensor (batch, 1)
        }

    @staticmethod
    def states_batch_dict_to_device(batch_dict: Dict, to_device: torch.device):
        for key in batch_dict:
            if torch.is_tensor(batch_dict[key]):
                batch_dict[key] = batch_dict[key].to(to_device)
        return batch_dict


class SituationNetwork(nn.Module):
    def __init__(self, config: Config, env_config: EnvConfig, device: torch.device = None, return_situation_vec = False):
        super().__init__()
        self.config = config
        self.env_config = env_config

        self.device = device

        self.latent_dimension = self.config.problem_specifics["latent_dimension"]
        self.num_mixer_blocks = self.config.problem_specifics["num_mixer_blocks"]
        self.return_situation_vec = return_situation_vec

        # Affine embedding of a line of the flowsheet matrix
        self.matrix_line_affine = nn.Linear(in_features=self.env_config.line_length_flowsheet_matrix, out_features=self.latent_dimension)

        # MLP Mixer
        mixer_blocks = []
        for _ in range(self.num_mixer_blocks):
            mixer_blocks.append(
                MixerBlock(latent_dim=self.latent_dimension,
                           num_tokens=self.env_config.num_lines_flowsheet_matrix,
                           expansion_factor=self.config.problem_specifics["expansion_factor_feature_mixing"],
                           expansion_factor_token_mixing=self.config.problem_specifics["expansion_factor_token_mixing"],
                           dropout=0.0,
                           normalization=self.config.problem_specifics["normalization"])
            )
        self.mlp_mixer = nn.ModuleList(mixer_blocks)

        if self.return_situation_vec:
            # Finish for value network
            self.situation_view = MixerBlockFinish(latent_dim=self.latent_dimension,
                                                   num_tokens=self.env_config.num_lines_flowsheet_matrix,
                                                   expansion_factor=self.config.problem_specifics["expansion_factor_feature_mixing"],
                                                   expansion_factor_token_mixing=self.config.problem_specifics["expansion_factor_token_mixing"],
                                                   dropout=0.0,
                                                   normalization=self.config.problem_specifics["normalization"])

            self.chosen_stream_lookup = nn.Embedding(num_embeddings=self.env_config.num_lines_flowsheet_matrix + 1,
                                                   embedding_dim=self.latent_dimension)

            self.situation_vector_feedforward = FeedForward(input_dim=self.latent_dimension*2 + self.env_config.num_units + self.env_config.num_disc_steps_2b_1 + 1,
                                                            hidden_dim=self.latent_dimension*2,
                                                            output_dim=self.latent_dimension)

    def forward(self, x: Dict):
        """
        Returns:
            transformed_flowsheet: [torch.Tensor] of shape (batch, num lines, latent dim), corresponding to a
                latent representation of the full flowsheet sequence
            situation_vector: [torch.Tensor] of shape (batch, latent dim), corresponding to a latent representation of
                the current state used for the value network.
        """
        # Flowsheet lines get embedded and then piped through MLP Mixer
        transformed_flowsheet = self.matrix_line_affine(x["flowsheet_matrix"])
        for mixer_block in self.mlp_mixer:
            transformed_flowsheet = mixer_block(transformed_flowsheet)

        situation_vector = None
        if self.return_situation_vec:
            _, situation_vector = self.situation_view(transformed_flowsheet)
            # append lookup of chosen stream, OHE of chosen unit and current npv
            chosen_stream_lookup_embedding = self.chosen_stream_lookup(x["chosen_stream_idcs_for_lookup"])
            situation_vector = self.situation_vector_feedforward(
                torch.concat([situation_vector, chosen_stream_lookup_embedding, x["chosen_unit"], x["chosen_continuous"], x["current_npv"]], dim=1)
            )

        return transformed_flowsheet, situation_vector


class PolicyHead(nn.Module):
    """
    General module for policies of all levels. Linearly transforms the sequence, obtains a situation vector,
    optionally gathers a certain vector from the sequence and concatenates it to the situation.
    Then uses an MLP to obtain logits.
    """
    def __init__(self, config, env_config: EnvConfig, mlp_input_dim, mlp_hidden_dim, num_logits):
        super().__init__()
        self.config = config
        self.env_config = env_config
        self.latent_dimension = config.problem_specifics["latent_dimension"]
        self.situation_view = MixerBlockFinish(latent_dim=self.latent_dimension,
                                               num_tokens=self.env_config.num_lines_flowsheet_matrix,
                                               expansion_factor=self.config.problem_specifics[
                                                   "expansion_factor_feature_mixing"],
                                               expansion_factor_token_mixing=self.config.problem_specifics[
                                                   "expansion_factor_token_mixing"],
                                               dropout=0.0,
                                               normalization=self.config.problem_specifics["normalization"])
        self.fully_connected = FeedForward(mlp_input_dim, mlp_hidden_dim, num_logits)

    def forward(self, x, chosen_stream_idx=None, logits_mask=None, additional_vector=None):
        """
        x: Sequence of shape (batch, num_tokens, latent_dim)
        chosen_stream_idx: Index of chosen stream given as long tensor of shape (batch, 1). Optional.
        logits_mask: Masking of infeasible logits, of shape (batch, num_units). Optional
        additional_vector: Additional vector to concat to situation and optional chosen stream before passing to MLP. Optional.
        """
        x, latent_rep = self.situation_view(x)
        to_concat = [latent_rep]
        if chosen_stream_idx is not None:
            # gather the chosen stream. index tensor must be of shape (batch, 1, latent dim),
            # with (i, 0, k) = chosen stream index of i-th batch element
            index_tensor = chosen_stream_idx.unsqueeze(-1).repeat(1, 1, self.latent_dimension)
            chosen_stream = torch.gather(input=x, dim=1,
                                         index=index_tensor).squeeze(dim=1)
            to_concat.append(chosen_stream)
        if additional_vector is not None:
            to_concat.append(additional_vector)

        logits = self.fully_connected(torch.cat(to_concat, dim=1))
        if logits_mask is not None:
            logits = logits + (1. - logits_mask) * -10000.

        return logits


class ValueHead(nn.Module):
    def __init__(self, config: Config, env_config: EnvConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.env_config = env_config

        self.device = device

        self.latent_dimension = self.config.problem_specifics["latent_dimension"]

        self.value_feedforward = self.value_feedforward = nn.Sequential(
            nn.Linear(self.latent_dimension, 2*self.latent_dimension),
            nn.ReLU(),
            nn.Linear(2*self.latent_dimension, 2*self.latent_dimension),
            nn.ReLU(),
            nn.Linear(2*self.latent_dimension, 1)
        )

        self.relu_clipping = None
        if self.config.objective_clipping is not None and self.config.objective_clipping[0] == 0:
            self.relu_clipping = nn.ReLU()

    def forward(self, situation_vector: torch.Tensor):
        val = self.value_feedforward(situation_vector)
        if self.relu_clipping is not None:
            val = self.relu_clipping(val)
        return val


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict
