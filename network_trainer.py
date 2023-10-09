import torch
import os
import numpy as np
import ray
import time
import copy

from environment.env_config import EnvConfig
from gaz_singleplayer.config_syngame import Config
from gaz_singleplayer.synthesis_network import SynthesisNetwork

from typing import Dict, Type, Optional

from shared_storage import SharedStorage


@ray.remote
class NetworkTrainer:
    """
    One instance of this class runs in a separate process, continuously training the network using the
    experience sampled from the playing actors and saving the weights to the shared storage.
    """
    def __init__(self, config: Config, env_config: EnvConfig, shared_storage: SharedStorage, network_class: Type[SynthesisNetwork],
                 initial_checkpoint: Dict = None, device: torch.device = None):
        self.config = config
        self.env_config = env_config
        self.device = device if device else torch.device("cpu")
        self.shared_storage = shared_storage

        if self.config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.CUDA_VISIBLE_DEVICES

        # Fix random generator seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network. best/newcomer is legacy code
        self.model = network_class(config, env_config, device=self.device)
        if initial_checkpoint["weights_newcomer"] is not None:
            self.model.set_weights(copy.deepcopy(initial_checkpoint["weights_newcomer"]))
        else:
            # If we do not have an initial checkpoint, we set the random weights both to 'newcomer' and 'best'.
            print("Setting identical random weights to 'newcomer' and 'best' model...")
            self.shared_storage.set_info.remote({
                "weights_timestamp_newcomer": round(time.time() * 1000),
                "weights_timestamp_best": round(time.time() * 1000),
                "weights_newcomer": copy.deepcopy(self.model.get_weights()),
                "weights_best": copy.deepcopy(self.model.get_weights())
            })

        self.model.to(self.device)
        self.model.train()

        self.training_step = initial_checkpoint["training_step"] if initial_checkpoint else 0

        if "cuda" not in str(next(self.model.parameters()).device):
            print("NOTE: You are not training on GPU.\n")

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_init,
            weight_decay=self.config.weight_decay
        )

        # Load optimizer state if available
        if initial_checkpoint is not None and initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

        print(f"Successfully set up network trainer on device {self.device}")

    def continuous_update_weights(self, replay_buffer, logger=None):
        """
        Continuously samples batches from the replay buffer for each level, make an optimization step, repeat...
        """
        # Wait for replay buffer to contain at least a certain number of games.
        while ray.get(replay_buffer.get_length.remote()) < max(1, self.config.start_train_after_episodes):
            time.sleep(1)

        batches_for_levels = [(
            replay_buffer.get_batch.remote(level, for_value=False),
            replay_buffer.get_batch.remote(level, for_value=True)
        ) for level in range(self.env_config.num_levels)]

        # Main training loop
        while not ray.get(self.shared_storage.get_info.remote("terminate")):
            # If we should pause, sleep for a while and then continue
            if ray.get(self.shared_storage.in_evaluation_mode.remote()):
                time.sleep(1)
                continue

            # perform exponential learning rate decay based on training steps. See config for more info
            self.update_lr()

            # full loss over all levels
            full_loss: Optional[torch.Tensor] = None
            full_policy_loss: Optional[torch.Tensor] = None
            full_value_loss: Optional[torch.Tensor] = None
            num_levels_inferred = 0.
            for level in range(self.env_config.num_levels):
                batch_policy = ray.get(batches_for_levels[level][0])
                batch_value = ray.get(batches_for_levels[level][1])

                if batch_policy is None or batch_value is None:
                    continue

                num_levels_inferred += 1
                # loss for this batch
                policy_loss = self.get_loss(batch_policy, level=level, for_value=False)

                # THIS IS A TEST ONLY
                p_loss = policy_loss.mean()
                self.optimizer.zero_grad()
                p_loss.backward()

                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clipping)

                self.optimizer.step()

                value_loss = self.get_loss(batch_value, level=level, for_value=True)

                v_loss = value_loss.mean()
                self.optimizer.zero_grad()
                v_loss.backward()

                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clipping)

                self.optimizer.step()

                if full_loss is None:
                    full_value_loss = value_loss.mean().item()
                    full_policy_loss = policy_loss.mean().item()
                    full_loss = full_value_loss + full_policy_loss
                else:
                    full_value_loss += value_loss.mean().item()
                    full_policy_loss += policy_loss.mean().item()
                    full_loss = full_loss + value_loss.mean().item() + policy_loss.mean().item()

            # already prepare next batch in the replay buffer worker, so we minimize waiting times
            batches_for_levels = [(
                replay_buffer.get_batch.remote(level, for_value=False),
                replay_buffer.get_batch.remote(level, for_value=True)
            ) for level in range(self.env_config.num_levels)]

            self.training_step += 1

            full_loss /= num_levels_inferred
            full_value_loss /= num_levels_inferred
            full_policy_loss /= num_levels_inferred

            # Save model to shared storage so it can be used by the actors
            if self.training_step % self.config.checkpoint_interval == 0:
                self.shared_storage.set_info.remote({
                    "weights_timestamp_newcomer": round(time.time() * 1000),
                    "weights_newcomer": copy.deepcopy(self.model.get_weights()),
                    "optimizer_state": copy.deepcopy(
                        dict_to_cpu(self.optimizer.state_dict())
                    )
                })

            # Send results to logger
            if logger is not None:
                logger.training_step.remote({
                    "loss": float(full_loss), "value_loss": float(full_value_loss), "policy_loss": float(full_policy_loss)
                })

            # Inform shared storage of training step. We do this at the end so there are no conflicts with
            # the arena mode.
            self.shared_storage.set_info.remote({
                "training_step": self.training_step
            })

            # Managing the episode / training ratio

            if self.config.ratio_range:
                infos: Dict = ray.get(
                    self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"]))
                ratio = infos["training_step"] / max(1, infos["num_played_games"] - self.config.start_train_after_episodes)

                while (ratio > self.config.ratio_range[1]
                       and not infos["terminate"] and not ray.get(self.shared_storage.in_evaluation_mode.remote())
                ):
                    infos: Dict = ray.get(
                        self.shared_storage.get_info.remote(["training_step", "num_played_games", "terminate"])
                    )
                    ratio = infos["training_step"] / max(1, infos["num_played_games"] - self.config.start_train_after_episodes)
                    #print(infos["training_step"], max(1, infos["num_played_games"]), ratio)
                    time.sleep(0.010)  # wait for 10ms

    def get_loss(self, batch, level: int, for_value=False):
        """
        Parameters:
            for_value [bool]: If True, value loss is returned, else policy loss.

        Returns:
            [torch.Tensor] of shape (batch size,)
        """
        (
            state_batch,  # observation
            _,  # Legacy.
            value_batch_tensor,  # (batch_size, 1)
            policy_batch_tensor,  # (batch_size, <max policy length in batch>)
            policy_averaging_tensor
        ) = batch

        # Send everything to device
        state_batch = self.model.states_batch_dict_to_device(state_batch, self.device)

        # Generate predictions
        if for_value:
            transformed_sequence, situation_vector_batch = self.model.situation_net_value(state_batch)
        else:
            transformed_sequence, situation_vector_batch = self.model.situation_net_policy(state_batch)

        policy_logits_padded = None
        predicted_value_batch_tensor = None

        if for_value:
            value_batch_tensor = value_batch_tensor.to(self.device)
            predicted_value_batch_tensor = self.model.value_head(situation_vector_batch)
        else:
            policy_batch_tensor = policy_batch_tensor.to(self.device)
            policy_averaging_tensor = policy_averaging_tensor.to(self.device)
            if level == 0:
                logits = self.model.policy_lvl_0(
                              x=transformed_sequence,
                              chosen_stream_idx=None,
                              logits_mask=None,  # Do not mask so network learns to exclude non-feasible actions
                              additional_vector=None)
            elif level == 1:
                logits = self.model.policy_lvl_1(x=transformed_sequence,
                                                 chosen_stream_idx=state_batch["chosen_stream_idcs"],
                                                 logits_mask=None,
                                                 additional_vector=None)
            elif level == 2:
                logits = self.model.policy_lvl_2(x=transformed_sequence,
                                                 chosen_stream_idx=state_batch["chosen_stream_idcs"],
                                                 logits_mask=None,
                                                 additional_vector=state_batch["chosen_unit"])
            elif level == 3 or level == 4:
                logits = self.model.policy_lvl_3_and_4(x=transformed_sequence,
                                                 chosen_stream_idx=state_batch["chosen_stream_idcs"],
                                                 logits_mask=None,
                                                 additional_vector=torch.concat((state_batch["chosen_unit"], state_batch["chosen_continuous"]), dim=1)
                                                )
                logits = logits[:, :self.env_config.num_disc_steps_2b_1] if level == 3 else logits[:, self.env_config.num_disc_steps_2b_1:]

            policy_logits_padded = logits
        # Compute loss for each step
        value_loss, policy_loss = self.loss_function(
            predicted_value_batch_tensor, policy_logits_padded,
            value_batch_tensor, policy_batch_tensor, policy_averaging_tensor,
            use_kl=not self.config.gumbel_simple_loss,
            average_policy=self.config.average_policy_loss_elementwise,
        )

        # Scale value loss
        if for_value:
            return value_loss
        else:
            return policy_loss

    @staticmethod
    def loss_function(value, policy_logits_padded, target_value, target_policy_tensor, policy_averaging_tensor,
                      use_kl=False, average_policy=True):
        """
        Parameters
            value: Tensor of shape (batch_size, 1)
            policy_logits_padded: Policy logits which are padded to have the same size.
                Tensor of shape (batch_size, <maximum policy size>)
            target_value: Tensor of shape (batch_size, 1)
            target_policy_tensor: Tensor of shape (batch_size, <max policy len in batch>)

        Returns
            value_loss, policy_loss
        """
        value_loss = torch.square(value - target_value).sum(dim=1) if value is not None else None
        if policy_logits_padded is None:
            return value_loss, None

        # Apply log softmax to the policy, and mask the padded values to 0.
        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_softmax_policy_masked = log_softmax(policy_logits_padded)

        if not use_kl:
            # Cross entropy loss between target distribution and predicted one
            policy_loss = torch.sum(- target_policy_tensor * log_softmax_policy_masked, dim=1)
        else:
            # Kullback-Leibler
            kl_loss = torch.nn.KLDivLoss(reduction='none')
            policy_loss = kl_loss(log_softmax_policy_masked, target_policy_tensor)
            policy_loss = torch.sum(policy_loss, dim=1)

        # Average policy loss element-wise by length of individual policies
        if average_policy:
            policy_loss = torch.div(policy_loss.unsqueeze(-1), policy_averaging_tensor)

        return value_loss, policy_loss

    def update_lr(self):
        """
        Update learning rate with an exponential scheme.
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (self.training_step / self.config.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


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
