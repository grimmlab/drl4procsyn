import torch
import ray
from typing import List, Dict
from gaz_singleplayer.config_syngame import Config
from shared_storage import SharedStorage
from copy import deepcopy
from environment.env_config import EnvConfig
import time


class LocalInferencer:
    def __init__(self, config: Config,
                 env_config: EnvConfig,
                 shared_storage: SharedStorage,
                 network_class,
                 model_named_keys: List[str],
                 initial_checkpoint: Dict = None,
                 device=None):
        self.config = config
        self.env_config = env_config
        self.network_class = network_class
        self.device = device if device is not None else torch.device("cpu")

        self.shared_storage = shared_storage

        # build up models and timestamp
        self.models = dict()
        self.last_checked_for_model = dict()
        self.model_weights_timestamp = dict()
        # for each model key have separate queues
        self.batch = dict()
        self.query_ids = dict()

        for key in model_named_keys:
            self.models[key] = self.network_class(config, env_config, self.device).to(self.device)
            self.model_weights_timestamp[key] = 0
            self.last_checked_for_model[key] = time.time()

            if initial_checkpoint is not None:
                self.models[key].set_weights(deepcopy(initial_checkpoint[f"weights_{key}"]))
                self.model_weights_timestamp[key] = initial_checkpoint[f"weights_timestamp_{key}"]
            else:
                self.set_latest_model_weights(key)

            self.models[key].eval()

    def set_latest_model_weights(self, model_key: str):
        # get the timestamp of the latest model weights and compare it to ours to see if we need to update
        latest_weights_timestamp = ray.get(self.shared_storage.get_info.remote(f"weights_timestamp_{model_key}"))

        if latest_weights_timestamp > self.model_weights_timestamp[model_key]:
            method = self.shared_storage.get_info.remote([f"weights_{model_key}", f"weights_timestamp_{model_key}"])
            info: Dict = ray.get(method)
            self.models[model_key].set_weights(weights=info[f"weights_{model_key}"])
            self.model_weights_timestamp[model_key] = info[f"weights_timestamp_{model_key}"]

    def infer_batch(self, batch, model_key: str):
        with torch.no_grad():
            current_time = time.time()
            # Check if we need to poll for latest model. Saves time if we don't do this all the time
            if current_time - self.last_checked_for_model[model_key] > self.config.check_for_new_model_every_n_seconds:
                self.last_checked_for_model[model_key] = current_time
                # Get the latest weights
                if self.shared_storage:
                    self.set_latest_model_weights(model_key)

            model = self.models[model_key]

            batch_dict = model.states_to_batch(batch, self.config, self.env_config, to_device=self.device)
            situation_vector_batch, value_batch, policy_logits_batch_level_list = model(batch_dict)
            policy_logits_batch_level_list = [policy_logits_batch_level.cpu().numpy()
                                              for policy_logits_batch_level in
                                              policy_logits_batch_level_list]
            value_batch = value_batch.cpu().numpy()
            return situation_vector_batch.cpu(), policy_logits_batch_level_list, value_batch