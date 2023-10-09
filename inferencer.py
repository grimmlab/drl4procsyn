import os
import torch
import numpy as np
import ray
import sys
import psutil
from typing import List, Dict
from gaz_singleplayer.config_syngame import Config
from shared_storage import SharedStorage
from environment.env_config import EnvConfig
from copy import deepcopy
import time
from threading import Lock


@ray.remote
class ModelInferencer:
    """
    Continuously pulls queries which have been submitted from MCTS threads and performs inference on given device.
    """
    def __init__(self, config: Config, env_config: EnvConfig, shared_storage: SharedStorage,
                 network_class,
                 model_named_keys: List[str], device: torch.device = None,
                 initial_checkpoint: Dict = None, random_seed: int = 42, cpu_core: int = None):
        if config.pin_workers_to_core and sys.platform == "linux" and cpu_core is not None:
            psutil.Process().cpu_affinity([cpu_core])

        self.config = config
        self.env_config = env_config
        self.device = device if device else torch.device("cpu")
        self.network_class = network_class

        if self.config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.CUDA_VISIBLE_DEVICES

        # Set the random seed for initial model weights
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.shared_storage = shared_storage

        # build up models and timestamp
        # An inferencer can hold multiple models which are identified with keys provided by `model_named_keys`
        print(f"Performing MCTS state inference on device: {self.device}")

        self.models = dict()
        self.last_checked_for_model = dict()
        self.model_weights_timestamp = dict()
        # for each model key have separate queues
        self.batch = dict()
        self.query_ids = dict()
        self.actor_returns = dict()
        self.last_added_to_queue = dict()

        for key in model_named_keys:
            self.models[key] = self.network_class(config, env_config, device=self.device)
            self.models[key].to(self.device)
            self.model_weights_timestamp[key] = 0
            self.last_checked_for_model[key] = time.time()

            if initial_checkpoint is not None:
                self.models[key].set_weights(deepcopy(initial_checkpoint[f"weights_{key}"]))
                self.model_weights_timestamp[key] = initial_checkpoint[f"weights_timestamp_{key}"]

            self.models[key].eval()
            self.batch[key] = []
            self.query_ids[key] = []
            self.actor_returns[key] = []
            self.last_added_to_queue[key] = 0.

        self.model_time = {
            "full": 0.,
            "batching": 0.,
            "model": 0.
        }

        self.registering_lock = Lock()
        self.registered_actors = []
        self.data_from_registered_actors = dict()

        self.tasks = []

    def set_latest_model_weights(self, model_key: str):
        # get the timestamp of the latest model weights and compare it to ours to see if we need to update
        latest_weights_timestamp = ray.get(self.shared_storage.get_info.remote(f"weights_timestamp_{model_key}"))

        if latest_weights_timestamp > self.model_weights_timestamp[model_key]:
            method = self.shared_storage.get_info.remote([f"weights_{model_key}", f"weights_timestamp_{model_key}"])
            info: Dict = ray.get(method)
            self.models[model_key].set_weights(weights=info[f"weights_{model_key}"])
            self.model_weights_timestamp[model_key] = info[f"weights_timestamp_{model_key}"]

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def continuous_inference(self):
        with torch.no_grad():
            while True:
                current_time = time.time()

                for key in self.models:
                    # Check if we need to poll for latest model. Saves time if we don't do this all the time
                    if current_time - self.last_checked_for_model[key] > self.config.check_for_new_model_every_n_seconds:
                        self.last_checked_for_model[key] = current_time
                        # Get the latest weights
                        if self.shared_storage:
                            self.set_latest_model_weights(key)

                # Check if we have data from all registered actors. If yes, perform inference for all models

                with self.registering_lock:
                    all_present = True
                    for actor in self.registered_actors:
                        if not self.data_from_registered_actors[actor]:
                            all_present = False
                            break
                if not all_present:
                    time.sleep(0)

                if all_present and len(self.registered_actors) > 0:
                    with self.registering_lock:

                        for key in self.models:
                            if not len(self.batch[key]):
                                continue
                            # monitor how much time we are spending for inference
                            model_time = time.perf_counter()

                            batch = self.batch[key]
                            model = self.models[key]

                            # Concatenate batch and pipe through model
                            batch_dict = model.states_to_batch(batch, self.config, self.env_config, self.device)

                            self.model_time["batching"] += time.perf_counter() - model_time

                            inf_time = time.perf_counter()
                            situation_vector_batch, value_batch, policy_logits_batch_level_list = model(batch_dict)
                            policy_logits_batch_level_list = [policy_logits_batch_level.cpu().numpy()
                                                              for policy_logits_batch_level in
                                                              policy_logits_batch_level_list]
                            value_batch = value_batch.cpu().numpy()
                            self.model_time["model"] += time.perf_counter() - inf_time
                            # Send batched results to inference results
                            self.return_result_batch(self.actor_returns[key], self.query_ids[key],
                                                     policy_logits_batch_level_list,
                                                     value_batch,
                                                     situation_vector_batch.cpu())
                            self.model_time["full"] += time.perf_counter() - model_time

                            # reset everything
                            self.batch[key] = []
                            self.query_ids[key] = []
                            self.actor_returns[key] = []

                        for actor in self.registered_actors:
                            self.data_from_registered_actors[actor] = False

                time.sleep(0)

    def return_result_batch(self, actor_returns, query_ids, policy_logits_batch_level_list, value_batch,
                            situation_vector_batch):
        for actor_id, from_idx, to_idx in actor_returns:
            actor = ray.get_actor(f"experience_worker_{actor_id}")
            actor.add_query_results.remote(
                (query_ids[from_idx: to_idx], situation_vector_batch[from_idx: to_idx],
                 [policy_logits_batch_level[from_idx: to_idx] for policy_logits_batch_level in policy_logits_batch_level_list],
                 value_batch[from_idx: to_idx])
            )

    def get_time(self):
        """
        Gets and resets time spent for inferencing in this model.
        """
        t = self.model_time
        self.model_time = {
            "full": 0.,
            "batching": 0.,
            "model": 0.
        }
        return t

    def add_list_to_queue(self, actor_id, query_ids: Dict, query_states: Dict, model_keys: List[str]):
        """
        List version of `add_to_queue`.
        """
        the_time = time.time()
        if actor_id not in self.registered_actors:
            raise Exception("Adding data from unregistered actor.")

        with self.registering_lock:
            self.data_from_registered_actors[actor_id] = True
            for model_key in model_keys:
                n = len(query_ids[model_key])
                if n:
                    current_batch_len = len(self.batch[model_key])
                    self.batch[model_key].extend(query_states[model_key])
                    self.query_ids[model_key].extend(query_ids[model_key])
                    self.actor_returns[model_key].append((actor_id, current_batch_len, current_batch_len + n))
                    self.last_added_to_queue[model_key] = the_time

    def register_actor(self, actor_id):
        if actor_id in self.registered_actors:
            raise Exception("Registering actor which is already registered")
        with self.registering_lock:
            self.registered_actors.append(actor_id)
            self.data_from_registered_actors[actor_id] = False

    def unregister_actor(self, actor_id):
        if not actor_id in self.registered_actors:
            raise Exception("Unregistering actor which has not been registered.")
        with self.registering_lock:
            self.registered_actors.remove(actor_id)
            del self.data_from_registered_actors[actor_id]