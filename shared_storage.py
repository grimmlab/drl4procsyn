import copy
import os

import ray
import torch

from typing import Dict
from gaz_singleplayer.config_syngame import Config


@ray.remote
class SharedStorage:
    """
    There will be only one instance of this class.
    SharedStorage runs in a dedicated process to store the network weights and some information about played episodes.
    """
    def __init__(self, checkpoint: Dict, config: Config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

        # Variables for evaluation mode
        self.evaluate_list = []  # Stores flowrates which should be evaluated
        self.evaluation_results = []
        self.evaluation_mode = False

    def save_checkpoint(self, filename: str):
        os.makedirs(self.config.results_path, exist_ok=True)
        path = os.path.join(self.config.results_path, filename)

        torch.save(self.current_checkpoint, path)
        
    def set_checkpoint(self, checkpoint: Dict):
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

    def in_evaluation_mode(self):
        return self.evaluation_mode

    def set_evaluation_mode(self, value: bool):
        self.evaluation_mode = value

    def get_to_evaluate(self):
        if len(self.evaluate_list) > 0:
            item = self.evaluate_list.pop()
            return copy.deepcopy(item)
        else:
            return None

    def set_to_evaluate(self, evaluate_list):
        self.evaluate_list = evaluate_list

    def push_evaluation_result(self, eval_result):
        self.evaluation_results.append(eval_result)

    def fetch_evaluation_results(self):
        results = self.evaluation_results
        self.evaluation_results = []
        return results