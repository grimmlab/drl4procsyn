import os
import pickle

import numpy as np
import ray
import time

from gaz_singleplayer.config_syngame import Config
from shared_storage import SharedStorage
from tqdm import tqdm


class Evaluation:
    def __init__(self, config: Config, shared_storage: SharedStorage):
        self.config = config
        self.shared_storage = shared_storage

    def start_evaluation(self):
        ray.get(self.shared_storage.set_evaluation_mode.remote(True))

    def stop_evaluation(self):
        ray.get(self.shared_storage.set_evaluation_mode.remote(False))

    def evaluate(self, n_episodes: int, set_path: str, save_results: bool = False):
        print("Performing Evaluation...")
        objectives_mcts = []
        objectives_beam = []

        # Get instances by loading them from the validation file.
        if ".pickle" in set_path:
            with open(set_path, "rb") as handle:
                validation_instances = pickle.load(handle)
        elif ".npy" in set_path:
            validation_instances = np.load(set_path)
        else:
            raise Exception("Unknown file type")

        instance_list = [(i, validation_instances[i], "test") for i in range(n_episodes)]

        ray.get(self.shared_storage.set_to_evaluate.remote(instance_list))

        eval_results = [None] * n_episodes

        with tqdm(total=n_episodes) as progress_bar:
            while None in eval_results:
                time.sleep(0.5)
                fetched_results = ray.get(self.shared_storage.fetch_evaluation_results.remote())
                for (i, result) in fetched_results:
                    eval_results[i] = result
                progress_bar.update(len(fetched_results))

        # we store dicts with the sequences of the evaluation games in this list
        list_to_print = []
        for i, result in enumerate(eval_results):
            if result == "broken":
                continue
            max_objective = max(result["objective"], result["baseline_objective"])
            objectives_mcts.append(result["objective"])
            objectives_beam.append(result["baseline_objective"])

            if "sequence" in result:
                list_to_print.append({
                    "winning obj": max_objective,
                    "learning actor sequence": result["sequence"],
                    "learning actor obj": result["objective"],
                    "baseline seq": result["baseline_sequence"],
                    "baseline obj": result["baseline_objective"]
                })

        objectives_mcts = np.array(objectives_mcts)
        objectives_beam = np.array(objectives_beam)

        # Save the objectives for computing margins
        if save_results:
            np.save(os.path.join(self.config.results_path, "eval_mcts.npy"), objectives_mcts)
            np.save(os.path.join(self.config.results_path, "eval_beam.npy"), objectives_beam)

        # Compute some stats and store the sequences
        stats = {
            "type": "Validation",
            "avg_objective": objectives_beam.mean(),
            "avg_objective_mcts": objectives_mcts.mean(),
            "avg_objective_beam": objectives_beam.mean(),
            "ratio_mcts_better": (objectives_mcts > objectives_beam).mean(),
            "dicts_to_print": list_to_print
        }

        return stats
