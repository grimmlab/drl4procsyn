import os
import argparse
import importlib
from typing import Type

import numpy as np
import torch
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import time

from gaz_singleplayer.config_syngame import Config
from gaz_singleplayer.synthesis_network import SynthesisNetwork
from environment.env_config import EnvConfig
from network_trainer import NetworkTrainer
from replay_buffer import ReplayBuffer
from logger import Logger
from evaluation import Evaluation
from inferencer import ModelInferencer
from shared_storage import SharedStorage
from gaz_singleplayer.syn_game import Game
from test_environment import generate_test_sets


class GAZLauncher:
    """
    Main class which builds up everything and spawns training
    """
    def __init__(self, config: Config, network_class: Type[SynthesisNetwork], game_class: Type[Game],
                 experience_worker_class):
        """
        Parameters:
            config [BaseConfig]: Config object
            network_class: Problem specific network module in "model"
        """
        self.config = config
        self.env_config = EnvConfig()

        # generate the test sets (constant seed)
        generate_test_sets(1000, self.env_config)

        # Fix random number generator seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.experience_worker_class = experience_worker_class
        self.network_class = network_class
        self.game_class = game_class

        # Get devices and number of processes which need gpus
        self.gpu_access = {}
        self.training_device = torch.device(
            self.config.cuda_device if self.config.cuda_device and torch.cuda.is_available() else "cpu"
        )
        if torch.cuda.is_available():
            if self.config.cuda_device:
                self.gpu_access[self.config.cuda_device] = 1
            for inference_device in self.config.cuda_devices_for_inference_workers:
                if inference_device and inference_device != "cpu":
                    if not inference_device in self.gpu_access and inference_device:
                        self.gpu_access[inference_device] = 0
                    self.gpu_access[inference_device] += 1

        print(f"{len(self.gpu_access.keys())} GPU devices are accessed by number of processes: {self.gpu_access}")

        ray.init(
            num_gpus=len(self.gpu_access.keys()),
            logging_level="info"
        )
        print(ray.available_resources())

        # Initialize checkpoint and replay buffer. Gets loaded later if needed
        self.checkpoint = {
            "weights_newcomer": None,  # Model weights for learning actor, aka "newcomer"
            "weights_timestamp_newcomer": 0,  # Timestamp for model weights for learning actor, so that unchanged models do not need to be copied.
            "weights_best": None,         # same as for newcomer, only for greedy actor, aka "best" model (legacy)
            "weights_timestamp_best": 0,  # same as for newcomer, only for currently "best" model (legacy)
            "optimizer_state": None,  # Saved optimizer state
            "training_step": 0,  # Number of training steps performed so far
            "num_played_games": 0,  # number of all played episodes so far
            "num_played_steps": 0,  # number of all played moves so far
            "terminate": False,
            "best_eval_score": float('-inf')
        }

        self.replay_buffer = None

        # Workers
        self.experience_workers = None
        self.training_net_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.logging_worker = None
        self.model_inference_workers = None

        # Get the number of model parameters
        temp_model = network_class(config=config, env_config=self.env_config)
        pytorch_total_params = sum(p.numel() for p in temp_model.parameters())
        print(f"Number of parameters in model: {pytorch_total_params:,}")

    def setup_workers(self):
        """
        Sets up all workers except the training worker.
        """
        core = 0  # CPU which is passed to each worker so that they can pin themselves to core

        self.shared_storage_worker = SharedStorage.remote(
            self.checkpoint, self.config
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(
            self.checkpoint, self.config, self.env_config, self.network_class, self.game_class, self.replay_buffer
        )

        if not self.config.inference_on_experience_workers:
            if self.config.pin_workers_to_core:
                print("Inference workers are pinned to CPU cores...")
            self.model_inference_workers = []
            for i in range(self.config.num_inference_workers):
                gpu_share = 1 / self.gpu_access[self.config.cuda_devices_for_inference_workers[i]] \
                    if torch.cuda.is_available() and self.config.cuda_devices_for_inference_workers[i] != "cpu" else 0
                device = torch.device(
                    self.config.cuda_devices_for_inference_workers[i] if torch.cuda.is_available() else "cpu"
                )

                self.model_inference_workers.append(
                    ModelInferencer.options(max_concurrency=2,
                                            name=f"inferencer_{i}",
                                            num_gpus=gpu_share
                                            ).remote(
                        config=self.config,
                        env_config=self.env_config,
                        shared_storage=self.shared_storage_worker,
                        network_class=self.network_class,
                        model_named_keys=["newcomer", "best"],
                        device=device,
                        initial_checkpoint=self.checkpoint,
                        random_seed=self.config.seed,
                        cpu_core=core
                        )
                )
                core += 1

        if self.config.pin_workers_to_core:
            print("Experience workers are pinned to CPU cores...")

        self.experience_workers = []
        for i in range(self.config.num_experience_workers):
            gpu_share = 0
            # get the correct inference worker.
            if not self.config.inference_on_experience_workers:
                inference_worker = self.model_inference_workers[i % self.config.num_inference_workers]
            else:
                device = "cpu"
                if len(self.config.cuda_devices_for_inference_workers) == self.config.num_experience_workers:
                    device = self.config.cuda_devices_for_inference_workers[i] if torch.cuda.is_available() else "cpu"
                    gpu_share = 1 / self.gpu_access[self.config.cuda_devices_for_inference_workers[i]] \
                        if torch.cuda.is_available() and self.config.cuda_devices_for_inference_workers[i] != "cpu" else 0
                inference_worker = device

            self.experience_workers.append(
                self.experience_worker_class.options(
                    name=f"experience_worker_{i}",
                    max_concurrency=2,
                    num_gpus=gpu_share
                ).remote(
                    actor_id=i, config=self.config, env_config=self.env_config,
                    shared_storage=self.shared_storage_worker,
                    model_inference_worker=inference_worker,
                    game_class=self.game_class,
                    network_class=self.network_class,
                    random_seed=self.config.seed + i,
                    cpu_core=core + i
                )
            )

        self.logging_worker = Logger.remote(self.config, self.shared_storage_worker,
                                            self.model_inference_workers)

    def train(self):
        """
        Spawn ray workers, load models, launch training
        """
        self.setup_workers()

        # Initialize training worker
        training_gpu_share = 1 / self.gpu_access[self.config.cuda_device] if torch.cuda.is_available() and self.config.cuda_device else 0
        training_device = torch.device(
            self.config.cuda_device if torch.cuda.is_available() else "cpu"
        )
        self.training_net_worker = NetworkTrainer.options(
            num_cpus=1, num_gpus=training_gpu_share
        ).remote(self.config, self.env_config, self.shared_storage_worker, self.network_class, self.checkpoint, training_device)

        ## Launch all the workers
        for i, experience_worker in enumerate(self.experience_workers):
            experience_worker.continuous_play.remote(
                self.replay_buffer_worker, self.logging_worker
            )

        if self.model_inference_workers is not None:
            for model_inference_worker in self.model_inference_workers:
                model_inference_worker.continuous_inference.remote()

        self.training_net_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.logging_worker
        )

        # Loop to check if we are done with training and evaluation
        last_evaluation_at_step = self.checkpoint["training_step"]
        while (
                ray.get(self.shared_storage_worker.get_info.remote("num_played_games")) < self.config.training_games
        ):
            # check if we need to evaluate
            training_step = ray.get(self.shared_storage_worker.get_info.remote("training_step"))

            if training_step - last_evaluation_at_step >= self.config.evaluate_every_n_steps:
                # otherwise evaluate
                self.perform_evaluation(n_episodes=self.config.num_evaluation_games,
                                        set_path=self.config.validation_set_path, save_model=self.config.save_model)
                last_evaluation_at_step = training_step

            time.sleep(1)
            
        print("Done Training. Evaluating last model.")
        self.perform_evaluation(n_episodes=self.config.num_evaluation_games,
                                set_path=self.config.validation_set_path, save_model=self.config.save_model)

        if self.config.save_model:
            path = os.path.join(self.config.results_path, "best_model.pt")
            self.checkpoint = torch.load(path)
            ray.get(self.shared_storage_worker.set_checkpoint.remote(self.checkpoint))

        model_type = "best" if self.config.save_model else "last"

        print(f"Evaluating {model_type} model on test set...")
        # wait until the best model has propagated
        time.sleep(40)
        self.perform_evaluation(n_episodes=self.config.num_evaluation_games, set_path=self.config.test_set_path, save_model=False)

        self.terminate_workers()
        
    def test(self):
        print("Testing model")
        self.setup_workers()

        ray.get(self.shared_storage_worker.set_evaluation_mode.remote(True))

        ## Launch all the workers
        for i, experience_worker in enumerate(self.experience_workers):
            experience_worker.continuous_play.remote(
                self.replay_buffer_worker, self.logging_worker
            )

        if self.model_inference_workers is not None:
            for model_inference_worker in self.model_inference_workers:
                model_inference_worker.continuous_inference.remote()

        self.perform_evaluation(n_episodes=self.config.num_evaluation_games, set_path=self.config.test_set_path, save_model=False)
        time.sleep(5)

    def perform_evaluation(self, n_episodes: int, set_path: str, save_model=True):
        evaluator = Evaluation(self.config, self.shared_storage_worker)
        evaluator.start_evaluation()

        stats = evaluator.evaluate(n_episodes=n_episodes, set_path=set_path, save_results=n_episodes == -1)
        stats["n_games"] = ray.get(self.shared_storage_worker.get_info.remote("num_played_games"))
        ray.get(self.logging_worker.evaluation_run.remote(stats))

        if stats["avg_objective"] > ray.get(self.shared_storage_worker.get_info.remote("best_eval_score")) and save_model:
            print("Saving as best model...")
            ray.get(self.shared_storage_worker.set_info.remote("best_eval_score", stats["avg_objective"]))
            # Save the current model as best model
            ray.get(self.shared_storage_worker.save_checkpoint.remote("best_model.pt"))
        elif save_model:
            # Save the current model as last model
            ray.get(self.shared_storage_worker.save_checkpoint.remote("last_model.pt"))

        evaluator.stop_evaluation()
        return stats

    def terminate_workers(self):
        """
        Softly terminate workers and garbage collect them.
        Also update self.checkpoint by last checkpoint of shared storage.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            # get last checkpoint
            self.checkpoint = ray.get(self.shared_storage_worker.get_checkpoint.remote())
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")
        self.experience_workers = None
        self.training_net_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        self.logging_worker = None
        self.model_inference_workers = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment.')
    parser.add_argument('--seed', type=int, help="Custom random seed, overruling config.py")
    args = parser.parse_args()

    # Get method and problem specific config
    method_module = "gaz_singleplayer"
    config_path = f"{method_module}.config_syngame"
    config_class = getattr(importlib.import_module(config_path), "Config")
    config = config_class()

    # Check custom random seed and overwrite in config.
    if args.seed:
        config.seed = args.seed

    # Get network class
    network_module = importlib.import_module(f"{method_module}.synthesis_network")
    network_class = getattr(network_module, "SynthesisNetwork")

    # Get game class
    game_module = importlib.import_module(f"{method_module}.syn_game")
    game_class = getattr(game_module, "Game")

    # Get experience worker class
    experience_worker_module = importlib.import_module(f"{method_module}.experience_worker")
    experience_worker_class = getattr(experience_worker_module, "ExperienceWorker")

    gaz = GAZLauncher(config, network_class, game_class, experience_worker_class)

    # Load previous model if specified
    if config.load_checkpoint_from_path:
        print(f"Loading checkpoint from path {config.load_checkpoint_from_path}")
        checkpoint = torch.load(config.load_checkpoint_from_path)
        if "best_plays_randomly" not in checkpoint:
            checkpoint["best_plays_randomly"] = False
        if config.only_load_model_weights:
            print("Only using model weights from loaded checkpoint")
            gaz.checkpoint["weights_newcomer"] = checkpoint["weights_newcomer"]
            gaz.checkpoint["weights_best"] = checkpoint["weights_best"]
            gaz.checkpoint["weights_timestamp_newcomer"] = checkpoint["weights_timestamp_newcomer"]
            gaz.checkpoint["weights_timestamp_best"] = checkpoint["weights_timestamp_best"]
            gaz.checkpoint["best_plays_randomly"] = checkpoint["best_plays_randomly"]
        else:
            gaz.checkpoint = checkpoint

    if config.training_games > 0:
        print("Starting Training...")
        gaz.train()
    else:
        if not config.load_checkpoint_from_path:
            print("WARNING: Testing mode, but no checkpoint to load was specified.")
        gaz.test()

    ray.shutdown()