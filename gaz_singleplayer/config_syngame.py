import os
import datetime


class Config:
    def __init__(self):
        super().__init__()

        self.problem_specifics = {
            "latent_dimension": 128,
            "num_mixer_blocks": 5,  # Number of MLP-Mixer blocks excluding the finishing mixer block for the heads
            "expansion_factor_feature_mixing": 4,  # factor * latent dim is hidden dim of MLP for feature mixing
            "expansion_factor_token_mixing": 2,  # factor * num_tokens is hidden dim of MLP for token mixing
            "normalization": "layer",  # "instance" or "batch" or "layer"
        }

        # Gumbel AlphaZero specific parameters
        self.gumbel_sample_n_actions = 16  # (Max) number of actions to sample at the root for the sequential halving procedure
        self.gumbel_c_visit = 50. # constant c_visit in sigma-function.
        self.gumbel_c_scale = 1.  # constant c_scale in sigma-function.
        self.gumbel_simple_loss = False  # If True, KL divergence is minimized w.r.t. one-hot-encoded move, and not
        # w.r.t. distribution based on completed Q-values
        self.gumbel_test_greedy_rollout = False  # If True, then in evaluation mode the policy of the learning actor is rolled out greedily (no MCTS)
        self.num_simulations = 200  # Number of search simulations in GAZ's tree search
        self.simple_rollout_greedy = True  # If False, the simple rollout in rollout obtainer is performed by sampling actions
        self.include_probability_based_beam_search = True
        self.include_value_based_beam_search = False
        self.beam_search_width = 5
        self.max_num_finished_trajectories_in_beam_search = 27  # As theoretically beam search can go on forever in this environment, this specifies number of finished trajectories after which beam search should stop
        # ----

        self.algorithm = "single_vanilla"
        self.singleplayer_options = {
            "method": "single",
            "bootstrap_final_objective": True,
            "bootstrap_n_steps": 0
        }

        self.seed = 43  # Random seed for torch, numpy, initial states.

        # --- Inferencer and experience generation options --- #
        self.num_experience_workers = 50  # Number of actors (processes) which generate experience.
        self.num_inference_workers = 1  # Number of workers which perform network inferences. Each inferencer is pinned to a CPU.
        self.inference_on_experience_workers: bool = True   # If True, states are not sent to central inference workers
                                                            # but performed directly on the experience worker
        self.check_for_new_model_every_n_seconds = 30  # Check the storage for a new model every n seconds

        self.pin_workers_to_core = True  # If True, workers are pinned to specific CPU cores, starting to count from 0. Only on linux.
        self.CUDA_VISIBLE_DEVICES = "0,1"  # Must be set, as ray can have problems detecting multiple GPUs
        self.cuda_device = "cuda:0"  # Cuda device on which to *train* the network. Set to `None` if not available and training should be performed on cpu.
        # For each inference worker, specify on which device it can run. Set to `None`, if shouldn't use a GPU.
        # Expects a list of devices, where i-th entry corresponds to the target device of i-th inference worker.
        # If `inference_on_experience_workers` is True, and the length of the given list below is equal to the number of experience workers,
        # then the variable sepcifies the device on which the *experience_worker* should infer its batches.
        self.cuda_devices_for_inference_workers = ["cpu"] * self.num_experience_workers

        # Number of most recent games to store in the replay buffer
        self.replay_buffer_size = 2000

        # --- Training / Optimizer specifics --- #

        # Tries to keep the ratio of training steps to number of episodes within the given range.
        # Set it to None to disable it
        self.ratio_range = [2.5, 3]
        self.start_train_after_episodes = 500  # wait for n episodes before starting training
        self.level_based_game_stepsize = 10  # Waits until at least n episodes with actions from a specific level are available before training on the level.
        self.level_based_game_step_upper_limit = 50  # Increases the least number of episodes from above until this given upper limit. For detailed logic see replay_buffer.py
        # Total number of batches to train the network on
        self.training_games = 50000

        self.batch_size = 128  # Batch size for training network
        self.inference_max_batch_size = 64  # Maximum batch size for inference on experience workers (coming from MCTS)
        self.lr_init = 0.0001  # Initial learning rate
        self.weight_decay = 1e-4  # L2 weights regularization
        self.gradient_clipping = 1  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.

        self.lr_init = 0.0001  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3  # means that after `decay_steps` training steps, the learning rate has decayed by `decay_rate`

        self.objective_scaling: float = 10.  # Linear scale of objective to transform the value prediction objective. Scaling is performed after clipping.
        self.objective_clipping = None  # first entry is lower bound to clip objective to, second entry is upper bound. Set to `None`
                                        # for no clipping at all and set upper or lower bound to `None` to omit clipping.
        self.average_policy_loss_elementwise: bool = False  # If True, the policy loss in each batch is averaged element-wise per sample.
        self.checkpoint_interval = 100  # Number of training steps before using the model for generating experience.

        self.results_path = os.path.join("./results",
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store model weights, blueprints, logs

        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.load_checkpoint_from_path = None  # If given, model weights and optimizer state is loaded from this path.
        self.only_load_model_weights = False  # If True, only the model weights are loaded from `load_checkpoint_from_path`
                                              # Optimizer state, number of played games etc., is freshly created.

        # --- Logging --- #
        self.log_avg_stats_every_n_episodes = 100  # Compute average game statistics over last n epusides and log them
        self.log_avg_loss_every_n_steps = 100  # Compute average loss over last n training steps and log them
        self.log_policies_for_moves = []  # Logs probability distributions for numbered moves
        self.do_log_to_file = True  # If additional logging to `log.txt` should be performed.

        # --- Evaluation --- #
        self.num_evaluation_games = 200  # For each validation run, how many instances should be solved in the
        # of the validation set (taken from the start)
        self.validation_set_path = "./test/syngaz_eval.pickle"
        self.test_set_path = "./test/syngaz_test.pickle"
        self.evaluate_every_n_steps = 2500 * 3  # Make evaluation run every n training steps
