import os

import ray
import time
import json
import pickle
from typing import Dict
from gaz_singleplayer.config_syngame import Config


@ray.remote
class Logger:
    def __init__(self, config: Config, shared_storage, inferencers):
        self.config = config
        self.shared_storage = shared_storage

        self.n_played_games = 0
        # Check number of games played before this run (if a training is resumed from some checkpoint)
        self.n_played_games_previous = ray.get(shared_storage.get_info.remote("num_played_games"))
        self.rolling_game_stats = None
        self.play_took_time = 0

        self.reset_rolling_game_stats()

        self.n_trained_steps = 0
        self.n_trained_steps_previous = ray.get(shared_storage.get_info.remote("training_step"))
        self.rolling_loss_stats = None
        self.reset_rolling_loss_stats()

        self.inferencers = inferencers

        # paths to write sequences of evaluation to
        self.file_eval_prints_path = os.path.join(self.config.results_path, "evaluation_prints.txt")
        self.file_eval_pickle_path = os.path.join(self.config.results_path, "evaluation_win_blueprint_bin.pickle")

        self.file_log_path = os.path.join(self.config.results_path, "log.txt")
        os.makedirs(self.config.results_path, exist_ok=True)

    def reset_rolling_game_stats(self):
        self.play_took_time = time.perf_counter()
        self.rolling_game_stats = {"max_policies_for_selected_moves": {}, "max_search_depth": 0, "game_time": 0,
                                   "waiting_time": 0, "objective": 0, "explicit_npv": 0, "baseline_objective": 0,
                                   "baseline_num_moves": 0, "num_level_0_moves": 0}

        for n_actions in self.config.log_policies_for_moves:
            self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] = 0

    def reset_rolling_loss_stats(self):
        self.rolling_loss_stats = {
            "loss": 0,
            "value_loss": 0,
            "policy_loss": 0
        }

    def played_game(self, game_stats: Dict, game_type="train"):
        """
        Notify logger of new played game. `game_stats` is a dict of the form
            {
            "objective": float("-inf"),
            "sequence": None,
            "num_level_0_moves": 0,
            "max_search_depth": 0,
            "policies_for_selected_moves": {},
            "baseline_objective": baseline_objective,
            "baseline_sequence": baseline_action_sequence,
            "baseline_num_moves": len(baseline_states) - 1
            }
        """
        self.n_played_games += 1
        self.rolling_game_stats["game_time"] += game_stats["game_time"]
        self.rolling_game_stats["max_search_depth"] += game_stats["max_search_depth"]
        if "waiting_time" in game_stats:
            self.rolling_game_stats["waiting_time"] += game_stats["waiting_time"]

        self.rolling_game_stats["objective"] += game_stats["objective"]
        self.rolling_game_stats["explicit_npv"] += game_stats["explicit_npv"]
        self.rolling_game_stats["baseline_objective"] += game_stats["baseline_objective"]
        self.rolling_game_stats["baseline_num_moves"] += game_stats["baseline_num_moves"]
        self.rolling_game_stats["num_level_0_moves"] += game_stats["num_level_0_moves"]

        for n_actions in self.rolling_game_stats["max_policies_for_selected_moves"].keys():
            self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] += \
                max(game_stats["policies_for_selected_moves"][n_actions])

        if self.n_played_games % self.config.log_avg_stats_every_n_episodes == 0:
            games_took_time = time.perf_counter() - self.play_took_time
            print(f'Num played games total: {self.n_played_games}')
            print(f"Episodes took time {games_took_time} s")

            # Get time it took for models on average
            avg_model_inference_time = 0
            if not self.config.inference_on_experience_workers:
                keys = ["full", "batching", "model"]
                inferencer_times = []
                for inferencer in self.inferencers:
                    inferencer_times.append(ray.get(inferencer.get_time.remote()))
                for key in keys:
                    inf_time = 0
                    for inferencer_time in inferencer_times:
                        inf_time += inferencer_time[key]
                    avg_model_inference_time = inf_time / len(self.inferencers)
                    print(f"Avg. model inference time '{key}': {avg_model_inference_time}")

            avg_objective = self.rolling_game_stats["objective"] / self.config.log_avg_stats_every_n_episodes
            avg_npv = self.rolling_game_stats["explicit_npv"] / self.config.log_avg_stats_every_n_episodes
            avg_baseline_objective = self.rolling_game_stats[
                                       "baseline_objective"] / self.config.log_avg_stats_every_n_episodes
            avg_baseline_num_moves = self.rolling_game_stats[
                                         "baseline_num_moves"] / self.config.log_avg_stats_every_n_episodes
            avg_num_moves = self.rolling_game_stats["num_level_0_moves"] / self.config.log_avg_stats_every_n_episodes

            # average maximum search depth of games
            avg_max_depth = self.rolling_game_stats["max_search_depth"] / self.config.log_avg_stats_every_n_episodes

            # Average maximum probability for selected moves
            for n_actions in self.config.log_policies_for_moves:
                self.rolling_game_stats["max_policies_for_selected_moves"][n_actions] /= self.config.log_avg_stats_every_n_episodes

            avg_time_per_game = self.rolling_game_stats["game_time"] / self.config.log_avg_stats_every_n_episodes
            avg_waiting_time_per_game = self.rolling_game_stats[
                                            "waiting_time"] / self.config.log_avg_stats_every_n_episodes
            print(f"Average time per game: {avg_time_per_game}")
            print(f"Average waiting time per game: {avg_waiting_time_per_game}")
            print(f'Avg max search depth per move: {avg_max_depth:.1f}')
            print(f'Avg objective: {avg_objective}')
            print(f'Avg NPV: {avg_npv}')
            print(f'Avg baseline objective: {avg_baseline_objective}')
            print(f'Avg baseline num moves: {avg_baseline_num_moves}')
            print(f'Avg num level zero moves: {avg_num_moves}')
            metrics_to_log = {"Avg objective": avg_objective, "Avg npv": avg_npv,
                              "Avg baseline objective": avg_baseline_objective,
                              "Avg baseline num moves": avg_baseline_num_moves,
                              "Avg num level zero moves": avg_num_moves, "Games time in secs": games_took_time,
                              "Avg game time in secs": avg_time_per_game,
                              "Avg Inferencer Time in secs": avg_model_inference_time,
                              "Avg max search depth per move": avg_max_depth}

            for n_actions in self.config.log_policies_for_moves:
                metrics_to_log[f"Max policy newcomer {n_actions}"] = \
                    self.rolling_game_stats["max_policies_for_selected_moves"][n_actions]

            self.reset_rolling_game_stats()

            if self.config.do_log_to_file:
                # Additional things for logging to file
                metrics_to_log["Total num played games"] = self.n_played_games
                metrics_to_log["Total num trained steps"] = self.n_trained_steps
                metrics_to_log["Timestamp in ms"] = int(time.time() * 1000)
                metrics_to_log["logtype"] = "played_game"

                with open(self.file_log_path, "a+") as f:
                    f.write(json.dumps(metrics_to_log))
                    f.write("\n")

    def training_step(self, loss_dict: Dict):
        """
        Notify logger of performed training step. loss_dict has keys "loss", "value_loss" and "policy_loss" (all floats)
        for a batch on which has been trained.
        """

        self.n_trained_steps += 1

        self.rolling_loss_stats["loss"] += loss_dict["loss"]
        self.rolling_loss_stats["value_loss"] += loss_dict["value_loss"]
        self.rolling_loss_stats["policy_loss"] += loss_dict["policy_loss"]

        if self.n_trained_steps % self.config.log_avg_loss_every_n_steps == 0:
            # Also get training_steps to played_steps ratio
            training_steps = ray.get(self.shared_storage.get_info.remote("training_step"))
            played_games = ray.get(self.shared_storage.get_info.remote("num_played_games"))
            avg_loss = self.rolling_loss_stats["loss"] / self.config.log_avg_loss_every_n_steps
            avg_value_loss = self.rolling_loss_stats["value_loss"] / self.config.log_avg_loss_every_n_steps
            avg_policy_loss = self.rolling_loss_stats["policy_loss"] / self.config.log_avg_loss_every_n_steps

            ratio_steps_games = training_steps/played_games

            print(f"Total number of training steps: {self.n_trained_steps}, "
                  f"Ratio training steps to played games: {ratio_steps_games:.2f}, "
                  f"Avg loss: {avg_loss}, Avg value Loss: {avg_value_loss}, "
                  f"Avg policy loss: {avg_policy_loss}")
            self.reset_rolling_loss_stats()

            metrics_to_log = {
                    "Ratio training steps to played games": ratio_steps_games,
                    "Avg loss": avg_loss,
                    "Avg value loss": avg_value_loss,
                    "Avg policy loss": avg_policy_loss
                }

            if self.config.do_log_to_file:
                # Additional things for logging to file
                metrics_to_log["Total num played games"] = self.n_played_games
                metrics_to_log["Total num trained steps"] = self.n_trained_steps
                metrics_to_log["Timestamp in ms"] = int(time.time() * 1000)
                metrics_to_log["logtype"] = "training_step"

                with open(self.file_log_path, "a+") as f:
                    f.write(json.dumps(metrics_to_log))
                    f.write("\n")

    def evaluation_run(self, stats_dict: Dict):
        print(
            f"EVALUATION. Average MCTS objective: {stats_dict['avg_objective_mcts']}, "
            f"Average Beam Search objective: {stats_dict['avg_objective_beam']}, "
            f"Ratio MCTS better: {stats_dict['ratio_mcts_better']}"
        )

        # we store the winning blueprints also as binary (for plots later)
        blueprint_list_for_binary_file = []

        # write sequences to file and log it to mlflow
        with open(self.file_eval_prints_path, "w") as f:
            f.write("\n\n")
            f.write("trained steps: " + str(self.n_trained_steps) + "\n")
            f.write("----------------------------------------\n\n")
            for d_ind, d in enumerate(stats_dict["dicts_to_print"]):
                blueprint_list_for_binary_file.append([d["baseline seq"],
                                                       d["learning actor sequence"]])

                f.write("game nr:" + str(d_ind + 1) + "\n")
                for k, v in d.items():
                    if k == "learning actor sequence":
                        f.write("situation index: " + str(v["initial_information"]["feed_situation_index"]) + "\n")
                        f.write("order in feed: " + str(v["initial_information"]["indices_components_in_feeds"]) + "\n")
                        f.write("feeds: " + str(v["initial_information"]["list_feed_streams"]) + "\n\n")
                        f.write("\ngame stats:\n")
                        f.write("num actions learning: " + str(len(v["move_seq"])) + "\n")
                        f.write("action sequence learning: " + str(v["move_seq"]) + "\n")
                        # print leaving streams
                        f.write("leaving streams learning: (index, flowrate, composition)\n")
                        for stream in v["leaving_streams"]:
                            f.write(str(stream["index"]) + ", " + "{:.3f}".format(stream["flowrate"]) +\
                                    ", " + str(stream["composition"]) + "\n")

                        f.write("\n")

                    elif k == "baseline seq":
                        f.write("num actions baseline: " + str(len(v["move_seq"])) + "\n")
                        f.write("action sequence baseline: " + str(v["move_seq"]) + "\n")
                        # print leaving streams
                        f.write("leaving streams baseline: (index, flowrate, composition)\n")
                        for stream in v["leaving_streams"]:
                            f.write(str(stream["index"]) + ", " + "{:.3f}".format(stream["flowrate"]) + \
                                    ", " + str(stream["composition"]) + "\n")

                        f.write("\n")

                    else:
                        f.write(str(k) + " >>> " + str(v) + "\n\n")

                f.write("--------------\n\n")

            f.write("----------------------------------------\n\n")

        # log binary
        pickle.dump(blueprint_list_for_binary_file, open(self.file_eval_pickle_path, "wb"))

        if self.config.do_log_to_file:
            # Additional things for logging to file
            metrics_to_log = {
                "Total num played games": self.n_played_games,
                "Total num trained steps": self.n_trained_steps,
                "Timestamp in ms": int(time.time() * 1000),
                "logtype": "evaluation",
                "Evaluation Type": stats_dict['type'],
                "Evaluation Value": stats_dict['avg_objective']
            }

            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics_to_log))
                f.write("\n")
