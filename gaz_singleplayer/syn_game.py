
import numpy as np
import copy
from typing import Tuple, List, Dict, Optional

from gaz_singleplayer.config_syngame import Config
from environment.flowsheet_simulation import FlowsheetSimulation
from environment.env_config import EnvConfig


class Game:
    """
    Class which poses flowsheet synthesis as single-player game.
    """
    def __init__(self, config: Config, flowsheet_simulation_config: EnvConfig, problem_instance: List,
                 suppress_env_creation: bool = False, opponent_outcome: Optional[float] = None):
        """
        Initializes a new singleplayer game.

        Parameters
        ----------
        flowsheet_simulation_config [EnvConfig]: Stores environment specific parameters and settings.
        problem_instance [list]: List of np.arrays, which represent the feed streams.
        suppress_env_creation [bool]: If `True`, no environments are created. This is only used for custom copying
            of games. See `copy`-method below.
        opponent_outcome [float]: Baseline against which to compete in this game.
        """
        self.config = config
        self.flowsheet_simulation_config = flowsheet_simulation_config
        self.problem_instance = problem_instance
        self.opponent_outcome = opponent_outcome

        # the decisions are structured hierarchically on multiple levels, here we keep track of
        # the level of the current player (decisions always start at level 0)
        # possible levels are 0, 1, 2, 3 (= 2a and 2b_1), 4 (2b_2)
        self.level_current_player = 0
        # a complete action of a player, which can be executed in the flowsheet simulation, consists always
        # of stream_index, apparatus_type, apparatus_specification (some of this may be a None if not
        # necessary)
        self.action_current_player = {"line_index": None, "unit_index": None,
                                      "spec_cont": None, "spec_disc": None}

        # flag for playing in a broken game (i.e. playing on when somebody produced a non conv flowsheet)
        self.game_broken = False

        if not suppress_env_creation:
            self.player_environment = FlowsheetSimulation(
                    copy.deepcopy(self.problem_instance), self.flowsheet_simulation_config)

            # np.array with binary entries, indicating if the action is feasible or not
            self.current_feasible_actions = self.player_environment.get_feasible_actions(
                current_level=self.level_current_player, chosen_stream=self.action_current_player["line_index"],
                chosen_unit=self.action_current_player["unit_index"])

        else:
            self.player_environment = None
            self.current_feasible_actions = None

        # holds the players' final npvs
        self.player_npv = -1 * float("inf")
        self.player_npv_explicit = -1 * float("inf") # unnormalized NPV

        # flag indicating whether the game is finished
        self.game_is_over = False

    def get_current_player(self):
        return 1

    def get_current_level(self):
        return self.level_current_player

    def get_objective(self, for_player: int) -> float:
        return self.player_npv

    def get_explicit_npv(self, for_player: int) -> float:
        return self.player_npv_explicit

    def get_sequence(self, for_player: int) -> Dict:
        return self.player_environment.blueprint

    def get_number_of_lvl_zero_moves(self) -> int:
        return self.player_environment.steps

    def get_num_actions(self) -> int:
        """
        Legal actions for the current player at the current level given as a list of ints.
        """
        return len(self.current_feasible_actions)

    def get_feasible_actions_ohe_vector(self) -> np.array:
        return self.current_feasible_actions

    def is_finished_and_winner(self) -> Tuple[bool, int]:
        # Irrelevant for singeplayer games
        return self.game_is_over, 0

    def make_move(self, action: int) -> Tuple[bool, float, bool]:
        """
        Performs a move in the game environment, in the flowsheet case this does not necessarily mean
        that a unit is placed as the action may not be complete.
        Parameters:
            action [int]: The index of the action to play. The action index should be feasible.
        Returns:
            game_done [bool]: flowheet finished
            reward [int]: if game_done, npv
            move_worked [bool]: move led to converging flowsheet or not
        """
        if self.game_broken:
            raise Exception('playing in a broken game')

        # transform to feasible actions array
        # action = np.nonzero(self.current_feasible_actions)[0][action_untransformed]
        if self.current_feasible_actions[action] != 1:
            raise Exception("Playing infeasible action.")

        if self.level_current_player == 0:
            self.action_current_player["line_index"] = action

        elif self.level_current_player == 1:
            self.action_current_player["unit_index"] = action

        elif self.level_current_player == 2:
            self.action_current_player["spec_disc"] = action

        # transform action to continuous parameter if current level is 2b_1 or 2b_2
        # (action_2b_1 * increment_2b_1) + (action_2b_2 * increment_2b_2)
        elif self.level_current_player == 3:
            self.action_current_player["spec_cont"] = [None, [action, None]]

        # level 2b_2
        else:
            # get the respective range (depending on the unit, which was already chosen)
            chosen_unit = self.action_current_player["unit_index"]

            # level 2b_1 calc
            increment = self.flowsheet_simulation_config.increments_per_unit[chosen_unit]["level_2b_1"]
            # action starts with a zero...
            part_of_action_for_simulator = self.action_current_player["spec_cont"][1][0] * increment

            # level 2b_2 calc
            increment = self.flowsheet_simulation_config.increments_per_unit[chosen_unit]["level_2b_2"]
            # action starts with a zero...
            action_for_simulator = (action * increment) + part_of_action_for_simulator

            self.action_current_player["spec_cont"][0] = action_for_simulator
            self.action_current_player["spec_cont"][1][1] = action

        # case distinction for next level
        next_level = None

        # first case: level is 0 (stream or termination)
        if self.level_current_player == 0:
            # termination of synthesis is chosen (num_lines_flowsheet_matrix). the action is complete
            if action == len(self.current_feasible_actions) - 1:
                next_level = 0

            # otherwise a unit needs to placed
            else:
                next_level = 1

        # second case: level is 1 (choose unit)
        elif self.level_current_player == 1:
            next_level = self.flowsheet_simulation_config.unit_types[
                self.flowsheet_simulation_config.units_map_indices_type[action]]["next_level"]

        # third case: level 2 or 2b_2 for unit specification
        elif self.level_current_player == 2 or self.level_current_player == 4:
            next_level = 0

        elif self.level_current_player == 3:
            next_level = 4

        # execute action in the flowsheet simulation if the next level is 0 (move_worked is always True if
        # the action is not complete yet), game cannot be done if the next level is not 0 and in this case
        # there is no reward yet
        move_worked = True
        game_done = False
        reward = 0.
        if next_level == 0:
            _, npv, npv_normed, flowsheet_synthesis_complete, convergent = \
                self.player_environment.place_apparatus(
                    line_index=self.action_current_player["line_index"],
                    apparatus_type_index=self.action_current_player["unit_index"],
                    specification_continuous=self.action_current_player["spec_cont"],
                    specification_discrete=self.action_current_player["spec_disc"])

            if convergent:
                # check if game done
                if flowsheet_synthesis_complete:
                    self.player_npv_explicit = npv
                    self.player_npv = npv if not self.player_environment.config.norm_npv else npv_normed
                    self.game_is_over = game_done = True
                    self.action_current_player = {"line_index": None, "unit_index": None,
                                                  "spec_cont": None, "spec_disc": None}
                    self.level_current_player = 0

                    # compute reward
                    if self.opponent_outcome is not None:
                        reward = -1 if self.player_npv <= self.opponent_outcome else 1
                    else:
                        reward = self.player_npv
                        if self.config.objective_clipping is not None:
                            if self.config.objective_clipping[0] is not None:
                                reward = max(reward, self.config.objective_clipping[0])
                            if self.config.objective_clipping[1] is not None:
                                reward = min(reward, self.config.objective_clipping[1])
                        reward = reward * self.config.objective_scaling

            else:
                move_worked = False
                self.game_broken = True

                # game_done, reward, move_worked
                self.level_current_player = None
                self.action_current_player = None
                self.current_feasible_actions = None
                self.player_environment = None
                return None, None, False

        # set new levels etc for next move/action
        if not self.game_is_over:
            # if next_level is 0, the action was executed in the simulation
            self.level_current_player = next_level
            if next_level == 0:
                self.action_current_player = {"line_index": None, "unit_index": None,
                                              "spec_cont": None, "spec_disc": None}

            # set new feasible actions
            if not self.game_broken:
                self.current_feasible_actions = self.player_environment.get_feasible_actions(
                    current_level=self.level_current_player, chosen_stream=self.action_current_player["line_index"],
                    chosen_unit=self.action_current_player["unit_index"])

        return game_done, reward, move_worked

    def get_current_state(self):
        """
        Returns the current situation as dict. This is the board from the complete game with
        the view of the current player. For the legacy mode we just return a matrix inside the dict
        and ohe of chosen stream and chosen unit.

        For the transformer mode (legacy==False), we return a dict with the following keys:
            flowsheet_finished: True/False

            chosen_stream: index of chosen stream or None

            chosen_unit: OHE of chosen unit or None

            list_line_information: list of dicts with the following keys (the following
            names are similar as in the description pdf, see X_i):
                input_w_stream: vector y in description pdf containing
                    molar fractions, molar flowrate, for each component T_c, p_c, omega and a vector
                    representing the binary interactions (act coeff at inf dilution)

                input_w_unit_specification: input are specified in env_config: self.keys_for_unit_spec_vector

                input_unit: OHE of the respective unit in this line

            flowsheet_connectivity: matrix b_ij from pdf, b_ij=1 -> destination from unit from
                line i to line j, b_ij=2 -> stream i stems from stream j
        """
        if self.game_broken:
            raise Exception("Getting state of a broken game.")

        dict_current_state = dict()

        # this is None for non convergent flowsheets, otherwise a scalar
        dict_current_state["current_npv"] = self.player_environment.current_net_present_value * self.config.objective_scaling
        dict_current_state["num_lines"] = len(self.player_environment.state_simulation["list_line_information"])
        dict_current_state["action_level"] = self.level_current_player
        dict_current_state["feasible_actions"] = self.current_feasible_actions
        dict_current_state["flowsheet_finished"] = self.player_environment.state_simulation["flowsheet_syn_done"]

        # set chosen stream index
        dict_current_state["chosen_stream"] = self.action_current_player[
            "line_index"]  # Note: This can be None. In this case, no attention bias should be set.

        # index of the chosen unit. If none has been chosen yet, a zero vector is passed along.
        one_hot_enc_chosen_unit = np.zeros(self.flowsheet_simulation_config.num_units)
        if self.action_current_player["unit_index"] is not None:
            one_hot_enc_chosen_unit[self.action_current_player["unit_index"]] = 1
        dict_current_state["chosen_unit"] = one_hot_enc_chosen_unit

        # cont level is now separated into 2b_1, 2b_2, if a decision is to be made at 2b_2,
        # we need the information, what was chosen at 2b_1
        dict_current_state["chosen_first_cont_interval"] = \
            np.zeros(self.flowsheet_simulation_config.num_disc_steps_2b_1)
        if self.action_current_player["spec_cont"] is not None:
            dict_current_state["chosen_first_cont_interval"][
                self.action_current_player["spec_cont"][1][0]] = 1

        if self.flowsheet_simulation_config.legacy:
            dict_current_state["flowsheet_matrix"] = \
                self.player_environment.create_legacy_flowsheet_matrix()
        else:
            # keys in list line information:
            # "input_w_stream" (len 19), "input_w_unit_specification" (len 5), "input_unit" (len 14)
            dict_current_state["list_line_information"] = \
                self.player_environment.state_simulation["list_line_information"]

            # to show the connectivity inside the flowsheet, we give for each player the following:
            # for each vector in "line_info", a OHE vector, which shows to
            # which lines the output streams of the unit are pointing (matrix b_ij)
            # This is given here as a numpy square matrix of shape (num lines, num lines)
            dict_current_state["flowsheet_connectivity_matrix"] = \
                self.player_environment.state_simulation["lines_connectivity_matrix"]

        return dict_current_state

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def generate_random_instance(flowsheet_simulation_config) -> Dict:
        return flowsheet_simulation_config.create_random_problem_instance()
