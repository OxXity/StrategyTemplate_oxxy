"""Processor to get referee commands"""

import json
from enum import Enum
from time import time
from typing import Optional

from strategy_bridge.common import config
from strategy_bridge.larcmacs.receiver import ZmqReceiver
from strategy_bridge.model.referee import RefereeCommand

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.const import State
from environment.setup_environment import get_from_env, get_from_env_specific_type


class Command(Enum):
    """Класс с командами от судей"""

    HALT = 0
    STOP = 1
    FORCE_START = 2
    TIMEOUT = 3
    POST_GAME = 4
    PREPARE_KICKOFF = 5
    NORMAL_START = 6
    PREPARE_PENALTY = 7
    NORMAL_START2 = 8
    FREE_KICK = 9
    INDIRECT_FREE = 10
    BALL_PLACEMENT = 11

    BALL_MOVED = 101
    PASS_10_SECONDS = 102


CommandMap: dict[int, Command] = {command.value: command for command in Command}
PreparationStateMap: dict[State, State] = {
    State.KICKOFF: State.PREPARE_KICKOFF,
    State.PENALTY: State.PREPARE_PENALTY,
    State.FREE_KICK: State.STOP,
}


class StateMachine:
    """Machine witch get command and return next state"""

    def __init__(self, initial_state: State = State.HALT) -> None:
        self.__state = initial_state
        self.__transitions: dict = {}
        self.__active = const.Color.ALL

        # TransNet publishes the current referee state, including when this
        # process joins an already running match or misses a preparation packet.
        referee_states = {
            Command.HALT: State.HALT,
            Command.STOP: State.STOP,
            Command.FORCE_START: State.RUN,
            Command.TIMEOUT: State.TIMEOUT,
            Command.POST_GAME: State.HALT,
            Command.PREPARE_KICKOFF: State.PREPARE_KICKOFF,
            Command.NORMAL_START: State.KICKOFF,
            Command.PREPARE_PENALTY: State.PREPARE_PENALTY,
            Command.NORMAL_START2: State.PENALTY,
            Command.FREE_KICK: State.FREE_KICK,
            Command.INDIRECT_FREE: State.FREE_KICK,
            Command.BALL_PLACEMENT: State.BALL_PLACEMENT,
        }
        for state in State:
            for command, target in referee_states.items():
                self.add_transition(state, target, command)

        self.add_transition(State.PENALTY, State.STOP, Command.PASS_10_SECONDS)

        self.add_transition(State.KICKOFF, State.RUN, Command.PASS_10_SECONDS)
        self.add_transition(State.KICKOFF, State.RUN, Command.BALL_MOVED)

        self.add_transition(State.FREE_KICK, State.RUN, Command.PASS_10_SECONDS)
        self.add_transition(State.FREE_KICK, State.RUN, Command.BALL_MOVED)

        self.add_transition(State.RUN, State.STOP, Command.STOP)

    def add_transition(self, from_state: State, to_state: State, transition: Command) -> None:
        """Add new transition from state"""
        if from_state not in self.__transitions:
            self.__transitions[from_state] = {}
        self.__transitions[from_state][transition] = to_state

    def make_transition(self, transition: int) -> None:
        """Make a transition (for user)"""
        command = CommandMap.get(transition)
        if command is None:
            self.__state = State.HALT
            return
        self.make_transition_(command)

    def make_transition_(self, transition: Optional[Command]) -> None:
        """Make a transition (for the program)"""
        if transition in self.__transitions[self.__state]:
            self.__state = self.__transitions[self.__state][transition]
        elif transition is not None:
            raise ValueError(f"No transition '{transition}' from state '{self.__state}'")

    def get_possible_transitions(self) -> list:
        """Returns a list with all possible transitions"""
        return list(self.__transitions[self.__state].keys()) if self.__state in self.__transitions else []

    def active_team(self, num: int) -> None:
        """Set active team"""
        if num == 0:
            self.__active = const.Color.ALL
        elif num == 1:
            self.__active = const.Color.BLUE
        elif num == 2:
            self.__active = const.Color.YELLOW

    def set_state(self, state: State) -> None:
        """Manually set new game state"""
        self.__state = state

    def get_state(self) -> tuple[State, const.Color]:
        """Returns the current state"""
        return self.__state, self.__active

    def __str__(self) -> str:
        return f"State: {self.__state}, Active: {self.__active}"


# SETTING ENVIRONMENT CONSTANTS
DEBUG_MODE: bool = get_from_env("DEBUG_MODE", bool)
if not DEBUG_MODE:
    const.DEBUG_HALF = 0
DEBUG_GAME_STATE: const.State = get_from_env_specific_type("DEBUG_GAME_STATE", const.State)
DEBUG_ACTIVE_TEAM: const.Color = get_from_env_specific_type("DEBUG_ACTIVE_TEAM", const.Color)
DEBUG_PREPARATION_DELAY: float = get_from_env("DEBUG_PREPARATION_DELAY", float)
# DO NOT CHANGE THESE VALUES


class RefereeStateProcessor:
    """Class to work with referee commands"""

    def __init__(self) -> None:
        """
        Инициализация
        """
        self.receiver = ZmqReceiver(port=config.REFEREE_COMMANDS_SUBSCRIBE_PORT)

        self.debug_mode = DEBUG_MODE
        self.debug_game_state = DEBUG_GAME_STATE
        self.debug_active_team = DEBUG_ACTIVE_TEAM
        self.debug_preparation_delay = DEBUG_PREPARATION_DELAY

        print(
            f"Found data about debug mode: \n\tIs used: \t\t{self.debug_mode}\n\t",
            f"DEBUG game state: \t{self.debug_game_state}\n\tDEBUG active team: \t{self.debug_active_team}\n\t",
            f"Time for preparation: \t{self.debug_preparation_delay}",
        )

        # Referee fields
        self.state_machine = StateMachine()
        self.cur_cmd_state: Optional[int] = None
        self.cur_command_key: Optional[tuple] = None
        self.restart_team = const.Color.ALL.value
        self.wait_10_sec_flag: bool = False
        self.wait_10_sec: float = 0.0
        self.wait_ball_moved_flag: bool = False
        self.ball_stop_pos: Optional[aux.Point] = aux.Point(0, 0)

        self.preparation_flag = False
        self.preparation_timer = 0.0
        if self.debug_mode:
            self.state_machine.active_team(0)

            if self.debug_game_state in [
                State.KICKOFF,
                State.FREE_KICK,
                State.PENALTY,
            ]:
                self.state_machine.active_team(self.debug_active_team.value)

                self.preparation_flag = True
                self.preparation_timer = time()

                self.state_machine.set_state(PreparationStateMap[self.debug_game_state])

            else:
                self.state_machine.set_state(self.debug_game_state)
                if self.debug_game_state in [
                    State.PREPARE_KICKOFF,
                    State.PREPARE_PENALTY,
                    State.BALL_PLACEMENT,
                ]:
                    self.state_machine.active_team(self.debug_active_team.value)

    def process(self, field: fld.Field) -> tuple[State, const.Color]:
        """
        Метод обратного вызова процесса
        """
        message = self.receiver.next_message()
        if not self.debug_mode and message is not None:
            parsed_message = json.loads(bytes(message))
            # print(parsed_message)
            cur_command = RefereeCommand(
                state=parsed_message["state"],
                commandForTeam=parsed_message["team"],
                isPartOfFieldLeft=parsed_message["is_left"],
                ball_pos=parsed_message.get("ball_pos", None),
            )

            # Old relays may omit the team on NORMAL_START; preserve the last
            # preparation's team. New relays also forward command_counter.
            team = cur_command.commandForTeam
            if cur_command.state in (Command.NORMAL_START.value, Command.NORMAL_START2.value) and team == 0:
                team = self.restart_team
            command_key = (cur_command.state, team, parsed_message.get("command_counter"))
            if command_key != self.cur_command_key:
                field.referee_command_counter = parsed_message.get("command_counter")
                self.state_machine.make_transition(cur_command.state)
                self.state_machine.active_team(team)
                self.cur_cmd_state = cur_command.state
                self.cur_command_key = command_key
                cur_state, _ = self.state_machine.get_state()
                if cur_state in (State.PREPARE_KICKOFF, State.PREPARE_PENALTY, State.KICKOFF, State.PENALTY):
                    self.restart_team = team
                else:
                    self.restart_team = 0

                if cur_state == State.BALL_PLACEMENT:
                    ball_info: Optional[tuple[float, float]] = parsed_message.get("ball_pos")
                    field.ball_placement_pos = aux.Point(ball_info[0], ball_info[1]) if ball_info is not None else None
                else:
                    field.ball_placement_pos = None

                self.wait_10_sec_flag = False
                self.wait_ball_moved_flag = False

                self.update_flags(field, cur_state)

        elif self.debug_mode and self.debug_game_state == State.BALL_PLACEMENT:
            field.ball_placement_pos = aux.Point(0, 0)

        elif self.debug_mode and self.preparation_flag and time() - self.preparation_timer > self.debug_preparation_delay:
            self.state_machine.set_state(self.debug_game_state)
            self.state_machine.active_team(self.debug_active_team.value)

            self.wait_10_sec_flag = False
            self.wait_ball_moved_flag = False

            self.update_flags(field, self.debug_game_state)

            self.preparation_flag = False

        if self.wait_10_sec_flag and time() - self.wait_10_sec > 10:
            self.state_machine.make_transition_(Command.PASS_10_SECONDS)
            self.state_machine.active_team(0)
            self.wait_10_sec_flag = False
            self.wait_ball_moved_flag = False

        if self.wait_ball_moved_flag:
            if self.ball_stop_pos is None:
                self.ball_stop_pos = field.ball.get_pos()
            elif self.is_ball_moved(field):
                self.state_machine.make_transition_(Command.BALL_MOVED)
                self.state_machine.active_team(0)
                self.wait_10_sec_flag = False
                self.wait_ball_moved_flag = False

        return self.state_machine.get_state()

    def update_flags(self, field: fld.Field, state: State) -> None:
        """Update wait_10_sec and wait_ball_moved flags"""
        if state in [
            State.KICKOFF,
            State.FREE_KICK,
            State.PENALTY,
        ]:
            self.wait_10_sec_flag = True
            self.wait_10_sec = time()

        if state in [
            State.KICKOFF,
            State.FREE_KICK,
        ]:
            self.wait_ball_moved_flag = True
            self.ball_stop_pos = field.ball.get_pos()

    def is_ball_moved(self, field: fld.Field) -> bool:
        """Return true if ball moved"""
        return self.ball_stop_pos is not None and (field.ball.get_pos() - self.ball_stop_pos).mag() > 50
