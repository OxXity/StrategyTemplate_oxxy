"""Offline regressions for referee-driven starts and real command conversion."""

import json
import math
import unittest
from unittest.mock import Mock, patch

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.processors import referee_state_processor as referee
from bridge.processors.field_creator import FieldCreator
from bridge.processors.router_processor import command_from_values
from bridge.router.actions.action import ActionDomain, ActionValues
from bridge.strategy.strategy import Strategy


def make_field(color=const.Color.BLUE):
    field = fld.Field(color)
    forward = field.ally_goal.eye_forw
    field.ball._pos = aux.Point(0, 0)
    for r_id, point in enumerate((-forward * 2100, -forward * 120, -forward * 900)):
        robot = field.allies[r_id]
        robot.used(1)
        robot._pos = point
        robot._angle = forward.arg()
    field.update_active_allies([field.allies[1], field.allies[2]])
    return field


def values_for(field, r_id, action):
    values = ActionValues()
    action.process(ActionDomain(field, field.game_state, field.active_team == field.ally_color, field.allies[r_id]), values)
    return values


class RefereeTests(unittest.TestCase):
    def setUp(self):
        self.receiver = Mock()
        self.receiver.next_message.return_value = None
        with patch.object(referee, "ZmqReceiver", return_value=self.receiver), patch.object(
            referee, "DEBUG_MODE", False
        ), patch("builtins.print"):
            self.processor = referee.RefereeStateProcessor()
        self.field = make_field()

    def receive(self, state, team=0, counter=0, now=100):
        self.receiver.next_message.return_value = json.dumps(
            dict(state=state, team=team, command_counter=counter, is_left=False)
        ).encode()
        with patch.object(referee, "time", return_value=now):
            self.field.game_state, self.field.active_team = self.processor.process(self.field)
        return self.field.game_state, self.field.active_team

    def test_indirect_free_kick_after_stop_and_late_join(self):
        self.receive(1)
        self.assertEqual(self.receive(10, 1, 1), (const.State.FREE_KICK, const.Color.BLUE))
        machine = referee.StateMachine()
        machine.make_transition(10)
        self.assertEqual(machine.get_state()[0], const.State.FREE_KICK)

    def test_all_referee_states_can_be_received_after_reconnect(self):
        expected = {
            0: const.State.HALT,
            1: const.State.STOP,
            2: const.State.RUN,
            3: const.State.TIMEOUT,
            4: const.State.HALT,
            5: const.State.PREPARE_KICKOFF,
            6: const.State.KICKOFF,
            7: const.State.PREPARE_PENALTY,
            8: const.State.PENALTY,
            9: const.State.FREE_KICK,
            10: const.State.FREE_KICK,
            11: const.State.BALL_PLACEMENT,
        }
        for start in const.State:
            for command, state in expected.items():
                with self.subTest(start=start, command=command):
                    machine = referee.StateMachine(start)
                    machine.make_transition(command)
                    self.assertEqual(machine.get_state()[0], state)

    def test_normal_start_keeps_owner_with_legacy_relay(self):
        self.receive(5, 2, 1)
        self.assertEqual(self.receive(6, 0, 2), (const.State.KICKOFF, const.Color.YELLOW))

    def test_repeated_packet_does_not_reenter_kickoff_after_ball_moves(self):
        self.receive(5, 1, 1)
        self.receive(6, 1, 2)
        self.field.ball._pos = aux.Point(100, 0)
        self.assertEqual(self.receive(6, 1, 2)[0], const.State.RUN)
        self.assertEqual(self.receive(6, 1, 2)[0], const.State.RUN)

    def test_new_counter_restarts_timer_and_team_change_is_not_lost(self):
        self.receive(9, 1, 1)
        self.assertEqual(self.receive(9, 2, 1, now=105)[1], const.Color.YELLOW)
        self.receive(9, 2, 2, now=109)
        self.assertEqual(self.receive(9, 2, 2, now=112)[0], const.State.FREE_KICK)
        self.assertEqual(self.processor.wait_10_sec, 109)

    def test_halt_is_published_even_without_new_vision(self):
        creator = FieldCreator()
        creator.field = self.field
        creator.field.game_state = const.State.RUN
        creator.process_field = Mock()
        creator.referee_processor = Mock()
        creator.referee_processor.process.return_value = (const.State.HALT, const.Color.ALL)
        creator.field_writer = Mock()
        creator.process()
        snapshot = creator.field_writer.write.call_args.args[0]
        self.assertEqual(snapshot.game_state, const.State.HALT)


class RestartTests(unittest.TestCase):
    def test_referee_owner_is_retained_when_start_state_reports_all_teams(self):
        for prepared, started in (
            (const.State.PREPARE_KICKOFF, const.State.KICKOFF),
            (const.State.PREPARE_PENALTY, const.State.PENALTY),
        ):
            with self.subTest(started=started):
                field = make_field()
                field.game_state, field.active_team = prepared, const.Color.BLUE
                strategy = Strategy()
                strategy.process(field)
                self.assertTrue(strategy.we_active)
                field.game_state, field.active_team = started, const.Color.ALL
                strategy.process(field)
                self.assertTrue(strategy.we_active)

    def test_stop_dispatches_normal_run_logic(self):
        field = make_field()
        field.game_state = const.State.STOP
        strategy = Strategy()
        strategy.run = unittest.mock.Mock()
        strategy.process(field)
        strategy.run.assert_called_once_with(field, unittest.mock.ANY)

    def test_restart_selects_kicker_when_vision_appears_after_referee(self):
        field = make_field()
        field.game_state, field.active_team = const.State.KICKOFF, field.ally_color
        for robot in field.allies:
            robot.used(0)
        field.update_active_allies([])
        strategy = Strategy()
        strategy.process(field)
        self.assertIsNone(strategy.restart_kicker)
        for r_id in (0, 1, 2):
            field.allies[r_id].used(1)
        field.update_active_allies([field.allies[1], field.allies[2]])
        strategy.process(field)
        self.assertEqual(strategy.restart_kicker, 1)

    def ready_restart(self, state, color=const.Color.BLUE):
        field = make_field(color)
        field.game_state = state
        field.active_team = color
        strategy = Strategy()
        actions = strategy.process(field)
        for r_id in (0, 2):
            field.allies[r_id]._pos = actions[r_id].target_pos
            field.allies[r_id]._angle = aux.angle_to_point(field.allies[r_id].get_pos(), field.ball.get_pos())
        target = strategy.choose_shot_target(field) if state == const.State.PENALTY else field.allies[2].get_pos()
        angle = aux.angle_to_point(field.ball.get_pos(), target)
        kicker = field.allies[1]
        kicker._angle = angle
        kicker._pos = field.ball.get_pos() - aux.rotate(aux.RIGHT, angle) * 100
        field.robot_with_ball = kicker
        return field, strategy

    def test_kickoff_and_free_kick_make_a_pass_for_both_teams(self):
        for color in (const.Color.BLUE, const.Color.YELLOW):
            for state in (const.State.KICKOFF, const.State.FREE_KICK):
                with self.subTest(color=color, state=state):
                    field, strategy = self.ready_restart(state, color)
                    actions = strategy.process(field)
                    self.assertEqual(strategy.transfer.receiver, 2)
                    self.assertEqual(values_for(field, 1, actions[1]).auto_kick, 1)

    def test_penalty_shoots_without_waiting_for_run(self):
        field, strategy = self.ready_restart(const.State.PENALTY)
        actions = strategy.process(field)
        self.assertIsNone(strategy.transfer.receiver)
        self.assertEqual(values_for(field, 1, actions[1]).auto_kick, 1)

    def test_preparation_and_opponent_restarts_never_fire(self):
        for state in (
            const.State.PREPARE_KICKOFF,
            const.State.PREPARE_PENALTY,
            const.State.KICKOFF,
            const.State.FREE_KICK,
            const.State.PENALTY,
        ):
            for owner in (const.Color.BLUE, const.Color.YELLOW, const.Color.ALL):
                if owner == const.Color.BLUE and state in (const.State.KICKOFF, const.State.FREE_KICK, const.State.PENALTY):
                    continue
                with self.subTest(state=state, owner=owner):
                    field = make_field()
                    field.game_state, field.active_team = state, owner
                    actions = Strategy().process(field)
                    for r_id in (0, 1, 2):
                        self.assertEqual(values_for(field, r_id, actions[r_id]).auto_kick, 0)

    def test_kickoff_support_stays_on_own_half(self):
        for color in (const.Color.BLUE, const.Color.YELLOW):
            field = make_field(color)
            field.game_state, field.active_team = const.State.PREPARE_KICKOFF, color
            actions = Strategy().process(field)
            for r_id in (0, 1, 2):
                self.assertLessEqual(aux.scal_mult(actions[r_id].target_pos, field.ally_goal.eye_forw), 0)

    def test_transfer_survives_switch_to_run_and_halt_cancels_it(self):
        field, strategy = self.ready_restart(const.State.KICKOFF)
        strategy.process(field)
        transfer = strategy.transfer
        field.game_state, field.active_team = const.State.RUN, const.Color.ALL
        strategy.process(field)
        self.assertIs(strategy.transfer, transfer)
        field.game_state = const.State.HALT
        actions = strategy.process(field)
        self.assertIsNone(strategy.transfer)
        self.assertTrue(all(action is not None for action in actions))
        for r_id in (0, 1, 2):
            values = values_for(field, r_id, actions[r_id])
            self.assertEqual(
                (values.vel.mag(), values.angle, values.auto_kick, values.dribbler_speed, values.kicker_voltage),
                (0, 0, 0, 0, 0),
            )


class TurnConversionTests(unittest.TestCase):
    def test_slow_turn_keeps_local_velocity_and_turn_direction(self):
        for simulator in (False, True):
            for heading in (-math.pi / 2, 0, math.pi / 2, math.pi):
                for turn in (-2.0, 2.0):
                    with self.subTest(simulator=simulator, heading=heading, turn=turn), patch.object(
                        const, "IS_SIMULATOR_USED", simulator
                    ):
                        field = make_field()
                        field.game_state = const.State.RUN
                        robot = field.allies[1]
                        robot._pos = aux.Point(0, 0)
                        robot._angle = heading
                        field.ball._pos = aux.rotate(aux.RIGHT, heading) * 100
                        target = field.ball.get_pos() + aux.rotate(aux.RIGHT, heading + turn) * 1000
                        action = Strategy.turn_with_ball(field, robot, target)
                        values = values_for(field, 1, action)
                        self.assertEqual(values.beep, 0)
                        self.assertLessEqual(values.vel.mag(), 250)
                        cmd = command_from_values(field, robot, values)
                        local = aux.rotate(values.vel, -heading)
                        self.assertAlmostEqual(cmd.forward_vel, local.x)
                        self.assertAlmostEqual(cmd.left_vel, local.y)
                        self.assertLessEqual(abs(cmd.angular_vel), 3)
                        self.assertGreater(cmd.angular_vel * turn * (1 if simulator else -1), 0)
                        self.assertIsNone(cmd.angle)


if __name__ == "__main__":
    unittest.main()
