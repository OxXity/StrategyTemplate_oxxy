"""Behavioral regressions for possession, geometry and the actual action router."""

import math
import os
import random
import unittest

# Deterministic Div C tests regardless of a developer's local .env.
os.environ.update(DIVISION="C", IS_SIMULATOR_USED="true", COLOR="BLUE", POLARITY="1", GK="0", ENEMY_GK="0")

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.router.actions import Actions, KickActions, StrategyActions
from bridge.router.actions.action import ActionDomain, ActionValues
from bridge.router.actions.dumb_actions import DumbActions
from bridge.strategy.strategy import FLIGHT_TIMEOUT, HOLD_TIME, SETUP_TIMEOUT, Strategy


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.s = Strategy(clock=lambda: self.now, rng=random.Random(4))
        self.f = fld.Field(const.Color.BLUE)
        self.f.game_state = const.State.RUN
        self.activate(self.f, (0, 1, 2))
        self.f.ball._pos = aux.Point(600, 300)

    @staticmethod
    def activate(field, ids):
        for robot in field.allies:
            robot.used(int(robot.r_id in ids))
            robot._pos = aux.Point(-1000 + robot.r_id * 700, 1000)
        field.update_active_allies([field.allies[i] for i in ids if i != field.gk_id])

    def formation(self):
        self.s.mode = "ATTACK"
        self.s.generate_attack_points(self.f)
        for role, r_id in self.s.roles.items():
            self.f.allies[r_id]._pos = self.s.attack_points[role]
        self.face_ball()

    def face_ball(self):
        for robot in self.f.active_allies(True):
            robot._angle = aux.angle_to_point(robot.get_pos(), self.f.ball.get_pos())

    def possess(self, r_id):
        robot = self.f.allies[r_id]
        self.f.ball._pos = robot.get_pos() + aux.rotate(aux.RIGHT, robot.get_angle()) * 85
        self.f.ball._vel = aux.Point(0, 0)
        self.f.robot_with_ball = robot
        self.face_ball()

    def test_full_combination_and_backpass_do_not_switch_to_defense(self):
        self.formation()
        near, far = self.s.roles["F_near"], self.s.roles["F_far"]
        self.possess(near)
        self.s.process(self.f)
        self.assertEqual(self.s.transfer.receiver, 0)
        self.assertEqual(self.s.attack_phase, "RECOVER")
        self.f.robot_with_ball = None
        self.f.ball._pos = aux.Point(-500, 100)
        self.f.ball._vel = aux.Point(-1500, -100)
        actions = self.s.process(self.f)
        self.assertEqual(self.s.mode, "ATTACK")
        self.assertNotIsInstance(actions[near], KickActions.Straight)
        self.possess(0)
        self.s.process(self.f)
        self.assertEqual(self.s.attack_phase, "PASS_1")
        self.assertEqual(self.s.transfer.receiver, far)
        self.possess(far)
        self.s.process(self.f)
        self.assertEqual(self.s.attack_phase, "PASS_2")
        self.assertEqual(self.s.transfer.receiver, near)
        self.possess(near)
        actions = self.s.process(self.f)
        self.assertEqual(self.s.attack_phase, "SHOT")
        self.assertIsNone(self.s.transfer.receiver)
        self.assertIsNone(actions[near].pass_pos)
        self.assertEqual(abs(actions[near].target_pos.y), 300)

    def test_waits_for_receiver_before_kicking(self):
        self.formation()
        self.possess(self.s.roles["F_near"])
        self.f.allies[0]._pos = aux.Point(-2000, 0)
        self.s.process(self.f)
        self.assertIsNone(self.s.transfer)
        self.assertEqual(self.s.attack_phase, "RECOVER")

    def test_unreachable_receivers_cannot_hold_the_ball_forever(self):
        self.formation()
        self.possess(self.s.roles["F_near"])
        self.f.allies[0]._pos = aux.Point(-2000, 0)
        self.f.allies[self.s.roles["F_far"]]._pos = aux.Point(-500, 1000)
        self.s.process(self.f)
        self.now = SETUP_TIMEOUT + 0.01
        self.s.process(self.f)
        self.now += HOLD_TIME + 0.01
        self.s.process(self.f)
        self.assertIsNotNone(self.s.transfer)
        self.assertIsNone(self.s.transfer.receiver)

    def test_interception_at_home_cancels_combination(self):
        self.formation()
        self.possess(self.s.roles["F_near"])
        self.s.process(self.f)
        enemy = self.f.enemies[1]
        enemy.used(1)
        self.f.robot_with_ball = enemy
        self.f.ball._pos = aux.Point(-600, 0)
        self.s.process(self.f)
        self.assertEqual(self.s.mode, "DEFENSE")
        self.assertIsNone(self.s.transfer)

    def test_interception_in_attack_starts_recovery(self):
        self.formation()
        self.s.attack_phase = "SHOT"
        self.f.enemies[1].used(1)
        self.f.robot_with_ball = self.f.enemies[1]
        self.s.process(self.f)
        self.assertEqual((self.s.mode, self.s.attack_phase), ("ATTACK", "RECOVER"))

    def test_missed_pass_timeout_does_not_lock_attack(self):
        self.formation()
        self.possess(self.s.roles["F_near"])
        self.s.process(self.f)
        self.f.robot_with_ball = None
        self.f.ball._pos = aux.Point(-700, 800)
        self.s.process(self.f)
        self.now += FLIGHT_TIMEOUT + 1
        self.s.process(self.f)
        self.s.process(self.f)
        self.assertEqual(self.s.mode, "DEFENSE")

    def test_two_robots_skip_far_forward(self):
        self.activate(self.f, (0, 2))
        self.formation()
        self.s.attack_phase = "PASS_1"
        self.possess(0)
        self.s.process(self.f)
        self.assertEqual(self.s.transfer.receiver, 2)
        self.possess(2)
        self.s.process(self.f)
        self.assertEqual(self.s.attack_phase, "SHOT")

    def test_missing_robots_and_nonzero_keeper(self):
        for ids in ((), (0,), (1,), (1, 2), (0, 1, 2)):
            with self.subTest(ids=ids):
                self.activate(self.f, ids)
                self.f.robot_with_ball = None
                actions = self.s.process(self.f)
                self.assertTrue(all(actions[i] is not None for i in ids))
        self.f.gk_id = 2
        self.activate(self.f, (0, 1, 2))
        self.s.generate_attack_points(self.f)
        self.assertEqual(self.s.roles["P0"], 2)

    def test_attack_geometry_for_both_sides_and_minimal_assignment(self):
        for color in (const.Color.BLUE, const.Color.YELLOW):
            for seed in range(30):
                field = fld.Field(color)
                self.activate(field, (0, 1, 2))
                strategy = Strategy(rng=random.Random(seed))
                strategy.generate_attack_points(field)
                near, far = (strategy.attack_points[role] for role in ("F_near", "F_far"))
                self.assertLess(near.y * far.y, 0)
                self.assertGreater(strategy.attack_depth(field, near), strategy.attack_depth(field, far))
                for point in (near, far):
                    self.assertTrue(aux.is_point_inside_poly(point, field.so_small_hull))
                    self.assertFalse(aux.is_point_inside_poly(point, field.enemy_goal.big_hull))
                    self.assertFalse(aux.is_point_inside_poly(point, field.ally_goal.big_hull))
                n, f = strategy.roles["F_near"], strategy.roles["F_far"]
                cost = aux.dist(field.allies[n].get_pos(), near) + aux.dist(field.allies[f].get_pos(), far)
                other = aux.dist(field.allies[f].get_pos(), near) + aux.dist(field.allies[n].get_pos(), far)
                self.assertLessEqual(cost, other)

    def test_half_hysteresis_and_stable_layout(self):
        self.s.process(self.f)
        points = self.s.attack_points.copy()
        for x in (50, -50, 180, -180):
            self.f.ball._pos = aux.Point(x, 0)
            self.s.process(self.f)
            self.assertEqual(self.s.mode, "ATTACK")
            self.assertEqual(self.s.attack_points, points)
        self.f.ball._pos = aux.Point(-250, 0)
        self.s.process(self.f)
        self.assertEqual(self.s.mode, "DEFENSE")

    def test_blocked_primary_uses_other_receiver(self):
        self.formation()
        self.s.attack_phase = "PASS_1"
        self.possess(0)
        far = self.f.allies[self.s.roles["F_far"]]
        enemy = self.f.enemies[1]
        enemy.used(1)
        enemy._pos = aux.lerp(self.f.ball.get_pos(), far.get_pos(), 0.8)
        self.f.update_active_enemies([enemy])
        self.s.process(self.f)
        self.assertEqual(self.s.transfer.receiver, self.s.roles["F_near"])

    def test_all_blocked_hold_then_shoot(self):
        self.formation()
        self.s.attack_phase = "PASS_1"
        self.possess(0)
        enemy = self.f.enemies[1]
        enemy.used(1)
        enemy._pos = self.f.ball.get_pos() + aux.Point(100, 0)
        self.f.update_active_enemies([enemy])
        actions = self.s.process(self.f)
        self.assertIsInstance(actions[0], Actions.BallGrab)
        self.now = HOLD_TIME + 0.01
        actions = self.s.process(self.f)
        self.assertIsInstance(actions[0], KickActions.Straight)
        self.assertIsNone(actions[0].pass_pos)

    def test_enemy_behind_segment_does_not_block_pass(self):
        enemy = self.f.enemies[1]
        enemy.used(1)
        enemy._pos = aux.Point(-500, 0)
        self.f.update_active_enemies([enemy])
        self.assertFalse(self.s.is_pass_blocked(self.f, aux.Point(0, 0), aux.Point(1000, 0)))
        enemy._pos = aux.Point(500, 100)
        self.assertTrue(self.s.is_pass_blocked(self.f, aux.Point(0, 0), aux.Point(1000, 0)))

    def test_keeper_handles_penalty_ball_without_forward_chasing(self):
        for x in (-1900, -1700):
            self.f.ball._pos = aux.Point(x, 100)
            actions = self.s.process(self.f)
            self.assertIsInstance(actions[0], KickActions.Straight)
            self.assertIsNotNone(actions[0].pass_pos)
            self.assertTrue(all(isinstance(actions[i], Actions.GoToPoint) for i in (1, 2)))

    def test_shot_aims_away_from_enemy_keeper(self):
        keeper = self.f.enemies[0]
        keeper.used(1)
        for y in (-200, 200):
            keeper._pos = aux.Point(2100, y)
            self.assertLess(self.s.choose_shot_target(self.f).y * y, 0)

    def test_halt_replaces_previous_actions(self):
        self.s.process(self.f)
        for state in (const.State.HALT, const.State.TIMEOUT):
            self.f.game_state = state
            actions = self.s.process(self.f)
            self.assertTrue(all(isinstance(actions[i], Actions.Stop) for i in (0, 1, 2)))
            self.assertIsNone(self.s.transfer)

    def test_stop_has_no_kick_or_dribbler_through_router(self):
        self.f.game_state = const.State.STOP
        self.f.ball._pos = aux.Point(0, 0)
        actions = self.s.process(self.f)
        for robot in self.f.active_allies(True):
            values = ActionValues()
            actions[robot.r_id].process(ActionDomain(self.f, const.State.STOP, True, robot), values)
            self.assertEqual(values.auto_kick, 0)
            self.assertEqual(values.dribbler_speed, 0)
            self.assertLessEqual(values.vel.mag(), const.STOP_SPEED + 1e-6)
        self.assertGreaterEqual(const.KEEP_BALL_DIST, 500 + const.ROBOT_R)

    def test_long_pass_voltage_is_limited_in_actual_action(self):
        robot = self.f.allies[0]
        robot._pos = aux.Point(0, 0)
        self.f.ball._pos = aux.Point(85, 0)
        domain = ActionDomain(self.f, const.State.RUN, True, robot)
        for length in (500, 2000, 4500, 8000):
            values = ActionValues()
            action = DumbActions.ControlVoltageAction(pass_pos=aux.Point(length, 0))
            action.process(domain, values)
            self.assertLessEqual(values.kicker_voltage * 0.4, 3)

    def test_goal_detection_uses_future_velocity_and_goal_mouth(self):
        self.f.ball._pos = aux.Point(-1000, 0)
        self.f.ball_start_point = aux.Point(-2000, 0)
        self.f.ball._vel = aux.Point(1000, 0)
        self.assertFalse(self.f.is_ball_moves_to_goal())
        self.f.ball._vel = aux.Point(-1000, 0)
        self.assertTrue(self.f.is_ball_moves_to_goal())
        self.f.ball._pos = aux.Point(-1000, 600)
        self.assertFalse(self.f.is_ball_moves_to_goal())

    def test_catch_uses_forward_ray_and_legal_router(self):
        self.f.ball._pos = aux.Point(0, 0)
        self.f.ball._vel = aux.Point(1000, 0)
        self.f.ball_start_point = aux.Point(0, 1000)  # deliberately stale history
        robot = self.f.allies[1]
        robot._pos = aux.Point(1000, 200)
        result = StrategyActions.CatchBall().use_behavior_of(
            ActionDomain(self.f, const.State.RUN, True, robot), ActionValues()
        )[0]
        self.assertIsInstance(result, Actions.GoToPoint)
        self.assertAlmostEqual(abs(result.target_angle), math.pi)
        self.assertAlmostEqual(result.target_pos.y, 0)
        self.assertAlmostEqual(result.target_pos.x, 1000 + const.GRAB_DIST)

    def test_receiver_meets_a_pass_that_stops_short(self):
        self.formation()
        self.s.attack_phase = "PASS_1"
        self.possess(0)
        self.s.process(self.f)
        receiver = self.f.allies[self.s.transfer.receiver]
        self.f.robot_with_ball = None
        self.f.ball._pos = aux.Point(0, 0)
        self.f.ball._vel = aux.Point(400, 0)
        self.s.transfer.released = self.now
        action = self.s.receive_ball(self.f, receiver, receiver.get_pos())
        self.assertIsInstance(action, StrategyActions.CatchBall)
        result = action.use_behavior_of(ActionDomain(self.f, const.State.RUN, True, receiver), ActionValues())[0]
        self.assertLessEqual(result.target_pos.x, 160 + const.GRAB_DIST)

    def test_grsim_contact_is_recognized_as_possession(self):
        robot = self.f.allies[1]
        robot._pos, robot._angle = aux.Point(0, 0), 0
        self.f.ball._pos = aux.Point(115, 0)
        self.assertTrue(self.f._is_ball_in(robot))
        self.f.ball._pos = aux.Point(0, 115)
        self.assertFalse(self.f._is_ball_in(robot))

    def test_kicker_waits_for_precise_aim_and_front_contact(self):
        robot = self.f.allies[1]
        robot._pos, robot._angle = aux.Point(0, 0), 0.1
        self.f.ball._pos = aux.Point(115, 0)
        shot = DumbActions.ShootAction(aux.Point(2250, 0))
        domain = ActionDomain(self.f, const.State.RUN, True, robot)
        self.assertFalse(shot.is_defined(domain))
        robot._angle = 0
        self.assertTrue(shot.is_defined(domain))
        self.f.ball._pos = aux.Point(115, 80)
        self.assertFalse(shot.is_defined(domain))
        self.f.ball._pos = aux.Point(115, 0)
        domain.game_state = const.State.STOP
        self.assertFalse(shot.is_defined(domain))


if __name__ == "__main__":
    unittest.main()
