"""Router-level regressions for controlled possession, turning, and kicking."""

import math
import os
import unittest

os.environ.update(
    DIVISION="C",
    IS_SIMULATOR_USED="true",
    COLOR="BLUE",
    POLARITY="1",
    GK="0",
    ENEMY_GK="0",
)

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.router.actions.action import ActionDomain, ActionValues
from bridge.strategy.strategy import Strategy, Transfer


class BallControlTests(unittest.TestCase):
    def setUp(self):
        self.field = fld.Field(const.Color.BLUE)
        self.field.game_state = const.State.RUN
        self.robot = self.field.allies[1]
        self.robot.used(1)
        self.robot._pos = aux.Point(0, 0)
        self.robot._angle = 0
        self.field.ball._pos = aux.Point(115, 0)
        self.field.ball._vel = aux.Point(0, 0)
        self.strategy = Strategy()
        self.domain = ActionDomain(self.field, const.State.RUN, True, self.robot)

    def process(self, action):
        values = ActionValues()
        action.process(self.domain, values)
        return values

    def test_turn_keeps_dribbler_on_and_limits_alignment_motion(self):
        action = self.strategy.turn_with_ball(
            self.field, self.robot, aux.Point(0, 1000)
        )

        values = self.process(action)

        self.assertEqual(values.dribbler_speed, 15)
        self.assertEqual(values.auto_kick, 0)
        self.assertLessEqual(values.vel.mag(), 350 + 1e-6)
        self.assertGreater(abs(values.angle), 1.0)

    def test_collector_uses_dribbler_while_approaching_ball(self):
        self.field.ball._pos = aux.Point(600, 0)
        action = self.strategy.collect_ball(self.field, self.robot, aux.Point(2250, 0))

        values = self.process(action)

        self.assertEqual(values.dribbler_speed, 15)
        self.assertGreater(values.vel.mag(), 0)

    def test_aligned_attacker_can_kick_through_real_action_router(self):
        action = self.strategy.kick_with_alignment(self.field, aux.Point(2250, 0))

        values = self.process(action)

        self.assertEqual(values.auto_kick, 1)
        self.assertEqual(values.dribbler_speed, 15)
        self.assertLessEqual(values.vel.mag(), 500 + 1e-6)

    def test_unaligned_attacker_rotates_without_firing_early(self):
        self.robot._angle = 0.2
        action = self.strategy.kick_with_alignment(self.field, aux.Point(2250, 0))

        values = self.process(action)

        self.assertEqual(values.auto_kick, 0)
        self.assertEqual(values.dribbler_speed, 15)
        self.assertLessEqual(values.vel.mag(), 500 + 1e-6)
        self.assertAlmostEqual(values.angle, 0)

    def test_pass_aim_controls_dribbler_and_uses_pass_voltage(self):
        target = aux.Point(math.cos(0.5) * 1115, math.sin(0.5) * 1115)
        self.robot._angle = 0.5
        self.field.ball._pos = aux.Point(115 * math.cos(0.5), 115 * math.sin(0.5))
        action = self.strategy.kick_with_alignment(self.field, target, is_pass=True)

        values = self.process(action)

        self.assertEqual(values.auto_kick, 1)
        self.assertEqual(values.dribbler_speed, 15)
        self.assertLessEqual(values.kicker_voltage, const.VOLTAGE_SHOOT)

    def test_backpass_to_keeper_does_not_cancel_attack_as_own_goal_threat(self):
        self.strategy.mode = "ATTACK"
        self.strategy.attack_passes = 3
        self.strategy.transfer = Transfer(
            1, 0, aux.Point(-1200, 0), "POSSESSION", aux.Point(-900, 0), 0
        )
        self.field.ball._pos = aux.Point(-900, 0)
        self.field.ball._vel = aux.Point(-1000, 0)

        self.strategy.update_mode(self.field)

        self.assertTrue(self.field.is_ball_moves_to_goal())
        self.assertEqual(self.strategy.mode, "ATTACK")
        self.assertEqual(self.strategy.attack_passes, 3)
        self.assertIsNotNone(self.strategy.transfer)

    def test_unassigned_kick_toward_own_goal_switches_to_defense(self):
        self.strategy.mode = "ATTACK"
        self.strategy.transfer = Transfer(
            1, None, aux.Point(-2250, 0), "RECOVER", aux.Point(-900, 0), 0
        )
        self.field.ball._pos = aux.Point(-900, 0)
        self.field.ball._vel = aux.Point(-1000, 0)

        self.strategy.update_mode(self.field)

        self.assertEqual(self.strategy.mode, "DEFENSE")
        self.assertIsNone(self.strategy.transfer)

    def test_shot_selector_prefers_open_center_over_a_corner(self):
        self.field.ball._pos = aux.Point(1000, 0)
        defender = self.field.enemies[1]
        defender.used(1)
        defender._pos = aux.Point(2150, 276)
        self.field.update_active_enemies([defender])

        target = self.strategy.choose_shot_target(self.field)

        self.assertAlmostEqual(target.y, self.field.enemy_goal.center.y)
        self.assertFalse(
            self.strategy.is_pass_blocked(self.field, self.field.ball.get_pos(), target)
        )


if __name__ == "__main__":
    unittest.main()
