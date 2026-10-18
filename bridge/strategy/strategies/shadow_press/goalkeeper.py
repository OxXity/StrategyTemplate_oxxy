"""Goalkeeper (G): never leaves its defense zone"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, KickActions

from . import config as cfg
from . import geometry as geom
from .aim import AimLatch
from .context import Context


class Goalkeeper:
    """Goalkeeper role"""

    def __init__(self) -> None:
        self.hold_since: Optional[float] = None  # time when GK started to take the ball
        self.state = ""
        self.aim = AimLatch()

    def act(self, ctx: Context, gk: rbt.Robot) -> Action:
        """Normal game"""
        field = ctx.field

        # 1. The ball flies to our goal: move along the goal line only
        block = self.block_point(ctx)
        if block is not None:
            self.hold_since = None
            self.aim.reset()
            self.state = "BLOCK"
            angle = geom.face(gk.get_pos(), ctx.ball, gk.get_angle())
            return Actions.GoToPointIgnore(block, angle, ball_catching=True)

        # 2. The ball stopped in our zone: take it and pass to A or clear it
        if ctx.ball_at_our_gk():
            if self.hold_since is None:
                self.hold_since = ctx.now
            pass_target = ctx.pass_target(ctx.ball, gk)
            if pass_target is not None:
                self.state = "PASS"
                return KickActions.Straight(self.aim.get(ctx.ball, pass_target, ctx.now), is_pass=True)
            clear = geom.clearance_point(field, ctx.ball, ctx.forward_pos())
            if ctx.now - self.hold_since > cfg.GK_MAX_HOLD:
                self.state = "CLEAR"
                return KickActions.Straight(self.aim.get(ctx.ball, clear, ctx.now), const.VOLTAGE_SHOOT)
            # wait for A to open, holding the ball
            self.state = "HOLD"
            target = self.aim.get(ctx.ball, ctx.attacker.get_pos() if ctx.attacker is not None else clear, ctx.now)
            return Actions.BallGrab(aux.angle_to_point(ctx.ball, target))
        self.hold_since = None
        self.aim.reset()

        # 3. Stand on the arc, on the line "goal center -> enemy with ball (or ball)"
        self.state = "ARC"
        enemy = ctx.enemy_with_ball
        ref = enemy.get_pos() if enemy is not None else ctx.ball
        pos = self.arc_point(ctx, ref)
        return Actions.GoToPoint(pos, geom.face(pos, ctx.ball, gk.get_angle()))

    def block_point(self, ctx: Context, line_forw: float = cfg.GK_LINE_FORW) -> Optional[aux.Point]:
        """Point on the goal line where the ball will come, None if the ball doesn't fly to the goal"""
        field = ctx.field
        goal = field.ally_goal
        vel = field.ball.get_vel()
        if vel.mag() < cfg.ROLLING_SPEED or aux.scal_mult(vel, goal.eye_forw) >= 0:
            return None
        if field.robot_with_ball is not None:
            return None
        cross = geom.ball_cross_x(field, goal.center.x)
        if cross is None or abs(cross.y - goal.center.y) > const.GOAL_DY / 2 + const.ROBOT_R:
            return None

        line_x = goal.center.x + goal.eye_forw.x * line_forw
        point = geom.ball_cross_x(field, line_x)
        y = point.y if point is not None else cross.y
        y = aux.minmax(y, goal.center.y - const.GOAL_DY / 2, goal.center.y + const.GOAL_DY / 2)
        return aux.Point(line_x, y)

    def arc_point(self, ctx: Context, ref: aux.Point) -> aux.Point:
        """Point on the arc in front of the goal on the line "goal center -> ref" """
        goal = ctx.field.ally_goal
        angle = aux.wind_down_angle((ref - goal.center).arg() - goal.eye_forw.arg())
        angle = aux.minmax(angle, -cfg.GK_ARC_MAX_ANGLE, cfg.GK_ARC_MAX_ANGLE)
        return goal.center + aux.rotate(goal.eye_forw, angle) * cfg.GK_ARC_R

    def penalty_defense(self, ctx: Context) -> Action:
        """Their penalty: move only along the goal line"""
        goal = ctx.field.ally_goal
        line_x = goal.center.x + goal.eye_forw.x * const.ROBOT_R
        block = self.block_point(ctx, const.ROBOT_R)
        if block is not None:
            target = block
        else:
            # aim at the line "kicker -> ball"
            enemies = ctx.field.active_enemies(True)
            y = ctx.ball.y
            if enemies:
                kicker = min(enemies, key=lambda e: aux.dist(e.get_pos(), ctx.ball))
                cross = aux.get_line_intersection(
                    kicker.get_pos(), ctx.ball, aux.Point(line_x, -1), aux.Point(line_x, 1), "RL"
                )
                if cross is not None:
                    y = cross.y
            half = const.GOAL_DY / 2 - const.ROBOT_R / 2
            target = aux.Point(line_x, aux.minmax(y, goal.center.y - half, goal.center.y + half))
        self.state = "PENALTY"
        return Actions.GoToPointIgnore(target, aux.angle_to_point(target, ctx.ball), ball_catching=True)

    def stay(self, ctx: Context) -> Action:
        """Set pieces: stand on the arc in front of the goal"""
        self.hold_since = None
        self.state = "ARC"
        pos = self.arc_point(ctx, ctx.ball)
        return Actions.GoToPoint(pos, aux.angle_to_point(pos, ctx.ball))
