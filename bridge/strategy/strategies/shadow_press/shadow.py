"""Shadow (S): marks their forward F and stands on the pass line"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, KickActions

from . import config as cfg
from . import geometry as geom
from .aim import AimLatch
from .context import Context


class Shadow:
    """Shadow role"""

    def __init__(self) -> None:
        self.state = ""
        self.aim = AimLatch()
        self.hold_since: Optional[float] = None  # time when S got the ball

    def act(self, ctx: Context, robot: rbt.Robot) -> Action:
        """Normal game"""
        field = ctx.field
        forward = ctx.forward

        # 1. The ball is at F: come from our goal side and kick the ball to A or to their half
        if forward is not None and ctx.enemy_with_ball == forward:
            self.hold_since = None
            action = self.kick_away(ctx, robot, allow_hold=False)
            self.state = "TACKLE " + self.state
            return action

        # 2. S has the ball: pass to A or clear it to the flank far from F
        if ctx.has_ball(robot):
            if self.hold_since is None:
                self.hold_since = ctx.now
            return self.kick_away(ctx, robot)
        self.hold_since = None

        # 3. A pass flies to F: go to the interception point
        if forward is not None and ctx.is_rolling() and field.is_ball_moves_to_point(forward.get_pos()):
            self.state = "INTERCEPT"
            f_path = aux.dist(ctx.ball, forward.get_pos()) - 2 * const.ROBOT_R
            target = geom.intercept_point(field, robot.get_pos(), max(f_path, 0))
            target = geom.out_of_zone(target, field.ally_goal)
            angle = geom.face(robot.get_pos(), ctx.ball, robot.get_angle())
            return Actions.GoToPoint(target, angle, ball_catch=True)

        # 4. Free ball on our half and S is closer than A
        if ctx.ball_taker == "S":
            if ctx.is_rolling():
                self.state = "CATCH"
                target = geom.out_of_zone(geom.intercept_point(field, robot.get_pos()), field.ally_goal)
                angle = geom.face(robot.get_pos(), ctx.ball, robot.get_angle())
                return Actions.GoToPoint(target, angle, ball_catch=True)
            return self.kick_away(ctx, robot)

        # 5. Marking
        self.state = "MARK"
        self.aim.reset()
        return self.mark(ctx, robot)

    @property
    def on_ball(self) -> bool:
        """S is busy with the ball (A must not go to the ball)"""
        return self.state.split(" ")[0] in ["TACKLE", "INTERCEPT", "CATCH", "PASS", "HOLD", "CLEAR"]

    def kick_away(self, ctx: Context, robot: rbt.Robot, allow_hold: bool = True) -> Action:
        """
        Pass to A if the line is free; if A is still on the way and nobody presses S - wait a bit,
        otherwise clear the ball to the flank far from F
        """
        pass_target = ctx.pass_target(ctx.ball, robot, to_receive_point=True)
        if pass_target is not None:
            self.state = "PASS"
            return KickActions.Straight(self.aim.get(ctx.ball, pass_target, ctx.now), is_pass=True)

        enemies = ctx.field.active_enemies(True)
        pressed = any(aux.dist(e.get_pos(), ctx.ball) < cfg.S_HOLD_SAFE for e in enemies)
        waited = self.hold_since is not None and ctx.now - self.hold_since > cfg.S_MAX_HOLD
        if allow_hold and ctx.attacker_on_the_way() and not pressed and not waited and ctx.receive_point is not None:
            self.state = "HOLD"
            target = self.aim.get(ctx.ball, ctx.receive_point, ctx.now)
            return Actions.BallGrab(aux.angle_to_point(ctx.ball, target))

        self.state = "CLEAR"
        target = self.aim.get(ctx.ball, geom.clearance_point(ctx.field, ctx.ball, ctx.forward_pos()), ctx.now)
        return KickActions.Straight(target, const.VOLTAGE_SHOOT)

    def mark_point(self, ctx: Context) -> aux.Point:
        """P = F + d * unit(B - F), d = clamp(0.25 * |B - F|, 0.3, 0.7) m"""
        field = ctx.field
        if ctx.forward is None:
            # nobody to mark: stand between the ball and our goal
            goal = field.ally_goal.center
            point = goal + (ctx.ball - goal).unity() * min(aux.dist(goal, ctx.ball) / 2, 1500)
        else:
            f_pos = ctx.forward.get_pos()
            d = aux.minmax(cfg.SHADOW_K * aux.dist(ctx.ball, f_pos), cfg.SHADOW_MIN_D, cfg.SHADOW_MAX_D)
            point = f_pos + (ctx.ball - f_pos).unity() * d
        point = aux.nearest_point_in_poly(point, field.small_hull)
        return geom.out_of_zone(point, field.ally_goal)

    def mark(self, ctx: Context, robot: rbt.Robot) -> Action:
        """Stand on the segment from F to the ball"""
        point = self.mark_point(ctx)
        ctx.field.strategy_image.draw_circle(point, (0, 200, 255), 40)
        return Actions.GoToPoint(point, geom.face(point, ctx.ball, robot.get_angle()))
