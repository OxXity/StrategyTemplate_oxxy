"""Attacker (A): presses, shoots and dribbles with pushes"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, KickActions

from . import config as cfg
from . import geometry as geom
from .aim import AimLatch
from .context import Context


class Attacker:
    """Attacker role"""

    def __init__(self) -> None:
        self.stuck_anchor: Optional[aux.Point] = None  # ball position when the fight for the ball started
        self.stuck_since = 0.0
        self.state = ""
        self.aim = AimLatch()

    def act(self, ctx: Context, robot: rbt.Robot) -> Action:
        """Normal game"""
        field = ctx.field

        # 1. The ball is at their GK: block the line "ball -> F", don't touch the ball in their zone
        if ctx.ball_at_their_gk():
            self.reset_stuck()
            self.aim.reset()
            self.state = "BLOCK GK"
            return self.block_line(ctx, cfg.ATTACKER_GK_BLOCK_DIST)

        # 2. The ball is at our GK: open for a pass near the center line
        if ctx.ball_at_our_gk():
            self.reset_stuck()
            self.aim.reset()
            self.state = "OPEN"
            point = self.exit_point(ctx)
            return Actions.GoToPoint(point, geom.face(point, ctx.ball, robot.get_angle()))

        # 2.1 S takes the ball: wait for the pass on the way to their goal, don't crowd at the ball
        s_busy = ctx.ball_taker == "S" or ctx.has_ball(ctx.shadow) or ctx.shadow_on_ball
        if s_busy and ctx.receive_point is not None and not self.pass_comes(ctx, robot):
            self.reset_stuck()
            self.aim.reset()
            self.state = "RECEIVE"
            point = ctx.receive_point
            # already turned to the goal as much as the dribbler can catch the ball
            return Actions.GoToPoint(point, geom.receive_angle(point, ctx.ball, field.enemy_goal.center))

        # 2.2 The pass comes to A: shift along the pass line and meet the ball at an angle
        if self.pass_comes(ctx, robot):
            self.reset_stuck()
            self.aim.reset()
            self.state = "RECEIVE PASS"
            return self.receive_pass(ctx, robot)

        # 3. The ball rolls: go to the interception point
        #    (except the own push: the ball rolls away from A nearby -> keep attacking, BallGrab chases it)
        if ctx.is_rolling() and not ctx.has_ball(robot) and not self.is_chasing(ctx, robot):
            self.reset_stuck()
            self.aim.reset()
            self.state = "INTERCEPT"
            target = geom.intercept_point(field, robot.get_pos())
            target = geom.out_of_zone(geom.out_of_zone(target, field.enemy_goal), field.ally_goal)
            angle = geom.face(robot.get_pos(), ctx.ball, robot.get_angle())
            return Actions.GoToPoint(target, angle, ball_catch=True)

        return self.attack(ctx, robot)

    def attack(self, ctx: Context, robot: rbt.Robot) -> Action:
        """Go behind the ball facing their goal, shoot or dribble"""
        field = ctx.field
        goal = field.enemy_goal

        # Stuck longer than STUCK_TIME: shoot as is
        if self.is_stuck(ctx, robot):
            self.state = "STUCK SHOT"
            return KickActions.Straight(self.aim.get(ctx.ball, goal.center, ctx.now), const.VOLTAGE_SHOOT)

        # Free window and close to the goal: shoot at max to the middle of the widest window
        window = geom.widest_window(ctx.ball, goal, ctx.obstacles(robot))
        if window is not None and aux.dist(ctx.ball, goal.center) < cfg.SHOT_DIST:
            self.state = "SHOT"
            target = self.aim.get(ctx.ball, window[0], ctx.now)
            field.strategy_image.draw_line(ctx.ball, target, (255, 0, 0), 3)
            return self.turn_with_ball(ctx, robot, target) or KickActions.Straight(target, const.VOLTAGE_SHOOT)

        # No window: dribble with pushes to the goal by the freer flank
        flank = geom.free_flank_point(field, ctx.ball)
        if aux.dist(ctx.ball, flank) < cfg.DRIBBLE_STEP / 2:
            self.state = "FLANK SHOT"
            target = self.aim.get(ctx.ball, goal.center, ctx.now)
            return self.turn_with_ball(ctx, robot, target) or KickActions.Straight(target, const.VOLTAGE_SHOOT)
        self.state = "DRIBBLE"
        target = self.aim.get(ctx.ball, ctx.ball + (flank - ctx.ball).unity() * cfg.DRIBBLE_STEP, ctx.now)
        field.strategy_image.draw_line(ctx.ball, target, (255, 255, 0), 2)
        return self.turn_with_ball(ctx, robot, target) or KickActions.Straight(target, is_pass=True)

    def turn_with_ball(self, ctx: Context, robot: rbt.Robot, target: aux.Point) -> Optional[Action]:
        """
        A holds the ball, but looks far from the target: turn smoothly around the ball
        (a fast turn throws the ball away), None if the turn isn't needed
        """
        if not ctx.has_ball(robot):
            return None
        diff = aux.wind_down_angle(aux.angle_to_point(ctx.ball, target) - robot.get_angle())
        if abs(diff) < cfg.TURN_WITH_BALL_ANGLE:
            return None
        self.state += " TURN"
        return Actions.BallGrab(robot.get_angle() + aux.minmax(diff, -cfg.TURN_LOOKAHEAD, cfg.TURN_LOOKAHEAD))

    def receive_pass(self, ctx: Context, robot: rbt.Robot) -> Action:
        """Meet the pass: stand on the ball line, dribbler on, turned to the goal by RECEIVE_ANGLE_OFFSET"""
        field = ctx.field
        ball_vel = field.ball.get_vel()
        on_line = aux.closest_point_on_line(ctx.ball, ctx.ball + ball_vel, robot.get_pos(), "R")
        if aux.dist(on_line, robot.get_pos()) < cfg.RECEIVE_LINE_DIST:
            target = on_line
        else:
            target = geom.intercept_point(field, robot.get_pos())
        target = geom.out_of_zone(geom.out_of_zone(target, field.enemy_goal), field.ally_goal)
        # the ball comes along -vel: face it, turned to the goal
        ball_from = target - ball_vel.unity() * 1000
        angle = geom.receive_angle(target, ball_from, field.enemy_goal.center)
        return Actions.GoToPoint(target, angle, ball_catch=True).compose(Actions.SetDribblerSpeed(15))

    def is_stuck(self, ctx: Context, robot: rbt.Robot) -> bool:
        """A is near the ball, but the ball doesn't move for STUCK_TIME"""
        if aux.dist(robot.get_pos(), ctx.ball) > cfg.STUCK_NEAR:
            self.reset_stuck()
            return False
        if self.stuck_anchor is None or aux.dist(self.stuck_anchor, ctx.ball) > cfg.STUCK_RADIUS:
            self.stuck_anchor = ctx.ball
            self.stuck_since = ctx.now
            return False
        return ctx.now - self.stuck_since > cfg.STUCK_TIME

    def pass_comes(self, ctx: Context, robot: rbt.Robot) -> bool:
        """The ball rolls to A (the pass is made): go to receive it"""
        return ctx.is_rolling() and ctx.field.is_ball_moves_to_point(robot.get_pos())

    def is_chasing(self, ctx: Context, robot: rbt.Robot) -> bool:
        """The ball rolls away from A nearby (after its own push)"""
        to_ball = ctx.ball - robot.get_pos()
        return to_ball.mag() < cfg.CHASE_DIST and aux.scal_mult(ctx.field.ball.get_vel(), to_ball) > 0

    def reset_stuck(self) -> None:
        """Reset the stuck timer"""
        self.stuck_anchor = None

    def block_point(self, ctx: Context, dist: float) -> aux.Point:
        """Point on the line "ball -> F" at the distance dist from the ball"""
        field = ctx.field
        ref = ctx.forward_pos()
        if ref is None:
            ref = field.ally_goal.center
        point = ctx.ball + (ref - ctx.ball).unity() * dist
        # keep the distance to their zone
        return geom.out_of_zone(point, field.enemy_goal)

    def block_line(self, ctx: Context, dist: float) -> Action:
        """Stand on the line "ball -> F" at the distance dist from the ball"""
        point = self.block_point(ctx, dist)
        return Actions.GoToPoint(point, geom.face(point, ctx.ball, aux.angle_to_point(point, ctx.ball)))

    def exit_point(self, ctx: Context) -> aux.Point:
        """Point for the pass near the center line, on the flank opposite to F"""
        field = ctx.field
        f_pos = ctx.forward_pos()
        side = -aux.sign(f_pos.y) if f_pos is not None and f_pos.y != 0 else 1
        return aux.Point(cfg.EXIT_X * field.polarity, side * cfg.EXIT_Y)

    def kickoff_shot(self, ctx: Context, robot: rbt.Robot) -> Action:
        """Our kickoff: shoot at the widest window at once"""
        self.state = "KICKOFF SHOT"
        goal = ctx.field.enemy_goal
        window = geom.widest_window(ctx.ball, goal, ctx.obstacles(robot))
        target = self.aim.get(ctx.ball, window[0] if window is not None else goal.center, ctx.now)
        return KickActions.Straight(target, const.VOLTAGE_SHOOT)

    def free_kick(self, ctx: Context, robot: rbt.Robot) -> Action:
        """Our free kick: shoot if there is a window, otherwise clear to the free flank of their half"""
        goal = ctx.field.enemy_goal
        window = geom.widest_window(ctx.ball, goal, ctx.obstacles(robot))
        if window is not None:
            self.state = "FREE KICK SHOT"
            return KickActions.Straight(self.aim.get(ctx.ball, window[0], ctx.now), const.VOLTAGE_SHOOT)
        self.state = "FREE KICK CLEAR"
        target = self.aim.get(ctx.ball, geom.clearance_point(ctx.field, ctx.ball, ctx.forward_pos()), ctx.now)
        return KickActions.Straight(target, is_pass=True)

    def penalty(self, ctx: Context) -> Action:
        """Our penalty: push the ball closer to the goal, then shoot at full power to the free corner"""
        goal = ctx.field.enemy_goal
        if aux.dist(ctx.ball, goal.center) > cfg.PENALTY_SHOOT_DIST:
            self.state = "PENALTY PUSH"
            target = ctx.ball + (goal.center - ctx.ball).unity() * cfg.PENALTY_PUSH
            return KickActions.Straight(target, is_pass=True)
        self.state = "PENALTY SHOT"
        return KickActions.Straight(
            self.aim.get(ctx.ball, geom.free_corner(ctx.field, ctx.ball), ctx.now), const.VOLTAGE_SHOOT
        )
