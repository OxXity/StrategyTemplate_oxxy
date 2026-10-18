"""
BALL_PLACEMENT (ТЗ, 9.8)

По умолчанию мяч ставит судья (Регламент 5.2): все освобождают «стадион» «мяч -> цель».
P2: AUTO_BALL_PLACEMENT=True и мы - активная команда: ближайший полевой отвозит мяч.
"""

from typing import Optional

from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, CarryLimits, DribbleActions
from bridge.strategy import params
from bridge.strategy.context import TickContext, angle_to
from bridge.strategy.guards import go_to, hold_position
from bridge.strategy.plays.kickoff import gk_position
from bridge.strategy.plays.open_play import safe_act

# Для размещения ограничение ведения не действует (мяч не в игре)
PLACEMENT_MAX_DISP = 10000.0


def _placement_limits() -> CarryLimits:
    return CarryLimits(
        min(params.BALL_HOLD_MAX_SPEED_FWD, params.PLACEMENT_MAX_SPEED),
        min(params.BALL_HOLD_MAX_SPEED_SIDE, params.PLACEMENT_MAX_SPEED),
        min(params.BALL_HOLD_MAX_SPEED_BACK, params.PLACEMENT_MAX_SPEED),
        params.BALL_HOLD_MAX_ACC,
        params.BALL_HOLD_MAX_ANG_SPEED,
        params.BALL_HOLD_MAX_ANG_ACC,
    )


def _placer_act(robot: rbt.Robot, ctx: TickContext) -> Action:
    world = ctx.world
    memory = ctx.memory
    ball = world.ball
    target = ctx.field.ball_placement_pos
    if target is None:
        return hold_position(ctx, robot)
    if aux.dist(ball, target) <= params.PLACEMENT_TOLERANCE and not ctx.has_ball(robot):
        if memory.placement_done_time is None:
            memory.placement_done_time = ctx.now
    if memory.placement_done_time is not None:
        # Дриблер 0, пауза, затем отъезд назад, не задевая мяч
        if ctx.now - memory.placement_done_time < params.PLACEMENT_WAIT:
            return Actions.Stop()
        away = (robot.get_pos() - ball).unity()
        point = ball + away * (params.PLACEMENT_RETREAT + params.BALL_HOLD_DIST)
        return Actions.GoToPointIgnore(point, angle_to(point, ball))
    angle = angle_to(ball, target)
    if not ctx.has_ball(robot):
        return Actions.BallGrab(angle, perform_ball_placement=True, dribbler_speed=params.DRIBBLER_HOLD)
    if aux.dist(ball, target) <= params.PLACEMENT_TOLERANCE:
        return Actions.SetDribblerSpeed(0).compose(Actions.Stop())
    return DribbleActions.DribbleTo(
        target,
        angle,
        ball,
        PLACEMENT_MAX_DISP,
        _placement_limits(),
        dribbler_speed=params.DRIBBLER_HOLD,
        hold_dist=params.BALL_HOLD_DIST,
        perform_ball_placement=True,
    )


def play(ctx: TickContext) -> dict[int, Action]:
    """BALL_PLACEMENT"""
    world = ctx.world
    actions: dict[int, Action] = {}
    placer: Optional[rbt.Robot] = None
    if params.AUTO_BALL_PLACEMENT and ctx.we_active and world.field_allies and ctx.field.ball_placement_pos is not None:
        placer = min(world.field_allies, key=lambda r: aux.dist(r.get_pos(), world.ball))
        ctx.memory.set_piece_kicker_id = placer.r_id
    else:
        ctx.memory.placement_done_time = None
    for robot in world.allies:
        if placer is not None and robot.r_id == placer.r_id:
            actions[robot.r_id] = safe_act(ctx, robot, "placement", _placer_act)
        elif ctx.is_gk(robot):
            actions[robot.r_id] = safe_act(ctx, robot, "placement_gk", gk_position)
        else:
            point = robot.get_pos()
            actions[robot.r_id] = go_to(ctx, robot, point, angle_to(point, world.ball))
    return actions
