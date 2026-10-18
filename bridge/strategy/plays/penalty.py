"""
PREPARE_PENALTY / PENALTY (ТЗ, 9.6)

Точка пенальти на нашей половине (Регламент 2.1.3): «буллит» с ведением, лимит 10 с.
Все, кроме бьющего и вратаря, не ближе PENALTY_OTHERS_DIST к мячу и за линией мяча.
"""

import math
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, DribbleActions
from bridge.strategy import evaluation, params
from bridge.strategy.context import (
    TickContext,
    angle_to,
    carry_limits,
    clamp_angle_towards,
    unit,
)
from bridge.strategy.guards import go_to, legal
from bridge.strategy.memory import Role
from bridge.strategy.plays.kickoff import gk_position
from bridge.strategy.plays.open_play import GOALKEEPER, safe_act
from bridge.strategy.roles import common, goalkeeper

# Разворот с мячом в пенальти не больше ±25° от направления на ворота
PENALTY_MAX_TURN = math.radians(25)
# Направление откидки сбрасывается, когда мяч дальше этого от бьющего, мм
TAP_RESET_DIST = 250
# Позиции остальных: углы от направления «назад» относительно атаки
OTHERS_ANGLES = (math.radians(50), math.radians(-50), math.radians(80), math.radians(-80))


def penalty_kicker(ctx: TickContext) -> Optional[rbt.Robot]:
    """Бьющий - текущий Нападающий; выбранный бьющий не меняется до конца розыгрыша"""
    prev = ctx.memory.set_piece_kicker_id
    if prev is not None and any(r.r_id == prev for r in ctx.world.field_allies):
        return ctx.field.allies[prev]
    attacker = ctx.robot_with_role(Role.ATTACKER)
    if attacker is not None:
        return attacker
    return ctx.world.field_allies[0] if ctx.world.field_allies else None


def others_point(ctx: TickContext, index: int) -> aux.Point:
    """Позиция остальных: не ближе PENALTY_OTHERS_DIST к мячу, за линией мяча"""
    world = ctx.world
    ball = world.ball
    penalty_dir = world.attack_sign if ctx.we_active else -world.attack_sign
    back = aux.Point(-penalty_dir, 0)
    angle = OTHERS_ANGLES[index % len(OTHERS_ANGLES)]
    return ball + aux.rotate(back, angle) * (params.PENALTY_OTHERS_DIST + const.ROBOT_R)


def _others(ctx: TickContext, exclude: list[int]) -> dict[int, Action]:
    actions: dict[int, Action] = {}
    index = 0
    for robot in ctx.world.field_allies:
        if robot.r_id in exclude:
            continue
        point = others_point(ctx, index)
        index += 1
        actions[robot.r_id] = go_to(ctx, robot, point, angle_to(point, ctx.world.ball))
    return actions


# --- Наш пенальти ---


def _kicker_prepare(robot: rbt.Robot, ctx: TickContext) -> Action:
    ball = ctx.world.ball
    goal = ctx.field.enemy_goal.center
    point = ball + (ball - goal).unity() * params.PENALTY_KICKER_DIST
    return go_to(ctx, robot, point, angle_to(point, goal))


def _kicker_play(robot: rbt.Robot, ctx: TickContext) -> Action:
    world = ctx.world
    memory = ctx.memory
    ball = world.ball
    goal = ctx.field.enemy_goal
    t = memory.state_time(ctx.now)
    deadline = t >= params.PENALTY_SHOOT_DEADLINE
    to_goal = angle_to(ball, goal.center)

    shot = evaluation.eval_shot(world, ball, robot.r_id)
    if aux.dist(robot.get_pos(), ball) > TAP_RESET_DIST:
        memory.penalty_tap_angle = None  # мяч откинут - следующая откидка выбирается заново
    if not ctx.has_ball(robot):
        if deadline or shot.p_goal >= params.PENALTY_SHOT_MIN_P:
            voltage = ctx.kick_model.shot_voltage(common.robot_vel_along(robot, angle_to(ball, shot.target)))
            direct = common.kick_on_contact(ctx, robot, shot.target, voltage, params.KICK_ALIGN_TOL_SHOT)
            if direct is not None:
                return direct
        return common.grab(ctx, robot, to_goal)

    if deadline or shot.p_goal >= params.PENALTY_SHOT_MIN_P:
        voltage = ctx.kick_model.shot_voltage(common.robot_vel_along(robot, angle_to(ball, shot.target)))
        tol = params.KICK_ALIGN_TOL_SHOT if not deadline else params.KICK_ALIGN_TOL_PASS
        return common.dribble_kick(ctx, robot, shot.target, voltage, tol)

    disp = memory.dribble_disp(robot.r_id, ball)
    track = memory.dribble.get(robot.r_id)
    if disp >= params.PENALTY_TAP_DISP or track is None:
        # Откидка вперёд в сторону, куда вратарь не успевает; направление фиксируется до удара
        tap_angle = memory.penalty_tap_angle
        if tap_angle is None:
            gk = world.enemy_gk
            side = 0.0
            if gk is not None:
                side = -1.0 if aux.vec_mult(unit(to_goal), gk.get_pos() - ball) > 0 else 1.0
            tap_angle = clamp_angle_towards(robot.get_angle(), to_goal + side * PENALTY_MAX_TURN / 2, PENALTY_MAX_TURN)
            memory.penalty_tap_angle = tap_angle
        voltage = ctx.kick_model.tap_voltage((params.TAP_DIST_MIN + params.TAP_DIST_MAX) / 2)
        ctx.memory.kick_voltage_issued[robot.r_id] = voltage
        return DribbleActions.TapForward(
            tap_angle, voltage, params.TAP_ALIGN_TOL, carry_limits(), hold_dist=params.BALL_HOLD_DIST
        )

    # Ведение только с неотрицательной составляющей к воротам
    face = clamp_angle_towards(to_goal, robot.get_angle(), PENALTY_MAX_TURN)
    target = ball + unit(face) * (params.DRIBBLE_SOFT_LIMIT - disp)
    stop_depth = const.GOAL_PEN_DX + const.ROBOT_R * 2
    goal_depth = (target.x - goal.center.x) * goal.eye_forw.x
    if goal_depth < stop_depth:
        target = target + goal.eye_forw * (stop_depth - goal_depth)
    return DribbleActions.DribbleTo(
        target,
        face,
        track.start_ball_pos,
        params.DRIBBLE_SOFT_LIMIT,
        carry_limits(),
        dribbler_speed=params.DRIBBLER_HOLD,
        hold_dist=params.BALL_HOLD_DIST,
    )


# --- Пенальти соперника ---


def _gk_on_line(robot: rbt.Robot, ctx: TickContext) -> Action:
    """До ввода мяча: на линии ворот, касаясь её, напротив мяча"""
    ball = ctx.world.ball
    y_lim = const.GOAL_DY / 2 - const.ROBOT_R
    point = goalkeeper.line_point(ctx, params.GK_PENALTY_LINE_OFFSET, aux.minmax(ball.y, -y_lim, y_lim))
    point = legal(ctx, robot, point)
    ctx.targets[robot.r_id] = point
    return Actions.GoToPointIgnore(point, angle_to(point, ball))


def _gk_defend(robot: rbt.Robot, ctx: TickContext) -> Action:
    """После ввода: забрать откинутый мяч или сокращать угол по биссектрисе с угадыванием по корпусу"""
    world = ctx.world
    ball = world.ball
    if ctx.field.is_ball_moves_to_goal() and world.ball_speed > params.GK_SHOT_SPEED and world.owner is None:
        return GOALKEEPER.block_shot(robot, ctx)
    if world.owner is None:
        t_gk = world.time_to_point(robot, ball)
        if t_gk < world.enemy_time_to(ball) - params.GK_PENALTY_TAKE_MARGIN:
            return common.grab(ctx, robot, angle_to(ball, ctx.field.enemy_goal.center))
    goal = ctx.field.ally_goal
    direction, _ = goalkeeper.bisector(ctx, ball)
    depth = aux.minmax(0.35 * aux.dist(ball, goal.center), params.GK_PENALTY_OUT_MIN, params.GK_PENALTY_OUT_MAX)
    depth = min(depth, max(goalkeeper.depth_of(ctx, ball) - const.ROBOT_R * 3, params.GK_LINE_OFFSET))
    a = goalkeeper.line_point(ctx, depth, -const.FIELD_DY)
    b = goalkeeper.line_point(ctx, depth, const.FIELD_DY)
    point = aux.get_line_intersection(ball, ball + direction, a, b, "RL")
    if point is None:
        return gk_position(robot, ctx)
    owner = world.owner if world.owner_is_theirs else None
    if owner is not None:
        heading = unit(owner.get_angle())
        pred = aux.get_line_intersection(ball, ball + heading, a, b, "RL")
        if pred is not None and abs(pred.y) < const.GOAL_DY:
            w = params.GK_HEADING_PREDICT_WEIGHT
            point = aux.Point(point.x, w * pred.y + (1 - w) * point.y)
    return go_to(ctx, robot, point, angle_to(point, ball))


def prepare(ctx: TickContext) -> dict[int, Action]:
    """PREPARE_PENALTY"""
    actions: dict[int, Action] = {}
    exclude: list[int] = []
    if ctx.we_active:
        kicker = penalty_kicker(ctx)
        ctx.memory.set_piece_kicker_id = kicker.r_id if kicker is not None else None
        if kicker is not None:
            exclude.append(kicker.r_id)
            actions[kicker.r_id] = safe_act(ctx, kicker, "penalty_kicker", _kicker_prepare)
    actions.update(_others(ctx, exclude))
    gk = ctx.world.gk
    if gk is not None:
        behavior = gk_position if ctx.we_active else _gk_on_line
        actions[gk.r_id] = safe_act(ctx, gk, "penalty_gk", behavior)
    return actions


def penalty(ctx: TickContext) -> dict[int, Action]:
    """PENALTY после Normal Start"""
    actions: dict[int, Action] = {}
    exclude: list[int] = []
    if ctx.we_active:
        kicker = penalty_kicker(ctx)
        ctx.memory.set_piece_kicker_id = kicker.r_id if kicker is not None else None
        if kicker is not None:
            exclude.append(kicker.r_id)
            actions[kicker.r_id] = safe_act(ctx, kicker, "penalty_kicker", _kicker_play)
    actions.update(_others(ctx, exclude))
    gk = ctx.world.gk
    if gk is not None:
        if ctx.we_active:
            behavior = gk_position
        else:
            behavior = _gk_defend if ctx.memory.penalty_ball_in_play else _gk_on_line
        actions[gk.r_id] = safe_act(ctx, gk, "penalty_gk", behavior)
    return actions
