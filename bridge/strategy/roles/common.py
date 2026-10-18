"""
Общие для ролей действия с мячом
"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, DribbleActions, StrategyActions
from bridge.router.actions.dumb_actions import DumbActions
from bridge.strategy import evaluation, params
from bridge.strategy.context import (
    TickContext,
    angle_diff,
    angle_to,
    carry_limits,
    unit,
)
from bridge.strategy.evaluation import feasible_grab_angle
from bridge.strategy.memory import PassInfo

# Пас: фактическая скорость мяча не больше расчётной в это число раз (иначе укатится за поле)
PASS_SPEED_CAP_K = 1.15


def robot_vel_along(robot: rbt.Robot, angle: float) -> float:
    """Скорость робота вдоль направления удара (G-7)"""
    return aux.scal_mult(robot.get_vel(), unit(angle))


def pass_kick(ctx: TickContext, robot: rbt.Robot, target: aux.Point) -> tuple[int, float]:
    """Напряжение паса из точки мяча в target (остановка в поле, потолок скорости) и время прихода мяча"""
    ball = ctx.world.ball
    speed, _ = evaluation.pass_speed_in_field(ball, target)
    vel_along = robot_vel_along(robot, angle_to(ball, target))
    voltage = ctx.kick_model.pass_voltage_for_speed(speed, vel_along, cap=speed * PASS_SPEED_CAP_K)
    real_speed = ctx.kick_model.speed_for_voltage(voltage) + max(0.0, vel_along)
    return voltage, evaluation.ball_time(aux.dist(ball, target), real_speed)


def grab(ctx: TickContext, robot: rbt.Robot, desired_angle: float) -> Action:
    """Подбор мяча с предориентацией (R-4.4): угол выбирается допустимым (7.5)"""
    world = ctx.world
    rel = world.ball - robot.get_pos()
    away_speed = aux.scal_mult(world.ball_vel, rel.unity())
    if away_speed > params.GRAB_FOLLOW_SPEED and rel.mag() < params.GRAB_FOLLOW_DIST:
        # Мяч только что откинут: следовать без дриблера, не подбирая его обратно
        point = world.ball - rel.unity() * params.GRAB_FOLLOW_DIST
        return Actions.GoToPoint(point, angle_to(point, world.ball), ignore_robots={})
    angle = feasible_grab_angle(ctx.world, robot.r_id, ctx.world.ball, desired_angle)
    ctx.targets[robot.r_id] = ctx.world.ball
    return Actions.BallGrab(angle, dribbler_speed=params.DRIBBLER_HOLD)


def kick_on_contact(
    ctx: TickContext, robot: rbt.Robot, target: aux.Point, voltage: int, align_tol: float
) -> Optional[Action]:
    """
    Подъезд сзади мяча прямо по линии удара и автоудар при касании - мяч не удерживается
    и не разворачивается (главный обход слабого дриблера). None, если подъезд по этой линии недопустим.
    """
    ball = ctx.world.ball
    angle = angle_to(ball, target)
    feasible = feasible_grab_angle(ctx.world, robot.r_id, ball, angle)
    if angle_diff(feasible, angle) > 1e-3:
        return None
    ctx.memory.kick_voltage_issued[robot.r_id] = voltage
    ctx.targets[robot.r_id] = ball
    return (
        Actions.BallGrab(angle, dribbler_speed=params.DRIBBLER_HOLD)
        .compose(DumbActions.ShootAction(target, angle_bounds=align_tol))
        .compose(DumbActions.ControlVoltageAction(voltage))
    )


def dribble_kick(ctx: TickContext, robot: rbt.Robot, target: aux.Point, voltage: int, align_tol: float) -> Action:
    """Удар с мячом в дриблере: разворот вокруг мяча с рампой и удар при довороте"""
    ctx.memory.kick_voltage_issued[robot.r_id] = voltage
    return DribbleActions.DribbleKick(
        target,
        voltage,
        align_tol,
        params.KICK_MAX_ANG_VEL,
        carry_limits(),
        dribbler_speed=params.DRIBBLER_HOLD,
        dribbler_off_on_kick=params.DRIBBLER_OFF_BEFORE_KICK,
        hold_dist=params.BALL_HOLD_DIST,
        use_kick_forward=params.USE_KICK_FORWARD,
    )


def receive(ctx: TickContext, robot: rbt.Robot, face_target: Optional[aux.Point], one_touch_voltage: int = 0) -> Action:
    """Приём паса (R-4.7)"""
    ctx.targets[robot.r_id] = ctx.world.ball
    if one_touch_voltage > 0:
        ctx.memory.kick_voltage_issued[robot.r_id] = one_touch_voltage
    return StrategyActions.Receive(
        face_target,
        max_bias=params.RECEIVE_MAX_BIAS,
        dribbler_speed=params.DRIBBLER_RECEIVE,
        cushion_speed=params.RECEIVE_CUSHION_SPEED,
        cushion_dist=params.RECEIVE_CUSHION_DIST,
        cushion_shift=params.RECEIVE_CUSHION_SHIFT,
        hold_dist=params.BALL_HOLD_DIST,
        one_touch_voltage=one_touch_voltage,
    )


def ball_coming_to(ctx: TickContext, robot: rbt.Robot) -> bool:
    """Мяч катится на робота: робот впереди по ходу мяча и близко к его линии"""
    world = ctx.world
    if not world.ball_moving or world.ball_speed < params.RECEIVE_MIN_BALL_SPEED or world.owner is not None:
        return False
    rel = robot.get_pos() - world.ball
    if aux.scal_mult(rel, world.ball_dir) <= 0:
        return False
    line_point = aux.closest_point_on_line(world.ball, world.ball + world.ball_dir, robot.get_pos(), "R")
    return aux.dist(line_point, robot.get_pos()) < params.RECEIVE_LINE_DIST


def one_touch_voltage(ctx: TickContext, robot: rbt.Robot, target: aux.Point) -> int:
    """P2: напряжение удара в одно касание или 0, если нельзя"""
    if not params.ONE_TOUCH_ENABLED or not ctx.world.ball_moving:
        return 0
    incoming = ctx.world.ball_dir * -1
    out_dir = (target - robot.get_pos()).unity()
    if angle_diff(incoming.arg(), out_dir.arg()) > params.ONE_TOUCH_MAX_ANGLE:
        return 0
    # Итоговая скорость мяча ≤ BALL_SPEED_SAFE: вычитаем приходящую скорость мяча с запасом
    arrival = ctx.world.ball_speed
    return ctx.kick_model.kick_voltage(params.BALL_SPEED_SAFE - arrival)


def register_pass(ctx: TickContext, passer: rbt.Robot, receiver_id: int, target: aux.Point, t_arrive: float) -> None:
    """Запомнить пас до удара: после удара он станет «пасом в полёте»"""
    ctx.memory.pending_pass = PassInfo(passer.r_id, receiver_id, target, ctx.now, ctx.now + t_arrive)


def wait_point_outside_area(ctx: TickContext, ball: aux.Point, ally: bool, gap: float) -> aux.Point:
    """Точка на линии «мяч → центр поля» вне штрафной (+gap)"""
    goal = ctx.field.ally_goal if ally else ctx.field.enemy_goal
    direction = (aux.Point(0, 0) - ball).unity()
    if direction.mag() < 0.5:
        direction = goal.eye_forw
    point = ball
    step = const.ROBOT_R / 2
    for _ in range(60):
        if not ctx.world.in_area(point, ally, const.ROBOT_R * 1.2 + gap):
            break
        point = point + direction * step
    return point
