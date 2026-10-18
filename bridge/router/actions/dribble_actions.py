"""
Действия для слабого дриблера (ТЗ, раздел 4 и 11)

Все численные лимиты передаёт стратегия (R-5.4): роутер не импортирует её модули.
Межтактовое состояние (рампа угла, точка разворота) хранится на rbt.Robot роутера.
"""

import math
from time import time

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions.action import Action, ActionDomain, ActionValues
from bridge.router.actions.base_actions import Actions

# Максимальный шаг времени рампы и ограничителя, с (защита от пауз роутера)
MAX_DT = 0.05
# Шаг первого такта ограничителя: период роутера (CommandSink, 100 Гц)
ROUTER_DT = 0.01


class CarryLimits:
    """Лимиты движения с мячом (только примитивы - объект pickle-ится каждые 20 мс)"""

    def __init__(
        self,
        max_fwd: float,
        max_side: float,
        max_back: float,
        max_acc: float,
        max_ang_speed: float,
        max_ang_acc: float,
    ) -> None:
        self.max_fwd = max_fwd
        self.max_side = max_side
        self.max_back = max_back
        self.max_acc = max_acc
        self.max_ang_speed = max_ang_speed
        self.max_ang_acc = max_ang_acc


def reset_carry_state(robot: rbt.Robot) -> None:
    """Сбросить рампы и точку разворота (мяч потерян)"""
    robot.carry_angle_sp = None
    robot.carry_ang_vel = 0.0
    robot.carry_prev_vel = None
    robot.pivot_point = None


def ramp_angle(robot: rbt.Robot, target: float, max_w: float, max_dw: float, now: float) -> float:
    """
    R-4.1: уставка угла движется к target не быстрее max_w и с ускорением не больше max_dw.
    Роботу отдаётся уставка, а не конечный угол.
    """
    dt = now - robot.carry_ramp_time
    if robot.carry_angle_sp is None or dt > const.CARRY_RAMP_RESET_TIME or dt < 0:
        robot.carry_angle_sp = robot.get_angle()
        robot.carry_ang_vel = 0.0
        robot.carry_ramp_time = now
        return robot.carry_angle_sp
    if dt <= 0:
        # Повторный вызов в тот же момент (HoldBall + BallCarryLimiter) - состояние не меняется
        return robot.carry_angle_sp
    dt = min(dt, MAX_DT)

    sp = robot.carry_angle_sp
    w = robot.carry_ang_vel
    err = aux.wind_down_angle(target - sp)

    w_des = math.copysign(min(max_w, math.sqrt(2 * max_dw * abs(err))), err)
    w += aux.minmax(w_des - w, -max_dw * dt, max_dw * dt)
    w = aux.minmax(w, -max_w, max_w)
    step = w * dt
    if abs(step) >= abs(err) and step * err >= 0:
        sp = target
        w = 0.0
    else:
        sp += step

    lag = aux.wind_down_angle(sp - robot.get_angle())
    if abs(lag) > const.CARRY_SP_MAX_LAG:
        sp = robot.get_angle() + math.copysign(const.CARRY_SP_MAX_LAG, lag)

    sp = aux.wind_down_angle(sp)
    robot.carry_angle_sp = sp
    robot.carry_ang_vel = w
    robot.carry_ramp_time = now
    return sp


def limit_carry_velocity(robot: rbt.Robot, vel: aux.Point, limits: CarryLimits, now: float) -> aux.Point:
    """R-4.3: скорость с мячом ограничена в системе координат робота, ускорение - по модулю"""
    angle = robot.get_angle()
    local = aux.rotate(vel, -angle)  # x - вперёд, y - влево
    fwd = aux.minmax(local.x, -limits.max_back, limits.max_fwd)
    side = aux.minmax(local.y, -limits.max_side, limits.max_side)
    new_vel = aux.rotate(aux.Point(fwd, side), angle)

    dt = now - robot.carry_vel_time
    prev = robot.carry_prev_vel
    if prev is None or dt > const.CARRY_RAMP_RESET_TIME or dt < 0:
        prev = robot.get_vel()
        dt = ROUTER_DT
    dt = min(dt, MAX_DT)

    delta = new_vel - prev
    max_delta = limits.max_acc * dt
    if delta.mag() > max_delta:
        new_vel = prev + delta.unity() * max_delta

    robot.carry_prev_vel = new_vel
    robot.carry_vel_time = now
    return new_vel


def has_ball(domain: ActionDomain, hold_dist: float) -> bool:
    """Мяч у дриблера этого робота"""
    if domain.field.robot_with_ball == domain.robot:
        return True
    robot = domain.robot
    ball = domain.field.ball.get_pos()
    front = robot.get_pos() + aux.rotate(aux.RIGHT, robot.get_angle()) * hold_dist
    return aux.dist(ball, front) < const.BALL_R + const.BALL_GRABBED_DIST - hold_dist


def update_pivot(domain: ActionDomain, hold_dist: float, now: float) -> aux.Point:
    """R-4.2: точка мяча, вокруг которой разворачивается робот"""
    robot = domain.robot
    ball = domain.field.ball.get_pos()
    front = robot.get_pos() + aux.rotate(aux.RIGHT, robot.get_angle()) * hold_dist
    pivot = robot.pivot_point
    stale = now - robot.pivot_time > const.CARRY_RAMP_RESET_TIME
    if pivot is None or stale or aux.dist(ball, pivot) > const.PIVOT_UPDATE_DIST:
        pivot = ball if aux.dist(ball, front) < hold_dist else front
    robot.pivot_point = pivot
    robot.pivot_time = now
    return pivot


class DribbleActions:
    """Действия с мячом для слабого дриблера"""

    class BallCarryLimiter(Action):
        """
        Пост-ограничитель движения с мячом: рампа угла (R-4.1), лимиты скорости и ускорения (R-4.3).
        Ставится последним в цепочке compose.
        """

        def __init__(
            self,
            max_fwd: float,
            max_side: float,
            max_back: float,
            max_acc: float,
            max_ang_speed: float,
            max_ang_acc: float,
        ) -> None:
            self.limits = CarryLimits(max_fwd, max_side, max_back, max_acc, max_ang_speed, max_ang_acc)

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            if current_action.beep:
                return
            now = time()
            robot = domain.robot
            current_action.angle = ramp_angle(
                robot, current_action.angle, self.limits.max_ang_speed, self.limits.max_ang_acc, now
            )
            current_action.vel = limit_carry_velocity(robot, current_action.vel, self.limits, now)

    class HoldBall(Action):
        """Удерживать мяч на месте и разворачиваться к target_angle вокруг мяча (R-4.2)"""

        def __init__(
            self,
            target_angle: float,
            limits: CarryLimits,
            *,
            dribbler_speed: int = 15,
            hold_dist: float = 90,
        ) -> None:
            self.target_angle = target_angle
            self.limits = limits
            self.dribbler_speed = dribbler_speed
            self.hold_dist = hold_dist

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
            now = time()
            robot = domain.robot
            pivot = update_pivot(domain, self.hold_dist, now)
            sp = ramp_angle(robot, self.target_angle, self.limits.max_ang_speed, self.limits.max_ang_acc, now)
            target = pivot - aux.rotate(aux.RIGHT, sp) * self.hold_dist
            lim = self.limits
            return [
                Actions.GoToPointIgnore(target, sp),
                Actions.SetDribblerSpeed(self.dribbler_speed),
                DribbleActions.BallCarryLimiter(
                    lim.max_fwd, lim.max_side, lim.max_back, lim.max_acc, lim.max_ang_speed, lim.max_ang_acc
                ),
            ]

    class DribbleTo(Action):
        """
        Ведение мяча: target_pos - куда привезти МЯЧ.
        Цель проецируется в круг радиусом max_disp - 50 вокруг start_ball_pos (G-6).
        """

        def __init__(
            self,
            target_pos: aux.Point,
            target_angle: float,
            start_ball_pos: aux.Point,
            max_disp: float,
            limits: CarryLimits,
            *,
            dribbler_speed: int = 15,
            hold_dist: float = 90,
            perform_ball_placement: bool = False,
        ) -> None:
            self.target_pos = target_pos
            self.target_angle = target_angle
            self.start_ball_pos = start_ball_pos
            self.max_disp = max_disp
            self.limits = limits
            self.dribbler_speed = dribbler_speed
            self.hold_dist = hold_dist
            self.perform_ball_placement = perform_ball_placement

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
            now = time()
            robot = domain.robot
            radius = max(0.0, self.max_disp - 50)
            ball_target = self.target_pos
            if aux.dist(ball_target, self.start_ball_pos) > radius:
                ball_target = self.start_ball_pos + (ball_target - self.start_ball_pos).unity() * radius
            sp = ramp_angle(robot, self.target_angle, self.limits.max_ang_speed, self.limits.max_ang_acc, now)
            robot_target = ball_target - aux.rotate(aux.RIGHT, sp) * self.hold_dist
            domain.field.router_image.draw_circle(self.start_ball_pos, (255, 200, 0), self.max_disp)
            lim = self.limits
            return [
                Actions.GoToPoint(
                    robot_target,
                    sp,
                    ignore_ball=True,
                    perform_ball_placement=self.perform_ball_placement,
                    ignore_robots={},
                    avoid_ball=False,
                ),
                Actions.SetDribblerSpeed(self.dribbler_speed),
                DribbleActions.BallCarryLimiter(
                    lim.max_fwd, lim.max_side, lim.max_back, lim.max_acc, lim.max_ang_speed, lim.max_ang_acc
                ),
            ]

    class DribbleKick(Action):
        """
        Довернуться с мячом на цель (разворот вокруг мяча, рампа) и ударить, когда
        |ошибка угла| < align_tol и угловая скорость робота < max_kick_ang_vel.
        Напряжение задаётся заранее. Если мяч потерян - не бьёт.
        """

        def __init__(
            self,
            target_pos: aux.Point,
            voltage: int,
            align_tol: float,
            max_kick_ang_vel: float,
            limits: CarryLimits,
            *,
            dribbler_speed: int = 15,
            dribbler_off_on_kick: bool = False,
            hold_dist: float = 90,
            use_kick_forward: bool = False,
        ) -> None:
            self.target_pos = target_pos
            self.voltage = voltage
            self.align_tol = align_tol
            self.max_kick_ang_vel = max_kick_ang_vel
            self.limits = limits
            self.dribbler_speed = dribbler_speed
            self.dribbler_off_on_kick = dribbler_off_on_kick
            self.hold_dist = hold_dist
            self.use_kick_forward = use_kick_forward

        def kick_angle(self, domain: ActionDomain) -> float:
            """Направление удара от мяча на цель"""
            return aux.angle_to_point(domain.field.ball.get_pos(), self.target_pos)

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
            return [
                DribbleActions.HoldBall(
                    self.kick_angle(domain), self.limits, dribbler_speed=self.dribbler_speed, hold_dist=self.hold_dist
                )
            ]

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            robot = domain.robot
            current_action.kicker_voltage = self.voltage
            aligned = (
                abs(aux.wind_down_angle(robot.get_angle() - self.kick_angle(domain))) < self.align_tol
                and abs(robot.get_anglevel()) < self.max_kick_ang_vel
            )
            if aligned and has_ball(domain, self.hold_dist):
                if self.dribbler_off_on_kick:
                    current_action.dribbler_speed = 0
                if self.use_kick_forward:
                    current_action.kick_forward = True
                else:
                    current_action.auto_kick = 1
                domain.field.router_image.draw_line(robot.get_pos(), self.target_pos, (255, 0, 0), 3)

    class TapForward(Action):
        """Откидка мяча малым напряжением; на время удара дриблер 0"""

        def __init__(
            self,
            direction_angle: float,
            voltage: int,
            align_tol: float,
            limits: CarryLimits,
            *,
            hold_dist: float = 90,
        ) -> None:
            self.direction_angle = direction_angle
            self.voltage = voltage
            self.align_tol = align_tol
            self.limits = limits
            self.hold_dist = hold_dist

        def aligned(self, domain: ActionDomain) -> bool:
            """Робот довёрнут на направление откидки"""
            return abs(aux.wind_down_angle(domain.robot.get_angle() - self.direction_angle)) < self.align_tol

        def use_behavior_of(self, domain: ActionDomain, current_action: ActionValues) -> list[Action]:
            if self.aligned(domain):
                # Довёрнут: стоять на месте с выключенным дриблером, чтобы не подобрать мяч обратно
                return [Actions.GoToPointIgnore(domain.robot.get_pos(), self.direction_angle)]
            return [DribbleActions.HoldBall(self.direction_angle, self.limits, hold_dist=self.hold_dist)]

        def behavior(self, domain: ActionDomain, current_action: ActionValues) -> None:
            current_action.kicker_voltage = self.voltage
            if self.aligned(domain):
                current_action.dribbler_speed = 0
                current_action.auto_kick = 1
