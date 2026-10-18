"""
Вратарь (ТЗ, 8.1)
"""

import math
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions
from bridge.strategy import evaluation, params
from bridge.strategy.context import TickContext, angle_to, unit
from bridge.strategy.guards import go_to, legal
from bridge.strategy.memory import Decision, Role
from bridge.strategy.roles import common


def line_point(ctx: TickContext, depth: float, y: float) -> aux.Point:
    """Точка на глубине depth от линии наших ворот"""
    goal = ctx.field.ally_goal
    return aux.Point(goal.center.x + goal.eye_forw.x * depth, y)


def depth_of(ctx: TickContext, point: aux.Point) -> float:
    """Глубина точки от линии наших ворот"""
    goal = ctx.field.ally_goal
    return (point.x - goal.center.x) * goal.eye_forw.x


def max_line_y() -> float:
    """|y| на линии вратаря"""
    return const.GOAL_DY / 2 - params.GK_Y_MARGIN


def _intersect_depth(ctx: TickContext, origin: aux.Point, direction: aux.Point, depth: float) -> Optional[aux.Point]:
    """Пересечение луча с линией на глубине depth"""
    a = line_point(ctx, depth, -const.FIELD_DY)
    b = line_point(ctx, depth, const.FIELD_DY)
    return aux.get_line_intersection(origin, origin + direction, a, b, "RL")


def bisector(ctx: TickContext, ball: aux.Point) -> tuple[aux.Point, float]:
    """Биссектриса угла «мяч -> две штанги» и сам угол"""
    goal = ctx.field.ally_goal
    u1 = (goal.up - ball).unity()
    u2 = (goal.down - ball).unity()
    direction = (u1 + u2).unity()
    if direction.mag() < 0.5:
        direction = (goal.center - ball).unity()
    alpha = abs(aux.wind_down_angle(u1.arg() - u2.arg()))
    return direction, alpha


def position_target(ctx: TickContext, ball: Optional[aux.Point] = None, allow_come_out: bool = True) -> aux.Point:
    """Цель позиционирования вратаря (8.1, п. 3-4)"""
    world = ctx.world
    goal = ctx.field.ally_goal
    if ball is None:
        ball = world.ball
    y_lim = max_line_y()
    direction, alpha = bisector(ctx, ball)

    point = _intersect_depth(ctx, ball, direction, params.GK_LINE_OFFSET)
    if point is None:
        point = line_point(ctx, params.GK_LINE_OFFSET, aux.minmax(ball.y, -y_lim, y_lim))
    ball_depth = depth_of(ctx, ball)
    if ball_depth < params.GK_CORNER_ZONE and abs(ball.y - goal.center.y) > const.GOAL_DY / 2:
        point = line_point(ctx, params.GK_LINE_OFFSET, math.copysign(y_lim, ball.y - goal.center.y))
    point = aux.Point(point.x, aux.minmax(point.y, -y_lim, y_lim))

    owner = world.owner if world.owner_is_theirs else None
    # Выход навстречу владеющему сопернику: корпус закрывает весь угол
    if allow_come_out and owner is not None and aux.dist(ball, goal.center) < params.GK_COME_OUT_DIST and alpha > 1e-3:
        cover = const.ROBOT_R / math.sin(alpha / 2)
        out = ball + direction * cover
        out_depth = depth_of(ctx, out)
        if out_depth > params.GK_MAX_OUT:
            capped = _intersect_depth(ctx, ball, direction, params.GK_MAX_OUT)
            out = capped if capped is not None else out
            out_depth = params.GK_MAX_OUT
        if out_depth > params.GK_LINE_OFFSET and abs(out.y) < const.GOAL_PEN_DY / 2:
            point = out

    # Предугадывание удара по корпусу соперника (P1)
    if owner is not None:
        heading = unit(owner.get_angle())
        cross = aux.get_line_intersection(ball, ball + heading, goal.down, goal.up, "RL")
        if cross is not None and abs(cross.y - goal.center.y) <= const.GOAL_DY / 2 + const.ROBOT_R:
            pred = _intersect_depth(ctx, ball, heading, depth_of(ctx, point))
            if pred is not None:
                w = params.GK_HEADING_PREDICT_WEIGHT
                y = aux.minmax(w * pred.y + (1 - w) * point.y, -const.GOAL_PEN_DY / 2, const.GOAL_PEN_DY / 2)
                if depth_of(ctx, point) <= params.GK_LINE_OFFSET + 1:
                    y = aux.minmax(y, -y_lim, y_lim)
                point = aux.Point(point.x, y)
    return point


def blocked_interval(ctx: TickContext, ball: aux.Point, gk_pos: aux.Point) -> tuple[aux.Point, aux.Point]:
    """Точки на линии ворот, между которыми удар закрыт вратарём (для отрисовки и «стенки»)"""
    goal = ctx.field.ally_goal
    d = max(aux.dist(ball, gk_pos), 1.0)
    half = math.asin(min(1.0, (const.ROBOT_R + const.BALL_R) / d))
    center = angle_to(ball, gk_pos)
    pts = []
    for ang in (center - half, center + half):
        cross = aux.get_line_intersection(ball, ball + unit(ang), goal.down, goal.up, "RL")
        pts.append(cross if cross is not None else gk_pos)
    return pts[0], pts[1]


class Goalkeeper:
    """Вратарь: по первому выполнимому пункту 8.1"""

    def act(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Действие вратаря в открытой игре"""
        world = ctx.world
        if world.ball_lost_vision:
            return self.positioning(robot, ctx)
        if ctx.field.is_ball_moves_to_goal() and world.ball_speed > params.GK_SHOT_SPEED and world.owner is None:
            return self.block_shot(robot, ctx)
        in_area = aux.is_point_inside_poly(world.ball, ctx.field.ally_goal.hull)
        if ctx.has_ball(robot) or (in_area and world.ball_speed < params.GK_BALL_SLOW_SPEED):
            return self.clear_ball(robot, ctx)
        return self.positioning(robot, ctx)

    def positioning(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Позиционирование, лицом к мячу"""
        target = position_target(ctx)
        ball = ctx.world.ball
        action = go_to(ctx, robot, target, angle_to(target, ball))
        if aux.dist(robot.get_pos(), ball) < params.DRIBBLER_ON_DIST:
            return action.compose(Actions.SetDribblerSpeed(params.DRIBBLER_RECEIVE))
        return action

    def block_shot(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Удар по нашим воротам: встать на траекторию"""
        world = ctx.world
        ball = world.ball
        line_target = _intersect_depth(ctx, ball, world.ball_dir, params.GK_LINE_OFFSET)
        target = line_target if line_target is not None else position_target(ctx)
        t_ball_line = evaluation.ball_time(aux.dist(ball, target), world.ball_speed)
        if world.time_to_point(robot, target, reach=const.BALL_R) > t_ball_line:
            # Пересечение недостижимо: ближайшая к нему достижимая точка траектории в штрафной
            best: Optional[aux.Point] = None
            best_dist = math.inf
            for i in range(len(world.traj_t)):
                point = aux.Point(float(world.traj_x[i]), float(world.traj_y[i]))
                depth = depth_of(ctx, point)
                if depth < 0 or depth > const.GOAL_PEN_DX or abs(point.y) > const.GOAL_PEN_DY / 2:
                    continue
                reachable = world.time_to_point(robot, point, reach=const.BALL_R) <= float(world.traj_t[i])
                if reachable and aux.dist(point, target) < best_dist:
                    best, best_dist = point, aux.dist(point, target)
            if best is not None:
                target = best
        target = legal(ctx, robot, target)
        ctx.targets[robot.r_id] = target
        return Actions.GoToPointIgnore(target, angle_to(target, ball), ball_catching=True).compose(
            Actions.SetDribblerSpeed(params.DRIBBLER_RECEIVE)
        )

    def _choose(self, robot: rbt.Robot, ctx: TickContext) -> Decision:
        """Пас Опорному или безопасный вынос (с таймером G-8)"""
        world = ctx.world
        memory = ctx.memory
        now = ctx.now
        ball = world.ball
        t_in = now - memory.ball_in_our_area_since if memory.ball_in_our_area_since is not None else 0.0
        min_p = params.GK_PASS_MIN_SUCCESS if t_in < params.GK_CLEAR_SOFT_TIME else params.PASS_MIN_SUCCESS

        best: Optional[Decision] = None
        for mate in world.field_allies:
            if ctx.roles.get(mate.r_id) not in (Role.SUPPORT, Role.ATTACKER, Role.DEFENDER):
                continue
            pas = evaluation.eval_pass(world, ball, mate, mate.get_pos())
            if pas.ok and pas.p_success >= min_p and (best is None or pas.value > best.utility):
                best = Decision("PASS", mate.get_pos(), now, pas.value, mate.r_id)
        if best is not None:
            return best
        target, _ = evaluation.safe_clear_target(world, ball)
        return Decision("CLEAR", target, now, 0.0)

    def clear_ball(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Подобрать мяч с предориентацией наружу и избавиться от него"""
        world = ctx.world
        memory = ctx.memory
        now = ctx.now
        ball = world.ball
        t_in = now - memory.ball_in_our_area_since if memory.ball_in_our_area_since is not None else 0.0

        decision = memory.gk_decision
        if decision is None or now - decision.since > params.DECISION_COMMIT_TIME or t_in > params.GK_CLEAR_SOFT_TIME:
            decision = self._choose(robot, ctx)
            memory.gk_decision = decision
        target = decision.target if decision.target is not None else ctx.field.enemy_goal.center

        angle = angle_to(ball, target)
        if decision.kind == "PASS":
            voltage, t_arrive = common.pass_kick(ctx, robot, target)
            tol = params.KICK_ALIGN_TOL_PASS
            if decision.receiver_id is not None:
                common.register_pass(ctx, robot, decision.receiver_id, target, t_arrive)
        else:
            speed = evaluation.km.stop_speed(aux.dist(ball, target))
            voltage = ctx.kick_model.kick_voltage(speed, common.robot_vel_along(robot, angle))
            tol = params.KICK_ALIGN_TOL_PASS
        if t_in > params.GK_CLEAR_HARD_TIME:
            tol = params.PRESSURE_MAX_TURN  # 10 с в штрафной близко - бить в любой допустимый

        ctx.draw.line(ball, target, (0, 200, 255) if decision.kind == "PASS" else (255, 200, 0), 2)
        if ctx.has_ball(robot):
            return common.dribble_kick(ctx, robot, target, voltage, tol)
        direct = common.kick_on_contact(ctx, robot, target, voltage, tol)
        if direct is not None:
            return direct
        return common.grab(ctx, robot, angle)
