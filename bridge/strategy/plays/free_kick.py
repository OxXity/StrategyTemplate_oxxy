"""
FREE_KICK (ТЗ, 9.5)
"""

import math
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions
from bridge.strategy import evaluation
from bridge.strategy import kick_model as km
from bridge.strategy import params
from bridge.strategy.context import TickContext, angle_diff, angle_to
from bridge.strategy.guards import go_to
from bridge.strategy.memory import Decision, Role
from bridge.strategy.plays.kickoff import gk_position, set_piece_kick
from bridge.strategy.plays.open_play import safe_act
from bridge.strategy.roles import common, goalkeeper, support

# Угловой: мяч ближе этого к чужой линии ворот и к боковой линии
CORNER_ZONE_X = 900
CORNER_ZONE_Y = 600
# Исполнитель «готов» к удару, если ближе этого к точке подготовки
KICKER_READY_DIST = 250
# Получатель на месте, если ближе этого к своей точке
RECEIVER_READY_DIST = 300
# Запас до SET_PIECE_MAX_WAIT на ожидание получателя, с
RECEIVER_WAIT_RESERVE = 1.0
# Решение на штрафном меняется, только если новое лучше на столько
DECISION_SWITCH_GAIN = 0.3
# Смена решения на штрафном - только на более сильный тип
DECISION_RANK = {"SHOOT": 3, "PASS": 2, "SAFE": 1, "CLEAR": 0}
# Безопасный ввод: вес продвижения к чужим воротам
SAFE_PROGRESS_W = 0.6


def choose_kicker(ctx: TickContext) -> Optional[rbt.Robot]:
    """Полевой с минимальным временем до точки за мячом (не вратарь, не под запретом)"""
    world = ctx.world
    memory = ctx.memory
    candidates = [r for r in world.field_allies if not ctx.is_banned(r)]
    if not candidates:
        return None
    prev = memory.set_piece_kicker_id
    for robot in candidates:
        if robot.r_id == prev:
            return robot
    return min(candidates, key=lambda r: world.time_to_point(r, world.ball))


def is_corner(ctx: TickContext) -> bool:
    """Мяч у угла чужой половины"""
    ball = ctx.world.ball
    return ctx.world.x_att(ball) > const.FIELD_DX - CORNER_ZONE_X and abs(ball.y) > const.FIELD_DY - CORNER_ZONE_Y


def receiver_target(ctx: TickContext, receiver: rbt.Robot, kicker: Optional[rbt.Robot]) -> aux.Point:
    """Точка Опорного: угловой - у края чужой штрафной; иначе - лучшая точка открывания"""
    world = ctx.world
    if is_corner(ctx):
        goal = ctx.field.enemy_goal
        depth = const.GOAL_PEN_DX + const.DEF_AREA_MARGIN_BALL_OUT + params.CORNER_RECEIVER_GAP
        best = None
        best_p = -1.0
        for y in (-300.0, 0.0, 300.0):
            point = aux.Point(goal.center.x + goal.eye_forw.x * depth, y)
            p = evaluation.eval_shot(world, point, receiver.r_id, extra_time=1.0).p_goal
            if p > best_p:
                best_p, best = p, point
        if best is not None:
            return best
    kicker_pos = kicker.get_pos() if kicker is not None else None
    return evaluation.best_support_point(world, ctx.memory, receiver, kicker_pos, ball_in_play=False)


def _feasible(ctx: TickContext, kicker: rbt.Robot, target: aux.Point) -> bool:
    ball = ctx.world.ball
    angle = angle_to(ball, target)
    return angle_diff(evaluation.feasible_grab_angle(ctx.world, kicker.r_id, ball, angle), angle) < 1e-3


def decide(ctx: TickContext, kicker: rbt.Robot, receiver: Optional[rbt.Robot], recv_point: Optional[aux.Point]) -> Decision:
    """Прямой удар / пас / безопасный ввод"""
    world = ctx.world
    ball = world.ball
    now = ctx.now
    shot = evaluation.eval_shot(world, ball, kicker.r_id)
    if shot.p_goal >= params.FREE_KICK_SHOT_MIN_P and _feasible(ctx, kicker, shot.target):
        return Decision("SHOOT", shot.target, now, shot.p_goal)
    if receiver is not None and recv_point is not None:
        pas = evaluation.eval_pass(world, ball, receiver, recv_point)
        if pas.ok and pas.p_success >= params.FREE_KICK_PASS_MIN_SUCCESS and _feasible(ctx, kicker, recv_point):
            return Decision("PASS", recv_point, now, pas.value, receiver.r_id)
    # Безопасный ввод: зона, куда наш робот успевает первым с запасом
    best: Optional[aux.Point] = None
    best_margin = -math.inf
    for k in range(12):
        angle = 2 * math.pi * k / 12
        for dist in (800.0, 1400.0):
            point = ball + aux.rotate(aux.RIGHT, angle) * dist
            if not world.in_field(point, params.CLEAR_BORDER_MARGIN) or not _feasible(ctx, kicker, point):
                continue
            if evaluation.pass_forbidden(world, ball, point, km.stop_speed(dist)) not in ("", "stops out"):
                continue
            t_ours = min(
                (world.time_to_point(r, point) for r in world.field_allies if r.r_id != kicker.r_id), default=math.inf
            )
            margin = world.enemy_time_to(point) - t_ours
            score = min(margin, 1.0) + SAFE_PROGRESS_W * (world.x_att(point) - world.x_att(ball)) / const.FIELD_DX
            if margin < params.FREE_KICK_SAFE_MARGIN:
                continue
            if score > best_margin:
                best_margin, best = score, point
    if best is not None:
        return Decision("SAFE", best, now, best_margin)
    target, _ = evaluation.safe_clear_target(world, ball)
    if not _feasible(ctx, kicker, target):
        angle = evaluation.feasible_grab_angle(world, kicker.r_id, ball, angle_to(ball, target))
        target = ball + aux.rotate(aux.RIGHT, angle) * aux.dist(ball, target)
    return Decision("CLEAR", target, now, 0.0)


def _ours(ctx: TickContext) -> dict[int, Action]:
    world = ctx.world
    memory = ctx.memory
    ball = world.ball
    actions: dict[int, Action] = {}
    kicker = choose_kicker(ctx)
    memory.set_piece_kicker_id = kicker.r_id if kicker is not None else None
    receiver = next(
        (r for r in world.field_allies if kicker is None or r.r_id != kicker.r_id),
        None,
    )
    recv_point = receiver_target(ctx, receiver, kicker) if receiver is not None else None
    t_state = memory.state_time(ctx.now)

    def kicker_act(robot: rbt.Robot, ctx_: TickContext) -> Action:
        decision = memory.set_piece_decision
        ready = False
        if decision is not None and decision.target is not None:
            back = (decision.target - ball).unity()
            ready = (
                aux.dist(robot.get_pos(), ball - back * (const.GRAB_ALIGN_DIST + params.SET_PIECE_PREP_GAP))
                < KICKER_READY_DIST
            )
        locked = decision is not None and (ready or t_state > params.SET_PIECE_MAX_WAIT - params.SET_PIECE_DECISION_LOCK)
        if not locked:
            new = decide(ctx_, robot, receiver, recv_point)
            # Гистерезис: исполнитель не перебегает на другую сторону мяча из-за мелких колебаний оценки
            if decision is None or DECISION_RANK[new.kind] > DECISION_RANK[decision.kind]:
                decision = new
            elif new.kind == decision.kind and new.utility > decision.utility + DECISION_SWITCH_GAIN:
                decision = new
            elif new.kind == decision.kind and decision.target is not None and new.target is not None:
                if aux.dist(decision.target, new.target) < 300:
                    decision = new
            memory.set_piece_decision = decision
        assert decision is not None and decision.target is not None
        target = decision.target
        if decision.kind == "SHOOT":
            voltage = ctx_.kick_model.shot_voltage()
        elif decision.kind == "PASS":
            voltage, t_arrive = common.pass_kick(ctx_, robot, target)
            if decision.receiver_id is not None:
                common.register_pass(ctx_, robot, decision.receiver_id, target, t_arrive)
            waiting = receiver is not None and aux.dist(receiver.get_pos(), target) > RECEIVER_READY_DIST
            if waiting and t_state < params.SET_PIECE_MAX_WAIT - RECEIVER_WAIT_RESERVE:
                back = (target - ball).unity()
                prep = ball - back * (const.GRAB_ALIGN_DIST + params.SET_PIECE_PREP_GAP)
                return go_to(ctx_, robot, prep, angle_to(ball, target), avoid_ball=False).compose(
                    Actions.SetDribblerSpeed(0)
                )
        else:
            voltage = ctx_.kick_model.kick_voltage(km.stop_speed(aux.dist(ball, target)))
        return set_piece_kick(ctx_, target, voltage)

    def receiver_act(robot: rbt.Robot, ctx_: TickContext) -> Action:
        if common.ball_coming_to(ctx_, robot):
            return common.receive(ctx_, robot, ctx_.field.enemy_goal.center)
        point = recv_point if recv_point is not None else robot.get_pos()
        return go_to(ctx_, robot, point, angle_to(point, ball))

    for robot in world.allies:
        if ctx.is_gk(robot):
            actions[robot.r_id] = safe_act(ctx, robot, "free_kick_gk", gk_position)
        elif kicker is not None and robot.r_id == kicker.r_id:
            actions[robot.r_id] = safe_act(ctx, robot, "free_kick_kicker", kicker_act)
        elif receiver is not None and robot.r_id == receiver.r_id:
            actions[robot.r_id] = safe_act(ctx, robot, "free_kick_receiver", receiver_act)
        else:
            point = support.shot_block_point(ctx)
            actions[robot.r_id] = go_to(ctx, robot, point, angle_to(point, ball))
    return actions


def _theirs(ctx: TickContext) -> dict[int, Action]:
    world = ctx.world
    ball = world.ball
    goal = ctx.field.ally_goal
    actions: dict[int, Action] = {}
    wall = ball + (goal.center - ball).unity() * (params.KEEP_BALL_DIST + 30)
    second_wall = wall + aux.rotate((goal.center - ball).unity(), math.pi / 2) * (2 * const.ROBOT_R + 20)
    direct_danger = aux.dist(ball, goal.center) < const.FIELD_DX
    exclude = [e.r_id for e in world.enemies if aux.dist(e.get_pos(), ball) < params.KEEP_BALL_DIST]
    walls = [wall]
    for robot in world.allies:
        role = ctx.roles.get(robot.r_id, Role.SUPPORT)
        if role == Role.GOALKEEPER:
            continue
        if role == Role.SUPPORT:
            actions[robot.r_id] = go_to(ctx, robot, wall, angle_to(wall, ball))
            continue
        target_enemy = support.most_dangerous_enemy(ctx, exclude)
        if direct_danger or target_enemy is None:
            walls.append(second_wall)
            actions[robot.r_id] = go_to(ctx, robot, second_wall, angle_to(second_wall, ball))
        else:
            exclude.append(target_enemy.r_id)
            point = support.mark_point(ctx, target_enemy)
            actions[robot.r_id] = go_to(ctx, robot, point, angle_to(point, ball))
    for robot in world.allies:
        if ctx.is_gk(robot):
            actions[robot.r_id] = safe_act(ctx, robot, "free_kick_gk", lambda r, c: _gk_behind_wall(r, c, walls))
    return actions


def _gk_behind_wall(robot: rbt.Robot, ctx: TickContext, walls: list[aux.Point]) -> Action:
    """Вратарь закрывает часть створа, которую не закрывает «стенка»"""
    world = ctx.world
    ball = world.ball
    goal = ctx.field.ally_goal
    ref = angle_to(ball, goal.center)
    lo = aux.wind_down_angle(angle_to(ball, goal.up) - ref)
    hi = aux.wind_down_angle(angle_to(ball, goal.down) - ref)
    lo, hi = min(lo, hi), max(lo, hi)
    blocks = []
    for wall in walls:
        d = max(aux.dist(ball, wall), 1.0)
        half = math.asin(min(1.0, (const.ROBOT_R + const.BALL_R) / d))
        center = aux.wind_down_angle(angle_to(ball, wall) - ref)
        blocks.append((center - half, center + half))
    free = evaluation.subtract_intervals(lo, hi, blocks)
    if not free:
        return gk_position(robot, ctx)
    f_lo, f_hi = max(free, key=lambda iv: iv[1] - iv[0])
    direction = aux.rotate(aux.RIGHT, ref + (f_lo + f_hi) / 2)
    y_lim = goalkeeper.max_line_y()
    a = goalkeeper.line_point(ctx, params.GK_LINE_OFFSET, -const.FIELD_DY)
    b = goalkeeper.line_point(ctx, params.GK_LINE_OFFSET, const.FIELD_DY)
    cross = aux.get_line_intersection(ball, ball + direction, a, b, "RL")
    if cross is None:
        return gk_position(robot, ctx)
    point = aux.Point(cross.x, aux.minmax(cross.y, -y_lim, y_lim))
    return go_to(ctx, robot, point, angle_to(point, ball))


def play(ctx: TickContext) -> dict[int, Action]:
    """FREE_KICK"""
    return _ours(ctx) if ctx.we_active else _theirs(ctx)
