"""
Опорный, он же универсал (ТЗ, 8.3)
"""

import math
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router import legality
from bridge.router.actions import Action
from bridge.strategy import evaluation, params
from bridge.strategy.context import TickContext, angle_to, clamp_angle_towards, unit
from bridge.strategy.guards import go_to
from bridge.strategy.memory import Possession, Role
from bridge.strategy.roles import common, goalkeeper


def enemy_danger(ctx: TickContext, enemy: rbt.Robot) -> float:
    """Опасность соперника без мяча: близость к нашим воротам, свобода, безопасность паса ему"""
    world = ctx.world
    pos = enemy.get_pos()
    d_goal = aux.dist(pos, ctx.field.ally_goal.center)
    closeness = 1.0 - evaluation.clip01(d_goal / (2 * const.FIELD_DX))
    free = 1000.0
    for ally in world.allies:
        free = min(free, aux.dist(ally.get_pos(), pos))
    blocked = 1000.0
    for ally in world.allies:
        on_line = aux.closest_point_on_line(world.ball, pos, ally.get_pos(), "S")
        blocked = min(blocked, aux.dist(on_line, ally.get_pos()))
    return closeness + 0.5 * free / 1000 + 0.3 * evaluation.clip01(blocked / 600)


def most_dangerous_enemy(ctx: TickContext, exclude: list[int]) -> Optional[rbt.Robot]:
    """Самый опасный соперник кроме владеющего и вратаря"""
    world = ctx.world
    best: Optional[rbt.Robot] = None
    best_score = -math.inf
    for enemy in world.enemies:
        if enemy.r_id in exclude or (world.enemy_gk is not None and enemy.r_id == world.enemy_gk.r_id):
            continue
        score = enemy_danger(ctx, enemy)
        if score > best_score:
            best_score = score
            best = enemy
    return best


def mark_point(ctx: TickContext, enemy: rbt.Robot) -> aux.Point:
    """Опека: на отрезке «опекаемый → мяч» в MARK_FRACTION от опекаемого, со смещением к нашим воротам"""
    pos = enemy.get_pos()
    point = aux.lerp(pos, ctx.world.ball, params.MARK_FRACTION)
    shift = (ctx.field.ally_goal.center - point).unity() * params.MARK_GOAL_SHIFT
    return point + shift


def shot_block_point(ctx: TickContext, ball: Optional[aux.Point] = None, gap: float = params.DEF_AREA_GAP) -> aux.Point:
    """
    Перекрытие удара: с учётом целевой позиции вратаря найти не закрытый им интервал створа
    и встать на луч из мяча через его середину в DEF_DIST от мяча (8.3)
    """
    world = ctx.world
    goal = ctx.field.ally_goal
    if ball is None:
        ball = world.ball
    ref = angle_to(ball, goal.center)
    lo = aux.wind_down_angle(angle_to(ball, goal.up) - ref)
    hi = aux.wind_down_angle(angle_to(ball, goal.down) - ref)
    lo, hi = min(lo, hi), max(lo, hi)
    blocks: list[tuple[float, float]] = []
    if world.gk is not None:
        gk_pos = goalkeeper.position_target(ctx, ball)
        d = max(aux.dist(ball, gk_pos), 1.0)
        half = math.asin(min(1.0, (const.ROBOT_R + const.BALL_R) / d))
        center = aux.wind_down_angle(angle_to(ball, gk_pos) - ref)
        blocks.append((center - half, center + half))
    free = evaluation.subtract_intervals(lo, hi, blocks)
    aim = 0.0
    if free:
        f_lo, f_hi = max(free, key=lambda iv: iv[1] - iv[0])
        aim = (f_lo + f_hi) / 2
    d_goal = aux.dist(ball, goal.center)
    def_dist = aux.minmax(params.DEF_DIST_K * d_goal, params.DEF_DIST_MIN, params.DEF_DIST_MAX)
    point = ball + unit(ref + aim) * def_dist
    return legality.push_out_of_area(point, goal, gap)


class Support:
    """Опорный"""

    def act(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Действие Опорного в открытой игре"""
        world = ctx.world
        memory = ctx.memory
        ball = world.ball
        attacker = ctx.robot_with_role(Role.ATTACKER)
        att_pos = attacker.get_pos() if attacker is not None else None

        if world.ball_lost_vision:
            return go_to(ctx, robot, robot.get_pos(), robot.get_angle())

        flight = memory.pass_in_flight
        if flight is not None and flight.to_id == robot.r_id:
            return common.receive(ctx, robot, self._shot_face(ctx, robot.get_pos()))

        steal = self._steal_pass(robot, ctx)
        if steal is not None:
            return steal

        mode = self._mode(ctx)
        if mode == "OPEN":
            return self.open_up(robot, ctx, att_pos)
        if mode == "DEFEND":
            return self.defend(robot, ctx, attacker)
        base = att_pos if att_pos is not None else ball
        point = base + (ctx.field.ally_goal.center - base).unity() * params.SUPPORT_COVER_BEHIND
        point = legality.push_out_of_area(point, ctx.field.ally_goal, params.DEF_AREA_GAP)
        return go_to(ctx, robot, point, angle_to(point, ball))

    @staticmethod
    def _mode(ctx: TickContext) -> str:
        """Режим Опорного (открывание / оборона / страховка) с гистерезисом по времени"""
        world = ctx.world
        memory = ctx.memory
        possession = world.possession
        no_gk = world.gk is None
        if possession == Possession.OURS and not no_gk:
            wanted = "OPEN"
        elif possession == Possession.THEIRS or (no_gk and possession != Possession.OURS):
            wanted = "DEFEND"
        elif world.ball_race < params.SUPPORT_COVER_RACE:
            wanted = "COVER"
        else:
            wanted = "OPEN"
        urgent = wanted == "DEFEND" and possession == Possession.THEIRS
        if wanted != memory.support_mode and (urgent or ctx.now - memory.support_mode_since >= params.SUPPORT_MODE_HOLD):
            memory.support_mode = wanted
            memory.support_mode_since = ctx.now
        return memory.support_mode

    def _shot_face(self, ctx: TickContext, point: aux.Point) -> aux.Point:
        shot = evaluation.eval_shot(ctx.world, point)
        return shot.target if shot.p_goal > 0 else ctx.field.enemy_goal.center

    def _steal_pass(self, robot: rbt.Robot, ctx: TickContext) -> Optional[Action]:
        """Перехват паса соперника: успеваем раньше их лучшего на INTERCEPT_STEAL_MARGIN"""
        world = ctx.world
        last = ctx.memory.last_touch
        if not world.ball_moving or world.owner is not None or last is None or last[0] == ctx.field.ally_color:
            return None
        inter = world.our_intercepts.get(robot.r_id)
        if inter is None or math.isinf(inter.t) or inter.t > world.their_best_t - params.INTERCEPT_STEAL_MARGIN:
            return None
        if common.ball_coming_to(ctx, robot):
            return common.receive(ctx, robot, self._shot_face(ctx, robot.get_pos()))
        point = inter.point + world.ball_dir * const.ROBOT_R
        return go_to(ctx, robot, point, angle_to(point, world.ball))

    def open_up(self, robot: rbt.Robot, ctx: TickContext, att_pos: Optional[aux.Point]) -> Action:
        """Открывание в лучшую точку (7.3), добивание при ударе Нападающего (P2)"""
        world = ctx.world
        decision = ctx.memory.attacker_decision
        if decision is not None and decision.kind == "SHOOT" and decision.target is not None:
            point = self._rebound_point(ctx, robot, decision.target)
        else:
            point = evaluation.best_support_point(world, ctx.memory, robot, att_pos)
        face = clamp_angle_towards(
            angle_to(point, world.ball), angle_to(point, ctx.field.enemy_goal.center), params.RECEIVE_MAX_BIAS
        )
        return go_to(ctx, robot, point, face)

    def _rebound_point(self, ctx: TickContext, robot: rbt.Robot, shot_target: aux.Point) -> aux.Point:
        """
        Добивание: у края чужой штрафной. ТЗ: со стороны, противоположной цели удара; но линию удара
        не пересекаем - если Опорный по другую сторону от неё, остаётся на своей стороне.
        """
        goal = ctx.field.enemy_goal
        ball = ctx.world.ball
        side = -1.0 if shot_target.y - goal.center.y > 0 else 1.0
        own_side = 1.0 if aux.vec_mult(shot_target - ball, robot.get_pos() - ball) > 0 else -1.0
        depth = const.GOAL_PEN_DX + const.ROBOT_R * 1.2 + params.REBOUND_AREA_GAP
        point = aux.Point(goal.center.x + goal.eye_forw.x * depth, side * const.GOAL_PEN_DY / 3)
        if (1.0 if aux.vec_mult(shot_target - ball, point - ball) > 0 else -1.0) != own_side:
            point = aux.Point(point.x, -point.y)
        return point

    def defend(self, robot: rbt.Robot, ctx: TickContext, attacker: Optional[rbt.Robot]) -> Action:
        """Соперник владеет: опека или перекрытие удара"""
        world = ctx.world
        ball = world.ball
        pressing = attacker is not None and aux.dist(attacker.get_pos(), ball) < params.PRESSURE_RADIUS + const.ROBOT_R
        if pressing and world.gk is not None:
            owner_id = world.owner.r_id if world.owner is not None and world.owner_is_theirs else -1
            target = most_dangerous_enemy(ctx, [owner_id])
            if target is not None:
                point = mark_point(ctx, target)
                return go_to(ctx, robot, point, angle_to(point, ball))
        point = shot_block_point(ctx)
        return go_to(ctx, robot, point, angle_to(point, ball))
