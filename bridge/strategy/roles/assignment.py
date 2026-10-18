"""
Назначение ролей (ТЗ, 8.0)
"""

import math

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.strategy import params
from bridge.strategy.context import TickContext
from bridge.strategy.memory import Role

SET_PIECE_STATES = (
    const.State.PREPARE_KICKOFF,
    const.State.KICKOFF,
    const.State.FREE_KICK,
    const.State.PREPARE_PENALTY,
    const.State.PENALTY,
)


def attacker_cost(ctx: TickContext, robot: rbt.Robot) -> float:
    """Стоимость захвата: t_перехвата (с разворотом) + штраф за неправильную сторону мяча"""
    world = ctx.world
    inter = world.our_intercepts.get(robot.r_id)
    if inter is None or math.isinf(inter.t):
        cost = 10.0 + aux.dist(robot.get_pos(), world.ball) / const.MAX_SPEED
        point = world.ball
    else:
        cost = inter.t
        point = inter.point
    if world.x_att(robot.get_pos()) > world.x_att(point) + const.ROBOT_R:
        cost += params.ROLE_WRONG_SIDE_PENALTY
    return cost


def assign_roles(ctx: TickContext) -> dict[int, Role]:
    """Роли всех активных роботов с гистерезисом и исключениями 8.0"""
    world = ctx.world
    memory = ctx.memory
    now = ctx.now
    roles: dict[int, Role] = {}
    if world.gk is not None:
        roles[world.gk.r_id] = Role.GOALKEEPER

    fields = sorted(world.field_allies, key=lambda r: r.r_id)
    field_ids = [r.r_id for r in fields]
    prev_attacker = next((r_id for r_id, r in memory.roles.items() if r == Role.ATTACKER and r_id in field_ids), None)
    prev_support = next((r_id for r_id, r in memory.roles.items() if r == Role.SUPPORT and r_id in field_ids), None)

    attacker = None
    if len(fields) == 1:
        attacker = fields[0].r_id
    elif len(fields) > 1:
        costs = {r.r_id: attacker_cost(ctx, r) for r in fields}
        banned = memory.double_touch_banned_id
        forced = None
        support_forced = None
        if world.owner_is_ours and world.owner is not None and world.owner.r_id in field_ids:
            forced = world.owner.r_id
        elif memory.pass_in_flight is not None and memory.pass_in_flight.to_id in field_ids:
            forced = memory.pass_in_flight.to_id
            if memory.pass_in_flight.from_id in field_ids:
                support_forced = memory.pass_in_flight.from_id
        kicker = memory.set_piece_kicker_id
        if ctx.we_active and ctx.field.game_state in SET_PIECE_STATES and kicker in field_ids:
            forced = kicker  # исполнитель стандарта не меняется посреди розыгрыша
        if forced is not None and forced == banned:
            forced = None

        if forced is not None:
            attacker = forced
        else:
            candidates = [r_id for r_id in field_ids if r_id != banned] or field_ids
            best = min(candidates, key=lambda r_id: (costs[r_id], r_id))
            current = prev_attacker if prev_attacker in candidates else None
            if current is not None and best != current:
                held = now - memory.role_since.get(current, 0.0)
                gain = costs[current] - costs[best]
                if gain <= params.ROLE_SWITCH_MARGIN or held < params.ROLE_MIN_HOLD:
                    best = current
            attacker = best

        others = [r_id for r_id in field_ids if r_id != attacker]
        if support_forced is not None and support_forced in others:
            support = support_forced
        elif prev_support in others:
            support = prev_support
        else:
            support = min(others, key=lambda r_id: (costs[r_id], r_id))
        roles[support] = Role.SUPPORT
        for r_id in others:
            if r_id != support:
                roles[r_id] = Role.DEFENDER

    if attacker is not None:
        roles[attacker] = Role.ATTACKER

    for r_id, role in roles.items():
        if memory.roles.get(r_id) != role:
            memory.role_since[r_id] = now
    memory.roles = roles
    ctx.roles = roles
    return roles
