"""
PREPARE_KICKOFF / KICKOFF (ТЗ, 9.4)
"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, KickActions
from bridge.strategy import evaluation, params
from bridge.strategy.context import TickContext, angle_to
from bridge.strategy.guards import go_to
from bridge.strategy.memory import Role
from bridge.strategy.plays.open_play import safe_act
from bridge.strategy.roles import common, goalkeeper


def kicker_of(ctx: TickContext) -> Optional[rbt.Robot]:
    """Исполнитель - Нападающий; выбранный исполнитель не меняется до конца розыгрыша"""
    prev = ctx.memory.set_piece_kicker_id
    if (
        prev is not None
        and any(r.r_id == prev for r in ctx.world.field_allies)
        and not ctx.is_banned(ctx.field.allies[prev])
    ):
        return ctx.field.allies[prev]
    attacker = ctx.robot_with_role(Role.ATTACKER)
    if attacker is not None:
        return attacker
    return ctx.world.field_allies[0] if ctx.world.field_allies else None


def receiver_point(ctx: TickContext) -> aux.Point:
    """S: на нашей половине на фланге, где меньше соперников"""
    upper = sum(1 for e in ctx.world.enemies if e.get_pos().y > 0)
    lower = sum(1 for e in ctx.world.enemies if e.get_pos().y < 0)
    side = -1.0 if upper > lower else 1.0
    return ctx.world.from_att(-params.KICKOFF_RECEIVER_X, side * params.KICKOFF_RECEIVER_Y)


def own_half_point(ctx: TickContext, index: int) -> aux.Point:
    """Запасная позиция на своей половине"""
    side = 1.0 if index % 2 == 0 else -1.0
    return ctx.world.from_att(-const.FIELD_DX / 2, side * const.FIELD_DY / 2)


def _ours(ctx: TickContext, started: bool) -> dict[int, Action]:
    world = ctx.world
    ball = world.ball
    actions: dict[int, Action] = {}
    kicker = kicker_of(ctx)
    s_point = receiver_point(ctx)
    ctx.memory.set_piece_kicker_id = kicker.r_id if kicker is not None else None
    receiver = ctx.robot_with_role(Role.SUPPORT)
    if receiver is None or (kicker is not None and receiver.r_id == kicker.r_id):
        receiver = None

    def kicker_act(robot: rbt.Robot, ctx_: TickContext) -> Action:
        if not started:
            # За мячом на линии S -> мяч в центральном круге, лицом к S, мяча не касаться
            point = ball + (ball - s_point).unity() * params.KICKOFF_KICKER_DIST
            return go_to(ctx_, robot, point, angle_to(point, s_point), avoid_ball=False)
        if params.KICKOFF_DIRECT_SHOT:
            shot = evaluation.eval_shot(world, ball, robot.r_id)
            if shot.p_goal >= params.SHOT_MIN_P:
                return set_piece_kick(ctx_, shot.target, ctx_.kick_model.shot_voltage())
        target = receiver.get_pos() if receiver is not None and aux.dist(receiver.get_pos(), s_point) < 300 else s_point
        voltage, t_arrive = common.pass_kick(ctx_, robot, target)
        if receiver is not None:
            common.register_pass(ctx_, robot, receiver.r_id, target, t_arrive)
        return set_piece_kick(ctx_, target, voltage)

    def receiver_act(robot: rbt.Robot, ctx_: TickContext) -> Action:
        if started and common.ball_coming_to(ctx_, robot):
            return common.receive(ctx_, robot, ctx_.field.enemy_goal.center)
        return go_to(ctx_, robot, s_point, angle_to(s_point, ball))

    for i, robot in enumerate(world.allies):
        if ctx.is_gk(robot):
            actions[robot.r_id] = safe_act(ctx, robot, "kickoff_gk", gk_position)
        elif kicker is not None and robot.r_id == kicker.r_id:
            actions[robot.r_id] = safe_act(ctx, robot, "kickoff_kicker", kicker_act)
        elif receiver is not None and robot.r_id == receiver.r_id:
            actions[robot.r_id] = safe_act(ctx, robot, "kickoff_receiver", receiver_act)
        else:
            point = own_half_point(ctx, i)
            actions[robot.r_id] = go_to(ctx, robot, point, angle_to(point, ball))
    return actions


def _theirs(ctx: TickContext) -> dict[int, Action]:
    world = ctx.world
    ball = world.ball
    goal = ctx.field.ally_goal.center
    actions: dict[int, Action] = {}
    upper = sum(1 for e in world.enemies if e.get_pos().y > 0)
    lower = sum(1 for e in world.enemies if e.get_pos().y < 0)
    pass_side = 1.0 if upper >= lower else -1.0
    for i, robot in enumerate(world.allies):
        role = ctx.roles.get(robot.r_id, Role.SUPPORT)
        if role == Role.GOALKEEPER:
            actions[robot.r_id] = safe_act(ctx, robot, "kickoff_gk", gk_position)
            continue
        if role == Role.ATTACKER:
            point = ball + (goal - ball).unity() * params.KICKOFF_ENEMY_BLOCK_DIST
        elif role == Role.SUPPORT:
            # Фланг ближе к нашим воротам: перекрыть вероятный первый пас
            point = world.from_att(-const.FIELD_DX / 3, pass_side * const.FIELD_DY / 2)
        else:
            point = own_half_point(ctx, i)
        actions[robot.r_id] = go_to(ctx, robot, point, angle_to(point, ball))
    return actions


def gk_position(robot: rbt.Robot, ctx: TickContext) -> Action:
    """Обычное позиционирование вратаря без выхода"""
    point = goalkeeper.position_target(ctx, allow_come_out=False)
    return go_to(ctx, robot, point, angle_to(point, ctx.world.ball))


def set_piece_kick(ctx: TickContext, target: aux.Point, voltage: int) -> Action:
    """Удар со стандарта: дриблер 0, медленный финальный подъезд, автоудар при довороте"""
    ctx.draw.line(ctx.world.ball, target, (0, 200, 255), 2)
    return KickActions.SetPieceKick(
        target,
        voltage,
        params.SET_PIECE_APPROACH_SPEED,
        params.SET_PIECE_ALIGN_TOL,
        prep_gap=params.SET_PIECE_PREP_GAP,
        lateral_tol=params.SET_PIECE_LATERAL_TOL,
    )


def prepare(ctx: TickContext) -> dict[int, Action]:
    """PREPARE_KICKOFF"""
    return _ours(ctx, started=False) if ctx.we_active else _theirs(ctx)


def kickoff(ctx: TickContext) -> dict[int, Action]:
    """KICKOFF после Normal Start"""
    return _ours(ctx, started=True) if ctx.we_active else _theirs(ctx)
