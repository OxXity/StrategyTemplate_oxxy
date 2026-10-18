"""
STOP (ТЗ, 9.3): никаких действий с мячом, расстановка для любого возобновления.
Та же расстановка - при мяче вне поля в RUN (R-6.9).
"""

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action
from bridge.strategy import evaluation, params
from bridge.strategy.context import TickContext, angle_to
from bridge.strategy.guards import go_to
from bridge.strategy.memory import Role
from bridge.strategy.plays.open_play import safe_act
from bridge.strategy.roles import goalkeeper, support


def stop_target(robot: rbt.Robot, ctx: TickContext) -> Action:
    """Цель робота в STOP по роли"""
    world = ctx.world
    ball = world.ball
    role = ctx.roles.get(robot.r_id, Role.SUPPORT)
    keep = params.KEEP_BALL_DIST + params.STOP_TARGET_EXTRA
    goal = ctx.field.ally_goal

    if role == Role.GOALKEEPER:
        point = goalkeeper.position_target(ctx, allow_come_out=False)
        return go_to(ctx, robot, point, angle_to(point, ball))

    if world.in_our_half(ball):
        if role == Role.SUPPORT or (role == Role.ATTACKER and len(world.field_allies) == 1):
            # «Стенка» на линии «мяч -> центр наших ворот»
            point = ball + (goal.center - ball).unity() * keep
            return go_to(ctx, robot, point, angle_to(point, ball))
        if role == Role.ATTACKER:
            # Противоположный фланг у центральной линии - для контратаки
            side = -1.0 if ball.y > 0 else 1.0
            point = world.from_att(-const.ROBOT_R * 3, side * const.FIELD_DY * 0.6)
            return go_to(ctx, robot, point, angle_to(point, ball))
    else:
        if role == Role.ATTACKER:
            # За мячом на линии «чужие ворота -> мяч», лицом к воротам: готов к своему штрафному
            enemy_goal = ctx.field.enemy_goal.center
            point = ball + (ball - enemy_goal).unity() * keep
            return go_to(ctx, robot, point, angle_to(point, enemy_goal))
        if role == Role.SUPPORT:
            attacker = ctx.robot_with_role(Role.ATTACKER)
            att_pos = attacker.get_pos() if attacker is not None else None
            point = evaluation.best_support_point(world, ctx.memory, robot, att_pos, ball_in_play=False)
            return go_to(ctx, robot, point, angle_to(point, ball))

    point = support.shot_block_point(ctx)
    if aux.dist(point, ball) < keep:
        point = ball + (goal.center - ball).unity() * keep + goal.eye_up * (2.5 * const.ROBOT_R)
    return go_to(ctx, robot, point, angle_to(point, ball))


def play(ctx: TickContext) -> dict[int, Action]:
    """Расстановка STOP"""
    return {robot.r_id: safe_act(ctx, robot, "stop", stop_target) for robot in ctx.world.allies}
