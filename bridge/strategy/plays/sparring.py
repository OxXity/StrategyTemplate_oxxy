"""
Простой базовый соперник для спарринга (ТЗ, 9.10, P2)

Ближайший к мячу едет к мячу и бьёт по воротам, остальные защищают ворота.
Работает только на роутерных действиях и легальности, без памяти стратегии.
"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.router import legality
from bridge.router.actions import Action, Actions, KickActions

# Защитники спарринга: на линии «мяч -> ворота» на таком расстоянии от ворот
SPARRING_DEFENSE_DIST = const.GOAL_PEN_DX + const.ROBOT_R * 3


def play(field: fld.Field) -> list[Optional[Action]]:
    """Действия всех роботов спарринг-команды"""
    actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
    allies = field.active_allies(True)
    state = field.game_state
    if state in (const.State.HALT, const.State.TIMEOUT):
        for robot in allies:
            actions[robot.r_id] = Actions.Stop()
        return actions

    ball = field.ball.get_pos()
    goal = field.ally_goal
    can_play = state in (const.State.RUN, const.State.DEBUG) or (
        state in (const.State.KICKOFF, const.State.FREE_KICK, const.State.PENALTY)
        and field.active_team in (const.Color.ALL, field.ally_color)
    )
    field_players = [r for r in allies if r.r_id != field.gk_id]
    striker = min(field_players, key=lambda r: aux.dist(r.get_pos(), ball)) if field_players else None
    keep = const.KEEP_BALL_DIST if fld.is_ball_not_in_play(state) else 0.0
    for i, robot in enumerate(allies):
        if robot.r_id == field.gk_id:
            y = aux.minmax(ball.y, -const.GOAL_DY / 2 + const.ROBOT_R, const.GOAL_DY / 2 - const.ROBOT_R)
            point = goal.center + goal.eye_forw * (const.ROBOT_R + 30) + aux.Point(0, y)
            point = legality.legal_target(field, robot.r_id, point, keep_ball_dist=keep)
            actions[robot.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ball), ignore_robots={})
        elif striker is not None and robot.r_id == striker.r_id and can_play:
            actions[robot.r_id] = KickActions.Straight(field.enemy_goal.center, const.VOLTAGE_SHOOT, dribbler_speed=0)
        else:
            shift = aux.rotate((ball - goal.center).unity(), 0.4 * (1 if i % 2 else -1))
            point = goal.center + shift * SPARRING_DEFENSE_DIST
            point = legality.legal_target(field, robot.r_id, point, keep_ball_dist=keep)
            actions[robot.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ball), ignore_robots={})
    return actions
