"""
Защитник (ТЗ, 8.4, P2): только если полевых больше двух
"""

from bridge import const
from bridge.auxiliary import rbt
from bridge.router import legality
from bridge.router.actions import Action
from bridge.strategy import params
from bridge.strategy.context import TickContext, angle_to
from bridge.strategy.guards import go_to
from bridge.strategy.memory import Possession, Role
from bridge.strategy.roles.support import mark_point, most_dangerous_enemy


class Defender:
    """Защитник (P2): вторая линия - опека второго по опасности или центр створа"""

    def act(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Действие Защитника"""
        world = ctx.world
        ball = world.ball
        exclude = [world.owner.r_id] if world.owner is not None and world.owner_is_theirs else []
        support = ctx.robot_with_role(Role.SUPPORT)
        if world.possession != Possession.OURS:
            first = most_dangerous_enemy(ctx, exclude)
            if first is not None:
                second = most_dangerous_enemy(ctx, exclude + [first.r_id])
                target = second if second is not None and support is not None else first
                point = mark_point(ctx, target)
                return go_to(ctx, robot, point, angle_to(point, ball))
        goal = ctx.field.ally_goal
        point = goal.center + (ball - goal.center).unity() * (const.GOAL_PEN_DX + const.ROBOT_R * 3)
        point = legality.push_out_of_area(point, goal, params.DEF_AREA_GAP)
        return go_to(ctx, robot, point, angle_to(point, ball))
