"""Example strategy from the template: one attacker grabs the ball and shoots, GK stands in front of the goal"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.router.actions import Action, Actions, KickActions
from bridge.strategy.base_strategy import BaseStrategy


class ExampleStrategy(BaseStrategy):
    """Example strategy"""

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """
        ONE ITERATION of strategy
        NOTE: robots will not start acting until this function returns an array of actions,
              if an action is overwritten during the process, only the last one will be executed)

        Examples of getting coordinates:
        - field.allies[8].get_pos(): aux.Point -   coordinates  of the 8th  robot from the allies
        - field.enemies[14].get_angle(): float - rotation angle of the 14th robot from the opponents

        - field.ally_goal.center: Point - center of the ally goal
        - field.enemy_goal.hull: list[Point] - polygon around the enemy goal area


        Examples of robot control:
        - actions[2] = Actions.GoToPoint(aux.Point(1000, 500), math.pi / 2)
                The robot number 2 will go to the point (1000, 500), looking in the direction π/2 (up, along the OY axis)

        - actions[3] = Actions.Kick(field.enemy_goal.center)
                The robot number 3 will hit the ball to 'field.enemy_goal.center' (to the center of the enemy goal)

        - actions[9] = Actions.BallGrab(0.0)
                The robot number 9 grabs the ball at an angle of 0.0 (it looks to the right, along the OX axis)
        """
        ball = field.ball.get_pos()
        goal = field.enemy_goal.center

        # 1. Choose the attacker: the active ally (without GK) closest to the ball
        allies = field.active_allies(include_gk=False)
        if not allies:
            return
        attacker = min(allies, key=lambda r: aux.dist(r.get_pos(), ball))
        idx = attacker.r_id

        # 2. Attacker: grab the ball facing the goal, then kick
        kick_angle = aux.angle_to_point(ball, goal)
        if not field.is_ball_in(attacker):
            # Phase 1: approach the ball from behind and grab it with the dribbler
            actions[idx] = Actions.BallGrab(kick_angle)
            phase = "GRAB"
        else:
            # Phase 2: ball is in the dribbler -> turn to the goal and shoot
            # (Straight itself keeps holding the ball and fires only when aligned)
            actions[idx] = KickActions.Straight(goal)
            phase = "KICK"

        # NOTE: KickActions.Straight(goal) alone already does grab + aim + shoot,
        #       phases are split here only to show how to build logic by state

        # 3. Goalkeeper: stand in front of the goal center, looking at the ball
        gk = field.allies[field.gk_id]
        if gk.is_used():
            gk_pos = field.ally_goal.center + field.ally_goal.eye_forw * const.GK_FORW
            actions[field.gk_id] = Actions.GoToPoint(gk_pos, aux.angle_to_point(gk_pos, ball))

        # 4. Other robots: stand still, looking at the ball
        for robot in allies:
            if robot.r_id != idx:
                actions[robot.r_id] = Actions.GoToPoint(robot.get_pos(), aux.angle_to_point(robot.get_pos(), ball))

        # 5. Debug drawing (visible in the strategy layer of the visualizer)
        field.strategy_image.draw_line(ball, goal, (255, 0, 0), 3)
        field.strategy_image.draw_circle(attacker.get_pos(), (0, 255, 0), const.ROBOT_R + 30)
        field.strategy_image.print(attacker.get_pos() + aux.Point(0, 250), f"{idx}: {phase}", (255, 255, 0))
