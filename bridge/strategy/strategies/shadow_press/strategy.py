"""
Strategy «Тень, прессинг, вратарь» (SSL Div. C, 3 × 3)

Roles are fixed to robots: goalkeeper G, shadow S (marks their forward F), attacker A.
Description: strategy_description.txt
"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.router.actions import Action, Actions
from bridge.strategy.base_strategy import BaseStrategy

from . import config as cfg
from . import geometry as geom
from .attacker import Attacker
from .context import Context
from .goalkeeper import Goalkeeper
from .shadow import Shadow
from .support import ReceivePoint


class ShadowPressStrategy(BaseStrategy):
    """«Shadow, press, goalkeeper» strategy"""

    def __init__(self) -> None:
        super().__init__()
        self.goalkeeper = Goalkeeper()
        self.shadow = Shadow()
        self.attacker = Attacker()
        self.receive = ReceivePoint()

        self.forward_id: Optional[int] = None
        self.ball_taker: Optional[str] = None
        self.extra: list[rbt.Robot] = []

    # ---------------------------------------------------------------- context

    def make_context(self, field: fld.Field) -> Context:
        """Assign roles and collect the data of the iteration"""
        gk, shadow, attacker = self.assign_roles(field)
        ctx = Context(field, self.we_active, gk, shadow, attacker, self.find_forward(field))
        ctx.ball_taker = self.choose_ball_taker(ctx)
        ctx.receive_point = self.receive.update(ctx)
        return ctx

    def assign_roles(self, field: fld.Field) -> tuple[Optional[rbt.Robot], Optional[rbt.Robot], Optional[rbt.Robot]]:
        """
        G is the GK from .env, S and A are taken by ids from config.
        If robots are missing: 2 robots -> G and S, 1 robot -> G
        """
        gk_robot = field.allies[field.gk_id]
        gk = gk_robot if gk_robot.is_used() else None

        free = {r.r_id: r for r in field.active_allies(include_gk=False)}
        shadow = free.pop(cfg.SHADOW_ID, None) if cfg.SHADOW_ID is not None else None
        attacker = free.pop(cfg.ATTACKER_ID, None) if cfg.ATTACKER_ID is not None else None

        rest = sorted(free.values(), key=lambda r: r.r_id)
        if shadow is None:
            if rest:
                shadow = rest.pop(0)
            elif attacker is not None:
                shadow, attacker = attacker, None
        if attacker is None and rest:
            attacker = rest.pop(0)

        self.extra = rest
        return gk, shadow, attacker

    def find_forward(self, field: fld.Field) -> Optional[rbt.Robot]:
        """Their forward F: the enemy (not GK) closest to our goal, with hysteresis"""
        enemies = field.active_enemies(include_gk=False)
        if not enemies:
            self.forward_id = None
            return None

        def dist_to_goal(robot: rbt.Robot) -> float:
            return aux.dist(robot.get_pos(), field.ally_goal.center)

        best = min(enemies, key=dist_to_goal)
        current = next((e for e in enemies if e.r_id == self.forward_id), None)
        if current is None or dist_to_goal(best) + cfg.FORWARD_HYST < dist_to_goal(current):
            current = best
        self.forward_id = current.r_id
        return current

    def choose_ball_taker(self, ctx: Context) -> Optional[str]:
        """A free ball on our half (out of the zone) is taken by the closer of S and A"""
        field = ctx.field
        if (
            ctx.shadow is None
            or ctx.attacker is None
            or field.robot_with_ball is not None
            or not geom.in_ally_half(field, ctx.ball)
            or aux.is_point_inside_poly(ctx.ball, field.ally_goal.hull)
        ):
            self.ball_taker = None
            return None

        # distance to the own interception point (for a still ball it's the distance to the ball)
        def reach(robot: rbt.Robot) -> float:
            return aux.dist(robot.get_pos(), geom.intercept_point(field, robot.get_pos()))

        d_s, d_a = reach(ctx.shadow), reach(ctx.attacker)
        if self.ball_taker == "S" and d_a + cfg.TAKER_HYST < d_s:
            self.ball_taker = "A"
        elif self.ball_taker == "A" and d_s + cfg.TAKER_HYST < d_a:
            self.ball_taker = "S"
        elif self.ball_taker is None:
            self.ball_taker = "S" if d_s < d_a else "A"
        return self.ball_taker

    def kicker(self, ctx: Context, penalty: bool = False) -> Optional[rbt.Robot]:
        """Robot that takes our set piece"""
        if penalty and cfg.PENALTY_KICKER_ID is not None and cfg.PENALTY_KICKER_ID != ctx.field.gk_id:
            robot = ctx.field.allies[cfg.PENALTY_KICKER_ID]
            if robot.is_used():
                return robot
        return ctx.attacker if ctx.attacker is not None else ctx.shadow

    # ---------------------------------------------------------------- game states

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        ctx = self.make_context(field)
        if ctx.gk is not None:
            actions[ctx.gk.r_id] = self.goalkeeper.act(ctx, ctx.gk)
        if ctx.shadow is not None:
            actions[ctx.shadow.r_id] = self.shadow.act(ctx, ctx.shadow)
            ctx.shadow_on_ball = self.shadow.on_ball
        if ctx.attacker is not None:
            actions[ctx.attacker.r_id] = self.attacker.act(ctx, ctx.attacker)
        self.extra_stay(ctx, actions)
        self.draw(ctx)

    def stop(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        ctx = self.make_context(field)
        self.defend(ctx, actions)
        self.draw(ctx)

    def ball_placement(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        ctx = self.make_context(field)
        self.defend(ctx, actions)
        self.draw(ctx)

    def free_kick(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        ctx = self.make_context(field)
        kicker = self.kicker(ctx) if self.we_active else None
        self.defend(ctx, actions, exclude=kicker)
        if kicker is not None:
            actions[kicker.r_id] = self.attacker.free_kick(ctx, kicker)
        self.draw(ctx)

    def prepare_kickoff(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        ctx = self.make_context(field)
        kicker = self.kicker(ctx) if self.we_active else None
        self.defend(ctx, actions, own_half=True, exclude=kicker)
        if kicker is not None:
            self.attacker.state = "KICKOFF PREP"
            point = geom.to_own_half(field, ctx.ball - field.ally_goal.eye_forw * cfg.KICKOFF_BEHIND)
            actions[kicker.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ctx.ball))
        self.draw(ctx)

    def kickoff(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        ctx = self.make_context(field)
        kicker = self.kicker(ctx) if self.we_active else None
        self.defend(ctx, actions, own_half=True, exclude=kicker)
        if kicker is not None:
            actions[kicker.r_id] = self.attacker.kickoff_shot(ctx, kicker)
        self.draw(ctx)

    def prepare_penalty(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        self.penalty_positions(self.make_context(field), actions, kick=False)

    def penalty(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        self.penalty_positions(self.make_context(field), actions, kick=True)

    # ---------------------------------------------------------------- set pieces

    def defend(
        self,
        ctx: Context,
        actions: list[Optional[Action]],
        own_half: bool = False,
        exclude: Optional[rbt.Robot] = None,
    ) -> None:
        """Their set pieces and STOP: A on the line "ball -> F", S marks F, G on its place"""
        field = ctx.field
        points: list[tuple[Optional[rbt.Robot], aux.Point]] = []
        if ctx.shadow is not None:
            self.shadow.state = "MARK"
            points.append((ctx.shadow, self.shadow.mark_point(ctx)))
        if ctx.attacker is not None:
            self.attacker.state = "BLOCK"
            points.append((ctx.attacker, self.attacker.block_point(ctx, cfg.SET_PIECE_DIST)))

        for robot, point in points:
            if robot is None or robot == exclude:
                continue
            if own_half:
                point = geom.to_own_half(field, point)
            actions[robot.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ctx.ball))

        if ctx.gk is not None:
            actions[ctx.gk.r_id] = self.goalkeeper.stay(ctx)
        self.extra_stay(ctx, actions)

    def penalty_positions(self, ctx: Context, actions: list[Optional[Action]], kick: bool) -> None:
        """Penalty: kicker shoots (ours) or G moves along the goal line (theirs), others stay behind the ball"""
        field = ctx.field
        forw = field.ally_goal.eye_forw  # from our goal to theirs
        kicker = self.kicker(ctx, penalty=True) if self.we_active else None

        if self.we_active:
            behind = ctx.ball - forw * cfg.PENALTY_OTHERS_DIST  # to our side
            if ctx.gk is not None and ctx.gk != kicker:
                actions[ctx.gk.r_id] = self.goalkeeper.stay(ctx)
        else:
            behind = ctx.ball + forw * cfg.PENALTY_OTHERS_DIST  # to their side
            if ctx.gk is not None:
                actions[ctx.gk.r_id] = self.goalkeeper.penalty_defense(ctx)

        others = [r for r in [ctx.shadow, ctx.attacker, *self.extra] if r is not None and r != kicker]
        for i, robot in enumerate(others):
            offset = (i // 2 + 1) * 400 * (1 if i % 2 == 0 else -1)
            point = aux.nearest_point_in_poly(behind + aux.Point(0, offset), field.small_hull)
            actions[robot.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ctx.ball))

        if kicker is not None:
            if kick:
                actions[kicker.r_id] = self.attacker.penalty(ctx)
            else:
                goal = field.enemy_goal.center
                point = ctx.ball - (goal - ctx.ball).unity() * (const.ROBOT_R + 200)
                actions[kicker.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, goal))
        self.draw(ctx)

    def extra_stay(self, ctx: Context, actions: list[Optional[Action]]) -> None:
        """Robots without a role (more than 3 robots): stand still, looking at the ball"""
        for robot in self.extra:
            actions[robot.r_id] = Actions.GoToPoint(robot.get_pos(), aux.angle_to_point(robot.get_pos(), ctx.ball))

    # ---------------------------------------------------------------- debug

    def draw(self, ctx: Context) -> None:
        """Roles and states in the strategy layer"""
        image = ctx.field.strategy_image
        labels = [
            (ctx.gk, "G", self.goalkeeper.state),
            (ctx.shadow, "S", self.shadow.state),
            (ctx.attacker, "A", self.attacker.state),
        ]
        for robot, role, state in labels:
            if robot is not None:
                image.print(robot.get_pos() + aux.Point(0, 250), f"{robot.r_id} {role}: {state}", (255, 255, 0))
        if ctx.forward is not None:
            image.draw_circle(ctx.forward.get_pos(), (255, 0, 0), const.ROBOT_R + 40)
            image.print(ctx.forward.get_pos() + aux.Point(0, 250), "F", (255, 0, 0))
        if ctx.receive_point is not None:
            image.draw_circle(ctx.receive_point, (0, 255, 0), 60)
            image.print(ctx.receive_point + aux.Point(0, 120), "receive", (0, 255, 0))
        if ctx.ball_taker is not None:
            image.print(ctx.ball + aux.Point(0, 150), f"taker: {ctx.ball_taker}", (0, 255, 255))
