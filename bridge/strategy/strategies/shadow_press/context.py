"""Data of one strategy iteration, shared between the roles"""

from time import time
from typing import Optional

from bridge.auxiliary import aux, fld, rbt

from . import config as cfg
from . import geometry as geom


class Context:
    """Data of one strategy iteration"""

    def __init__(
        self,
        field: fld.Field,
        we_active: bool,
        gk: Optional[rbt.Robot],
        shadow: Optional[rbt.Robot],
        attacker: Optional[rbt.Robot],
        forward: Optional[rbt.Robot],
    ) -> None:
        self.field = field
        self.we_active = we_active
        self.now = time()
        self.ball = field.ball.get_pos()

        self.gk = gk
        self.shadow = shadow
        self.attacker = attacker
        self.forward = forward  # their forward F

        # who of S and A takes the free ball on our half: "S", "A" or None
        self.ball_taker: Optional[str] = None
        # where A waits for a pass from S
        self.receive_point: Optional[aux.Point] = None
        # S is busy with the ball (tackle, interception, pass): A must not go to the ball
        self.shadow_on_ball = False

    @property
    def enemy_with_ball(self) -> Optional[rbt.Robot]:
        """Enemy robot that holds the ball"""
        owner = self.field.robot_with_ball
        if owner is not None and owner.color != self.field.ally_color:
            return owner
        return None

    def has_ball(self, robot: Optional[rbt.Robot]) -> bool:
        """The robot holds the ball"""
        return robot is not None and self.field.is_ball_in(robot)

    def is_rolling(self) -> bool:
        """The ball rolls freely"""
        return self.field.robot_with_ball is None and self.field.ball.get_vel().mag() > cfg.ROLLING_SPEED

    def ball_at_our_gk(self) -> bool:
        """The ball is held by our goalkeeper or lies in our zone"""
        if self.has_ball(self.gk):
            return True
        return (
            self.gk is not None
            and aux.is_point_inside_poly(self.ball, self.field.ally_goal.hull)
            and self.field.ball.get_vel().mag() < cfg.GK_TAKE_SPEED
        )

    def ball_at_their_gk(self) -> bool:
        """The ball is held by their goalkeeper or lies in their zone"""
        owner = self.enemy_with_ball
        if owner is not None and owner.r_id == self.field.enemy_gk_id:
            return True
        return aux.is_point_inside_poly(self.ball, self.field.enemy_goal.hull)

    def obstacles(self, exclude: Optional[rbt.Robot] = None) -> list[rbt.Robot]:
        """All robots on the field except exclude"""
        robots = self.field.active_enemies(True) + self.field.active_allies(True)
        return [r for r in robots if r != exclude]

    def pass_target(self, src: aux.Point, passer: rbt.Robot, to_receive_point: bool = False) -> Optional[aux.Point]:
        """
        Point for a pass to A, None if the pass line isn't free or A is close to F
        to_receive_point: pass to the receive point (on lead, when A is close to it), otherwise to A itself
        """
        if self.attacker is None:
            return None
        target = self.attacker.get_pos()
        if to_receive_point:
            if self.receive_point is None or aux.dist(target, self.receive_point) > cfg.RECEIVE_READY_DIST:
                return None
            target = self.receive_point
        if aux.is_point_inside_poly(target, self.field.ally_goal.big_hull):
            return None
        if aux.dist(src, target) < cfg.PASS_MIN_DIST:
            return None
        if self.forward is not None and aux.dist(target, self.forward.get_pos()) < cfg.PASS_SAFE_DIST:
            return None
        others = [r for r in self.obstacles(passer) if r != self.attacker]
        return target if geom.is_line_free(src, target, others) else None

    def attacker_on_the_way(self) -> bool:
        """A goes to its receive point and will be there soon"""
        return (
            self.attacker is not None
            and self.receive_point is not None
            and aux.dist(self.attacker.get_pos(), self.receive_point) < cfg.RECEIVE_WAIT_DIST
        )

    def forward_pos(self) -> Optional[aux.Point]:
        """Position of their forward"""
        return self.forward.get_pos() if self.forward is not None else None
