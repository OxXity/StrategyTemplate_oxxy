"""Point where A waits for a pass from S: on the way to their goal, with a free pass line"""

from typing import Optional

from bridge import const
from bridge.auxiliary import aux

from . import config as cfg
from . import geometry as geom
from .context import Context


class ReceivePoint:
    """Chooses the point for A to receive a pass (with hysteresis)"""

    def __init__(self) -> None:
        self.point: Optional[aux.Point] = None

    def update(self, ctx: Context) -> Optional[aux.Point]:
        """Best point on the grid between the center line and their zone"""
        if ctx.attacker is None:
            self.point = None
            return None

        field = ctx.field
        forw = aux.sign(field.enemy_goal.center.x)  # direction to their goal along x
        x_min: float = cfg.RECEIVE_X_MIN
        x_max: float = const.GOAL_DX - const.GOAL_PEN_DX - cfg.RECEIVE_ZONE_GAP
        # the point is ahead of the ball, if possible
        x_min = aux.minmax(ctx.ball.x * forw + cfg.RECEIVE_AHEAD, x_min, x_max)

        candidates = []
        steps_x, steps_y = 6, 9
        for i in range(steps_x):
            x = (x_min + (x_max - x_min) * i / (steps_x - 1)) * forw
            for j in range(steps_y):
                y = -cfg.RECEIVE_Y + 2 * cfg.RECEIVE_Y * j / (steps_y - 1)
                candidates.append(aux.Point(x, y))

        best = max(candidates, key=lambda p: self.score(ctx, p, forw))
        if self.point is not None and self.score(ctx, self.point, forw) + cfg.RECEIVE_HYST > self.score(ctx, best, forw):
            best = self.point
        self.point = best
        return best

    def score(self, ctx: Context, point: aux.Point, forw: int) -> float:
        """The higher, the better the point for receiving"""
        field = ctx.field
        enemies = field.active_enemies(True)
        value = 0.0

        # free pass line from the ball
        if not geom.is_line_free(ctx.ball, point, enemies):
            value -= 3000

        # not too close to the ball (and to S), not too far for a pass
        d = aux.dist(ctx.ball, point)
        if d < cfg.RECEIVE_MIN_DIST:
            value -= (cfg.RECEIVE_MIN_DIST - d) * cfg.RECEIVE_MIN_DIST_K
        if d > cfg.RECEIVE_MAX_DIST:
            value -= (d - cfg.RECEIVE_MAX_DIST) * 2

        # far from enemies, especially from F
        clearance = min((aux.dist(point, e.get_pos()) for e in enemies), default=1e4)
        value += 1.5 * min(clearance, 1000)
        f_pos = ctx.forward_pos()
        if f_pos is not None and aux.dist(point, f_pos) < cfg.PASS_SAFE_DIST:
            value -= 1000

        # small turn after receiving: the ball comes from the side, not from the goal direction
        value -= cfg.RECEIVE_TURN_K * geom.turn_after_receive(point, ctx.ball, field.enemy_goal.center)

        # not near the side line
        value -= cfg.RECEIVE_SIDE_K * max(abs(point.y) - cfg.RECEIVE_SIDE_FREE, 0)

        # closer to their goal, with a chance to shoot
        value += 0.5 * point.x * forw
        if geom.widest_window(point, field.enemy_goal, enemies) is not None:
            value += 400

        # don't go far without need
        if ctx.attacker is not None:
            value -= cfg.RECEIVE_TRAVEL_K * aux.dist(ctx.attacker.get_pos(), point)
        return value
