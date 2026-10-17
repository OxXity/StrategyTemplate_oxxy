"""Div C: recover to P0, two cross-field passes, then a corner shot.

A Transfer tracks the kick and its flight separately from the attack phase.
Issuing a kick is not evidence that a pass has arrived.
"""

import math
import random
from dataclasses import dataclass
from itertools import permutations
from time import monotonic
from typing import Callable, Optional

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.const import State as GameStates
from bridge.router.actions import Action, Actions, KickActions, StrategyActions

ATTACK_JITTER = 300
HALF_HYST = 200
PASS_BLOCK_DIST = 2 * const.ROBOT_R + const.BALL_R
POST_MARGIN = 100
HOLD_TIME = 1.5
FIELD_MARGIN = 2 * const.ROBOT_R
COLLECTOR_HYST = 250
RECEIVER_READY_DIST = 180
SETUP_TIMEOUT = 4.0
KICK_TIMEOUT = 4.0
FLIGHT_TIMEOUT = 6.0
CLEAR_DEPTHS = (600, 1400)
CLEAR_YS = (-1000, 0, 1000)
COVER_RATIO = 0.7


@dataclass
class Transfer:
    """One committed pass (or shot when receiver is None)."""

    sender: int
    receiver: Optional[int]
    target: aux.Point
    next_phase: str
    origin: aux.Point
    started: float
    released: Optional[float] = None


class Strategy:
    """State is private to this team and survives successive vision frames."""

    def __init__(self, *, clock: Callable[[], float] = monotonic, rng: Optional[random.Random] = None) -> None:
        self.clock = clock
        self.rng = rng if rng is not None else random.Random()
        self.we_active = False
        self.mode = "DEFENSE"
        self.attack_phase = "RECOVER"
        self.roles: dict[str, int] = {}
        self.attack_points: dict[str, aux.Point] = {}
        self.last_touch: Optional[const.Color] = None
        self.hold_start: Optional[float] = None
        self.hold_owner: Optional[int] = None
        self.setup_start: Optional[float] = None
        self.setup_owner: Optional[int] = None
        self.transfer: Optional[Transfer] = None
        self.collector: Optional[int] = None
        self.roster: tuple[int, ...] = ()

    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Replace active commands even on HALT: None leaves the old router action alive."""
        self.we_active = field.active_team in (const.Color.ALL, field.ally_color)
        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
        for robot in field.active_allies(True):
            actions[robot.r_id] = Actions.Stop()
        if field.game_state == GameStates.RUN:
            self.run(field, actions)
        else:
            self.reset_attack()
            self.mode = "DEFENSE"
            self.last_touch = None
            if field.game_state == GameStates.STOP:
                gk = field.allies[field.gk_id]
                if gk.is_used():
                    point = self.keeper_point(field)
                    actions[gk.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, field.ball.get_pos()))
                self.cover_pass_lines(field, field.active_allies(False), actions)
        return actions

    def reset_attack(self) -> None:
        self.attack_phase = "RECOVER"
        self.transfer = None
        self.collector = None
        self.hold_start = None
        self.hold_owner = None
        self.setup_start = None
        self.setup_owner = None

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        holder = field.robot_with_ball
        if holder is not None:
            self.last_touch = holder.color
        self.update_mode(field)
        if self.mode == "ATTACK":
            self.run_attack(field, actions)
        else:
            self.run_defense(field, actions)
        self.draw_state(field)

    def attack_depth(self, field: fld.Field, point: aux.Point) -> float:
        return aux.scal_mult(point, field.ally_goal.eye_forw)

    def update_mode(self, field: fld.Field) -> None:
        ball = field.ball.get_pos()
        holder = field.robot_with_ball
        enemy_has_ball = holder is not None and holder.color != field.ally_color
        # Protect a live combination, never an indefinitely stale last_touch.
        protected = (
            self.mode == "ATTACK"
            and not enemy_has_ball
            and (self.transfer is not None or (holder is not None and self.attack_phase in ("PASS_1", "PASS_2", "SHOT")))
        )
        behind_keeper = self.attack_depth(field, ball - field.allies[field.gk_id].get_pos()) < -150
        danger = field.is_ball_moves_to_goal() and (not protected or behind_keeper)
        depth = self.attack_depth(field, ball)
        if danger or (depth < -HALF_HYST and not protected):
            if self.mode != "DEFENSE":
                self.reset_attack()
            self.mode = "DEFENSE"
        elif depth > HALF_HYST and self.mode != "ATTACK":
            self.mode = "ATTACK"
            self.reset_attack()
            self.generate_attack_points(field)

    def generate_attack_points(self, field: fld.Field) -> None:
        """One layout per attack; reassign only if the available team changes."""
        self.roster = tuple(sorted(robot.r_id for robot in field.active_allies(True)))
        forward = field.ally_goal.eye_forw
        lateral = aux.Point(-forward.y, forward.x)
        side = self.rng.choice((-1, 1))
        self.attack_points = {"P0": field.ally_goal.frw + forward * 300}
        for role, depth, flank in (("F_near", 1500, side), ("F_far", 700, -side)):
            point = forward * (depth + self.rng.uniform(-ATTACK_JITTER, ATTACK_JITTER))
            point += lateral * (flank * 1000 + self.rng.uniform(-ATTACK_JITTER, ATTACK_JITTER))
            self.attack_points[role] = self.safe_point(field, point)
        self.roles = {}
        if field.allies[field.gk_id].is_used():
            self.roles["P0"] = field.gk_id
        forwards = field.active_allies(False)
        names = ("F_near", "F_far")[: len(forwards)]
        if names:
            selected = min(
                permutations(forwards, len(names)),
                key=lambda robots: sum(
                    aux.dist(robot.get_pos(), self.attack_points[name]) for name, robot in zip(names, robots)
                ),
            )
            self.roles.update({name: robot.r_id for name, robot in zip(names, selected)})

    def run_attack(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        roster = tuple(sorted(robot.r_id for robot in field.active_allies(True)))
        if roster != self.roster:
            self.reset_attack()
            self.generate_attack_points(field)
        ball = field.ball.get_pos()
        for role, r_id in self.roles.items():
            point = self.attack_points[role]
            actions[r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ball))
        holder = field.robot_with_ball
        if holder is not None and holder.color != field.ally_color:
            self.reset_attack()
        if self.drive_transfer(field, actions):
            return
        if aux.is_point_inside_poly(ball, field.enemy_goal.hull):
            self.reset_attack()
            return
        if aux.is_point_inside_poly(ball, field.ally_goal.hull):
            self.reset_attack()
            self.mode = "DEFENSE"
            self.run_defense(field, actions)
            return

        if self.attack_phase == "RECOVER":
            if "P0" in self.roles and field.is_ball_in(field.allies[self.roles["P0"]]):
                self.attack_phase = "PASS_1"
            else:
                collector = self.choose_collector(field)
                if collector is None:
                    if "P0" in self.roles:
                        actions[self.roles["P0"]] = self.goalkeeper(field)
                    return
                if field.is_ball_in(collector):
                    targets = [
                        name
                        for name in ("P0", "F_far", "F_near")
                        if name in self.roles and self.roles[name] != collector.r_id
                    ]
                    self.pass_or_hold(field, actions, collector, targets)
                else:
                    target = self.attack_points["P0"] if "P0" in self.roles else field.enemy_goal.center
                    actions[collector.r_id] = self.collect_ball(field, collector, target)
                return

        sender_role = {"PASS_1": "P0", "PASS_2": "F_far", "SHOT": "F_near"}[self.attack_phase]
        sender_id = self.roles.get(sender_role)
        if sender_id is None:
            self.reset_attack()
            return
        sender = field.allies[sender_id]
        if not field.is_ball_in(sender):
            self.reset_attack()
            return
        if self.attack_phase == "SHOT":
            self.start_transfer(field, actions, sender, None, self.choose_shot_target(field), "RECOVER")
        else:
            targets = ["F_far", "F_near"] if self.attack_phase == "PASS_1" else ["F_near", "P0"]
            self.pass_or_hold(field, actions, sender, targets)

    def choose_collector(self, field: fld.Field) -> Optional[rbt.Robot]:
        candidates = field.active_allies(False)
        if not candidates:
            return None
        holder = field.robot_with_ball
        if holder is not None and holder in candidates:
            self.collector = holder.r_id
            return holder
        ball = field.ball.get_pos()
        nearest = min(candidates, key=lambda robot: aux.dist(robot.get_pos(), ball))
        previous = next((robot for robot in candidates if robot.r_id == self.collector), None)
        if previous is not None and aux.dist(previous.get_pos(), ball) <= aux.dist(nearest.get_pos(), ball) + COLLECTOR_HYST:
            nearest = previous
        self.collector = nearest.r_id
        return nearest

    def collect_ball(self, field: fld.Field, robot: rbt.Robot, target: aux.Point) -> Action:
        if field.is_ball_moves_to_point(robot.get_pos()):
            return StrategyActions.CatchBall()
        return Actions.BallGrab(aux.angle_to_point(field.ball.get_pos(), target))

    def pass_or_hold(self, field: fld.Field, actions: list[Optional[Action]], sender: rbt.Robot, targets: list[str]) -> None:
        ball = field.ball.get_pos()
        if self.setup_owner != sender.r_id:
            self.setup_owner = sender.r_id
            self.setup_start = self.clock()
        for role in targets:
            r_id = self.roles.get(role)
            if r_id is None or r_id == sender.r_id:
                continue
            receiver = field.allies[r_id]
            station = self.attack_points[role]
            if self.is_pass_blocked(field, ball, station) or self.is_pass_blocked(field, ball, receiver.get_pos()):
                continue
            ready = (
                aux.dist(receiver.get_pos(), station) < RECEIVER_READY_DIST
                and receiver.get_vel().mag() < 250
                and abs(aux.wind_down_angle(aux.angle_to_point(receiver.get_pos(), ball) - receiver.get_angle())) < 0.35
            )
            if ready:
                phase = {"P0": "PASS_1", "F_far": "PASS_2", "F_near": "SHOT"}[role]
                self.start_transfer(field, actions, sender, r_id, receiver.get_pos(), phase)
            else:
                if self.setup_start is not None and self.clock() - self.setup_start >= SETUP_TIMEOUT:
                    continue
                self.hold_start = None
                self.hold_owner = None
                actions[sender.r_id] = Actions.BallGrab(aux.angle_to_point(ball, station))
            return
        if self.hold_start is None or self.hold_owner != sender.r_id:
            self.hold_start = self.clock()
            self.hold_owner = sender.r_id
        if self.clock() - self.hold_start >= HOLD_TIME:
            self.start_transfer(field, actions, sender, None, self.choose_shot_target(field), "RECOVER")
        else:
            actions[sender.r_id] = Actions.BallGrab(aux.angle_to_point(ball, field.enemy_goal.center))

    def start_transfer(
        self,
        field: fld.Field,
        actions: list[Optional[Action]],
        sender: rbt.Robot,
        receiver: Optional[int],
        target: aux.Point,
        next_phase: str,
    ) -> None:
        self.hold_start = None
        self.hold_owner = None
        self.setup_start = None
        self.setup_owner = None
        self.transfer = Transfer(sender.r_id, receiver, target, next_phase, field.ball.get_pos(), self.clock())
        actions[sender.r_id] = KickActions.Straight(target, is_pass=receiver is not None)
        if receiver is not None:
            actions[receiver] = self.receive_ball(field, field.allies[receiver], target)

    def drive_transfer(self, field: fld.Field, actions: list[Optional[Action]]) -> bool:
        transfer = self.transfer
        if transfer is None:
            return False
        now = self.clock()
        sender = field.allies[transfer.sender]
        if transfer.receiver is not None and field.is_ball_in(field.allies[transfer.receiver]):
            self.attack_phase = transfer.next_phase
            self.transfer = None
            return False
        holder = field.robot_with_ball
        if holder is not None and (holder.color != field.ally_color or holder.r_id != transfer.sender):
            self.reset_attack()
            return False
        if transfer.released is None:
            if not field.is_ball_in(sender) and aux.dist(field.ball.get_pos(), transfer.origin) > 160:
                transfer.released = now
            elif now - transfer.started > KICK_TIMEOUT:
                self.reset_attack()
                return False
            else:
                actions[sender.r_id] = KickActions.Straight(transfer.target, is_pass=transfer.receiver is not None)
        if transfer.released is not None:
            if now - transfer.released > FLIGHT_TIMEOUT:
                self.reset_attack()
                return False
            # Sender keeps its formation assignment; only the receiver follows the ball.
            if transfer.receiver is None:
                if now - transfer.released > 1.0 and not field.is_ball_moves_to_enemy_goal():
                    self.reset_attack()
                return True
        if transfer.receiver is not None:
            actions[transfer.receiver] = self.receive_ball(field, field.allies[transfer.receiver], transfer.target)
        return True

    def receive_ball(self, field: fld.Field, robot: rbt.Robot, station: aux.Point) -> Action:
        ball = field.ball.get_pos()
        if field.robot_with_ball is None:
            if field.is_ball_moves_to_point(robot.get_pos()):
                return StrategyActions.CatchBall()
            if self.transfer is not None and self.transfer.released is not None and field.is_ball_moves():
                # Long legal passes may stop short: meet the reachable part of their path.
                return StrategyActions.CatchBall()
            if aux.dist(ball, robot.get_pos()) < 400 or (
                self.transfer is not None and self.transfer.released is not None and not field.is_ball_moves()
            ):
                return Actions.BallGrab(aux.angle_to_point(robot.get_pos(), ball))
        return Actions.GoToPoint(station, aux.angle_to_point(station, ball)).compose(Actions.SetDribblerSpeed())

    def run_defense(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        ball = field.ball.get_pos()
        gk = field.allies[field.gk_id]
        if gk.is_used():
            actions[gk.r_id] = self.goalkeeper(field)
        defenders = field.active_allies(False)
        holder = field.robot_with_ball
        enemy_has_ball = holder is not None and holder.color != field.ally_color
        ball_in_box = aux.is_point_inside_poly(ball, field.ally_goal.big_hull)
        if defenders and not enemy_has_ball and holder != gk and not ball_in_box:
            clearer = self.choose_collector(field)
            if clearer is not None:
                actions[clearer.r_id] = KickActions.Straight(self.choose_clear_target(field), is_pass=True)
                defenders = [robot for robot in defenders if robot.r_id != clearer.r_id]
        self.cover_pass_lines(field, defenders, actions)

    def goalkeeper(self, field: fld.Field) -> Action:
        gk = field.allies[field.gk_id]
        keeper_can_collect = (
            aux.is_point_inside_poly(field.ball.get_pos(), field.ally_goal.big_hull) and not field.is_ball_moves()
        )
        if field.is_ball_in(gk) or keeper_can_collect:
            return KickActions.Straight(self.choose_clear_target(field), is_pass=True)
        if field.is_ball_moves_to_goal():
            return StrategyActions.CatchBall()
        point = self.keeper_point(field)
        return Actions.GoToPoint(point, aux.angle_to_point(point, field.ball.get_pos()))

    def keeper_point(self, field: fld.Field) -> aux.Point:
        goal = field.ally_goal
        ball = field.ball.get_pos()
        x = goal.center.x + goal.eye_forw.x * const.GK_FORW
        inter = aux.get_line_intersection(goal.center, ball, aux.Point(x, -1), aux.Point(x, 1), "LL")
        return aux.Point(x, aux.minmax(inter.y if inter is not None else ball.y, goal.down.y, goal.up.y))

    def cover_pass_lines(self, field: fld.Field, defenders: list[rbt.Robot], actions: list[Optional[Action]]) -> None:
        if not defenders:
            return
        ball = field.ball.get_pos()
        enemies = field.active_enemies(True)
        points: list[aux.Point] = []
        if enemies:
            holder = field.robot_with_ball
            if holder is None or holder.color == field.ally_color:
                holder = fld.find_nearest_robot(ball, enemies)
            receivers = [enemy for enemy in field.active_enemies(False) if enemy != holder]
            receivers.sort(key=lambda enemy: aux.dist(enemy.get_pos(), field.ally_goal.center))
            points = [aux.lerp(holder.get_pos(), enemy.get_pos(), COVER_RATIO) for enemy in receivers]
        for index in range(len(defenders)):
            distance = min(700 + index * 400, max(0, aux.dist(ball, field.ally_goal.center) - 600))
            points.append(aux.point_on_line(ball, field.ally_goal.center, distance))
        points = [self.safe_point(field, point) for point in points[: len(defenders)]]
        assignment = min(
            permutations(range(len(points))),
            key=lambda order: sum(aux.dist(robot.get_pos(), points[index]) for robot, index in zip(defenders, order)),
        )
        for robot, index in zip(defenders, assignment):
            actions[robot.r_id] = Actions.GoToPoint(points[index], aux.angle_to_point(points[index], ball))

    def safe_point(self, field: fld.Field, point: aux.Point) -> aux.Point:
        point = aux.Point(
            aux.minmax(point.x, const.FIELD_DX - FIELD_MARGIN), aux.minmax(point.y, const.FIELD_DY - FIELD_MARGIN)
        )
        for goal in (field.ally_goal, field.enemy_goal):
            if aux.is_point_inside_poly(point, goal.big_hull):
                point = aux.Point(goal.frw.x + goal.eye_forw.x * (1.2 * const.ROBOT_R + 10), point.y)
        return point

    def choose_shot_target(self, field: fld.Field) -> aux.Point:
        goal = field.enemy_goal
        keeper = field.enemies[field.enemy_gk_id]
        reference = keeper.get_pos() if keeper.is_used() else field.ball.get_pos()
        post = max((goal.up, goal.down), key=lambda point: aux.dist(point, reference))
        return post + (goal.center - post).unity() * POST_MARGIN

    def choose_clear_target(self, field: fld.Field) -> aux.Point:
        ball = field.ball.get_pos()
        enemies = [robot.get_pos() for robot in field.active_enemies(True)]
        points = [field.ally_goal.eye_forw * depth + aux.Point(0, y) for depth in CLEAR_DEPTHS for y in CLEAR_YS]
        available = [point for point in points if not self.is_pass_blocked(field, ball, point)]
        return max(
            available,
            key=lambda point: min((aux.dist(point, enemy) for enemy in enemies), default=math.inf),
            default=field.enemy_goal.center,
        )

    def is_pass_blocked(self, field: fld.Field, src: aux.Point, dst: aux.Point) -> bool:
        return any(
            aux.dist_to_line(src, dst, enemy.get_pos(), "S") < PASS_BLOCK_DIST for enemy in field.active_enemies(True)
        )

    def draw_state(self, field: fld.Field) -> None:
        field.strategy_image.print(aux.Point(0, const.FIELD_DY - 100), f"{self.mode} / {self.attack_phase}")
        if self.mode == "ATTACK":
            for role, r_id in self.roles.items():
                point = self.attack_points[role]
                field.strategy_image.draw_circle(point, (255, 128, 0), 60)
                field.strategy_image.print(point + aux.Point(0, 100), f"{role}: {r_id}")
        if self.transfer is not None:
            field.strategy_image.draw_line(field.ball.get_pos(), self.transfer.target, (0, 255, 128), 2)
        field.strategy_image.send_telemetry(
            "STRATEGY",
            f"{self.mode} {self.attack_phase}; roles={self.roles}; "
            f"last_touch={self.last_touch}; transfer={self.transfer}",
        )
