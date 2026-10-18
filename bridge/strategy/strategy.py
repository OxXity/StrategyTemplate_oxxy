"""Possession-based attack with a moving triangle and arc-based goalkeeping."""

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
from bridge.router.actions.dumb_actions import DumbActions

HALF_HYST = 200
PASS_BLOCK_DIST = 2 * const.ROBOT_R + const.BALL_R
POST_MARGIN = 100
MAX_ATTACK_PASSES = 5
TURN_WITH_BALL_MAX_SPEED = 250
TURN_WITH_BALL_MAX_ANGULAR_SPEED = 3.0
KICK_ALIGNMENT_MAX_SPEED = 500
FIELD_MARGIN = 2 * const.ROBOT_R
COLLECTOR_HYST = 250
RECEIVER_READY_DIST = 180
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

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.clock = clock
        self.rng = rng if rng is not None else random.Random()
        self.we_active = False
        self.active_team_initialized = False
        self.mode = "DEFENSE"
        self.attack_phase = "RECOVER"
        self.roles: dict[str, int] = {}
        self.attack_points: dict[str, aux.Point] = {}
        self.last_touch: Optional[const.Color] = None
        self.transfer: Optional[Transfer] = None
        self.attack_passes = 0
        self.collector: Optional[int] = None
        self.roster: tuple[int, ...] = ()
        self.triangle_side = 1
        self.restart_key: Optional[tuple] = None
        self.restart_kicker: Optional[int] = None

    def process(self, field: fld.Field) -> list[Optional[Action]]:
        """Build commands for the current referee state."""
        # The referee may clear active_team during a kickoff or penalty start.
        # Keep the owner learned during preparation until play leaves that state.
        if (
            field.game_state not in (GameStates.KICKOFF, GameStates.PENALTY)
            or not self.active_team_initialized
        ):
            self.we_active = field.active_team in (const.Color.ALL, field.ally_color)
            self.active_team_initialized = True

        if field.game_state == GameStates.HALT:
            self.reset_attack()
            self.restart_key = None
            self.restart_kicker = None
            self.mode = "DEFENSE"
            self.last_touch = None
            self.draw_state(field)
            return [Actions.Stop()] * const.TEAM_ROBOTS_MAX_COUNT

        actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
        for robot in field.active_allies(True):
            actions[robot.r_id] = Actions.Stop()
        match field.game_state:
            case GameStates.RUN:
                self.restart_key = None
                self.restart_kicker = None
                self.run(field, actions)
            case GameStates.TIMEOUT:
                self.reset_attack()
                self.restart_key = None
                self.restart_kicker = None
                self.mode = "DEFENSE"
                self.last_touch = None
            case GameStates.PREPARE_KICKOFF | GameStates.KICKOFF:
                self.run_restart(field, actions)
            case GameStates.PREPARE_PENALTY | GameStates.PENALTY:
                self.run_restart(field, actions)
            case GameStates.FREE_KICK:
                self.run_restart(field, actions)
            case GameStates.STOP:
                # Normal movement is legal; the router keeps players clear of the ball.
                self.run(field, actions)
            case GameStates.BALL_PLACEMENT:
                self.reset_attack()
                self.restart_key = None
                self.restart_kicker = None
                self.mode = "DEFENSE"
                self.last_touch = None
                self.cover_pass_lines(field, field.active_allies(False), actions)
            case GameStates.DEBUG:
                pass
            case _:
                self.reset_attack()
                self.restart_key = None
                self.restart_kicker = None
                self.mode = "DEFENSE"
                self.last_touch = None
        self.draw_state(field)
        return actions

    def run_restart(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        """Prepare a restart, then let only its designated team take the kick."""
        # ALL cannot identify a restart owner, unlike ordinary RUN play.
        ours = self.we_active
        key = (field.game_state, field.active_team, field.referee_command_counter)
        players = field.active_allies(False)
        kickers = players or field.active_allies(True)
        if key != self.restart_key or (tuple(sorted(robot.r_id for robot in field.active_allies(True))) != self.roster):
            self.reset_attack()
            self.generate_attack_points(field)
            self.restart_key = key
            self.restart_kicker = (
                min(kickers, key=lambda robot: aux.dist(robot.get_pos(), field.ball.get_pos())).r_id
                if ours and kickers
                else None
            )
        self.mode = "ATTACK" if ours else "DEFENSE"
        self.attack_phase = field.game_state.name
        ball = field.ball.get_pos()
        forward = field.ally_goal.eye_forw
        lateral = aux.Point(-forward.y, forward.x)
        kickoff = field.game_state in (GameStates.PREPARE_KICKOFF, GameStates.KICKOFF)
        penalty = field.game_state in (GameStates.PREPARE_PENALTY, GameStates.PENALTY)
        ready_to_kick = field.game_state in (GameStates.KICKOFF, GameStates.FREE_KICK, GameStates.PENALTY)
        stations: dict[int, aux.Point] = {}
        roles_by_id = {r_id: role for role, r_id in self.roles.items()}
        for index, robot in enumerate(field.active_allies(True)):
            if robot.r_id == self.restart_kicker:
                continue
            if robot.r_id == field.gk_id:
                point = self.keeper_point(field)
                if penalty and not ours:
                    point = aux.Point(field.ally_goal.center.x, point.y)
            else:
                side = 1 if index % 2 else -1
                if kickoff:
                    point = -forward * 300 + lateral * (side * 850)
                else:
                    depth_offset = 700 if ours else -1000
                    if penalty:
                        depth_offset = -900 if ours else 900
                    point = ball + forward * depth_offset + lateral * (side * 800)
                point = self.safe_point(field, point)
            stations[robot.r_id] = point
            if robot.r_id in roles_by_id:
                self.attack_points[roles_by_id[robot.r_id]] = point
            actions[robot.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ball))
        if not ours:
            keeper = field.allies[field.gk_id]
            if penalty and ready_to_kick and keeper.is_used() and field.is_ball_moves_to_goal():
                actions[keeper.r_id] = StrategyActions.CatchBall()
            return
        if self.restart_kicker is None:
            return
        kicker = field.allies[self.restart_kicker]
        if ready_to_kick and self.drive_transfer(field, actions):
            return
        receivers = sorted(
            (field.allies[r_id] for r_id in stations),
            key=lambda robot: aux.dist(robot.get_pos(), field.enemy_goal.center),
        )
        target = (
            self.choose_shot_target(field)
            if penalty
            else (stations[receivers[0].r_id] if receivers else field.enemy_goal.center)
        )
        if not ready_to_kick:
            point = self.safe_point(field, ball - forward * (const.KEEP_BALL_DIST + 50))
            actions[kicker.r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, target))
            return
        if penalty:
            self.start_transfer(field, actions, kicker, None, target, "RECOVER")
            return
        # Both direct and indirect free kicks start with a pass. This also
        # prevents an indirect free kick from being sent straight into goal.
        for receiver in receivers:
            if (
                aux.dist(receiver.get_pos(), stations[receiver.r_id]) < RECEIVER_READY_DIST
                and receiver.get_vel().mag() < 250
                and abs(aux.wind_down_angle(aux.angle_to_point(receiver.get_pos(), ball) - receiver.get_angle())) < 0.35
                and not self.is_pass_blocked(field, ball, receiver.get_pos())
            ):
                self.start_transfer(field, actions, kicker, receiver.r_id, receiver.get_pos(), "POSSESSION")
                return
        actions[kicker.r_id] = self.turn_with_ball(field, kicker, target)

    def reset_attack(self) -> None:
        self.attack_phase = "RECOVER"
        self.transfer = None
        self.attack_passes = 0
        self.collector = None

    def run(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        holder = field.robot_with_ball
        if holder is not None:
            self.last_touch = holder.color
        self.update_mode(field)
        if self.mode == "ATTACK":
            self.run_attack(field, actions)
        else:
            self.run_defense(field, actions)

    def attack_depth(self, field: fld.Field, point: aux.Point) -> float:
        return aux.scal_mult(point, field.ally_goal.eye_forw)

    def update_mode(self, field: fld.Field) -> None:
        ball = field.ball.get_pos()
        holder = field.robot_with_ball
        enemy_has_ball = holder is not None and holder.color != field.ally_color
        depth = self.attack_depth(field, ball)
        if enemy_has_ball:
            if self.mode != "DEFENSE":
                self.reset_attack()
            self.mode = "DEFENSE"
        elif self.transfer is not None and self.mode == "ATTACK":
            # A tracked pass to a teammate can travel toward our goal because P0
            # is a deliberate outlet on our half. Preserve that transfer; only
            # an unassigned shot/clearance toward our own goal is a threat.
            if self.transfer.receiver is None and field.is_ball_moves_to_goal():
                self.reset_attack()
                self.mode = "DEFENSE"
            return
        elif holder is not None and holder.color == field.ally_color:
            if self.mode != "ATTACK":
                self.reset_attack()
                self.generate_attack_points(field)
            self.mode = "ATTACK"
        elif depth < -HALF_HYST:
            if self.mode != "DEFENSE":
                self.reset_attack()
            self.mode = "DEFENSE"
        elif depth > HALF_HYST and self.mode != "ATTACK":
            self.mode = "ATTACK"
            self.reset_attack()
            self.generate_attack_points(field)

    def generate_attack_points(self, field: fld.Field) -> None:
        """Initialize a triangle which is refreshed smoothly from the live ball location."""
        self.roster = tuple(sorted(robot.r_id for robot in field.active_allies(True)))
        forward = field.ally_goal.eye_forw
        lateral = aux.Point(-forward.y, forward.x)
        self.triangle_side = self.rng.choice((-1, 1))
        self.attack_points = {}
        self.update_attack_points(field, smooth=False)
        self.roles = {}
        if field.allies[field.gk_id].is_used():
            self.roles["P0"] = field.gk_id
        forwards = field.active_allies(False)
        names = ("F_near", "F_far")[: len(forwards)]
        if names:
            selected = min(
                permutations(forwards, len(names)),
                key=lambda robots: sum(
                    aux.dist(robot.get_pos(), self.attack_points[name])
                    for name, robot in zip(names, robots)
                ),
            )
            self.roles.update(
                {name: robot.r_id for name, robot in zip(names, selected)}
            )

    def update_attack_points(self, field: fld.Field, *, smooth: bool = True) -> None:
        """Move three target vertices with the ball while keeping P0 behind halfway."""
        forward = field.ally_goal.eye_forw
        lateral = aux.Point(-forward.y, forward.x)
        ball = field.ball.get_pos()
        depth = self.attack_depth(field, ball)
        shift = aux.minmax(ball.y * 0.35, -650, 650)
        raw = {
            "P0": -forward * 350 + lateral * aux.minmax(ball.y * 0.2, -220, 220),
            "F_near": forward * aux.minmax(depth + 1050, 450, 1800)
            + lateral * (shift + self.triangle_side * 500),
            "F_far": forward * aux.minmax(depth + 500, 100, 1450)
            + lateral * (shift - self.triangle_side * 550),
        }
        for role, target in raw.items():
            target = self.safe_point(field, target)
            if smooth and role in self.attack_points:
                old = self.attack_points[role]
                target = old * 0.82 + target * 0.18
            self.attack_points[role] = target

    def run_attack(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        roster = tuple(sorted(robot.r_id for robot in field.active_allies(True)))
        if roster != self.roster:
            self.reset_attack()
            self.generate_attack_points(field)
        self.update_attack_points(field)
        ball = field.ball.get_pos()
        for role, r_id in self.roles.items():
            point = self.attack_points[role]
            actions[r_id] = Actions.GoToPoint(point, aux.angle_to_point(point, ball))
        holder = field.robot_with_ball
        if self.drive_transfer(field, actions):
            return
        candidates = field.active_allies(True)
        if holder is None or holder.color != field.ally_color:
            collector = self.choose_collector(field)
            if collector is not None:
                actions[collector.r_id] = self.collect_ball(
                    field, collector, field.enemy_goal.center
                )
            return

        forwards = field.active_allies(False)
        if holder.r_id == field.gk_id:
            # P0 is a distributor: give it a nearby forward outlet, then keep its
            # own-half position instead of joining the attack near the goal.
            target_ids = [
                robot.r_id
                for robot in sorted(
                    forwards,
                    key=lambda robot: aux.dist(
                        robot.get_pos(), field.enemy_goal.center
                    ),
                )
            ]
        else:
            ordered = sorted(
                candidates,
                key=lambda robot: aux.dist(robot.get_pos(), field.enemy_goal.center),
            )
            closest_to_goal = ordered[0] if ordered else holder
            close_enough_to_shoot = self.attack_depth(field, ball) > 700
            shot_target = self.choose_shot_target(field)
            if (
                holder == closest_to_goal
                and close_enough_to_shoot
                and not self.is_pass_blocked(field, ball, shot_target)
            ):
                actions[holder.r_id] = self.kick_with_alignment(field, shot_target)
                self.attack_phase = "POSSESSION"
                return
            target_ids = [robot.r_id for robot in ordered if robot.r_id != holder.r_id]

        roles_by_id = {r_id: role for role, r_id in self.roles.items()}
        target_roles = [roles_by_id[r_id] for r_id in target_ids if r_id in roles_by_id]
        if target_roles:
            self.pass_or_hold(field, actions, holder, target_roles)
            return
        shot_target = self.choose_shot_target(field)
        if (
            holder.r_id != self.roles.get("P0")
            and self.attack_depth(field, ball) > 700
            and not self.is_pass_blocked(field, ball, shot_target)
        ):
            actions[holder.r_id] = self.kick_with_alignment(field, shot_target)
        else:
            actions[holder.r_id] = self.turn_with_ball(
                field, holder, field.enemy_goal.center
            )

    @staticmethod
    def turn_with_ball(field: fld.Field, robot: rbt.Robot, target: aux.Point) -> Action:
        """Turn while keeping the ball centered in the dribbler with low translation speed."""
        angle = aux.angle_to_point(field.ball.get_pos(), target)
        return Actions.BallGrab(angle, max_angular_speed=TURN_WITH_BALL_MAX_ANGULAR_SPEED).compose(
            DumbActions.LimitSpeed(TURN_WITH_BALL_MAX_SPEED)
        )

    @staticmethod
    def kick_with_alignment(
        field: fld.Field, target: aux.Point, *, is_pass: bool = False
    ) -> Action:
        """Align with the ball under controlled speed, then kick when contact is valid."""
        return KickActions.Straight(
            target, is_pass=is_pass, max_angular_speed=TURN_WITH_BALL_MAX_ANGULAR_SPEED
        ).compose(
            DumbActions.LimitSpeed(KICK_ALIGNMENT_MAX_SPEED)
        )

    def choose_collector(self, field: fld.Field) -> Optional[rbt.Robot]:
        ball = field.ball.get_pos()
        if aux.is_point_inside_poly(ball, field.ally_goal.big_hull):
            keeper = field.allies[field.gk_id]
            return keeper if keeper.is_used() else None
        candidates = field.active_allies(False)
        if not candidates:
            return None
        holder = field.robot_with_ball
        if holder is not None and holder in candidates:
            self.collector = holder.r_id
            return holder
        nearest = min(candidates, key=lambda robot: aux.dist(robot.get_pos(), ball))
        previous = next(
            (robot for robot in candidates if robot.r_id == self.collector), None
        )
        if (
            previous is not None
            and aux.dist(previous.get_pos(), ball)
            <= aux.dist(nearest.get_pos(), ball) + COLLECTOR_HYST
        ):
            nearest = previous
        self.collector = nearest.r_id
        return nearest

    def collect_ball(
        self, field: fld.Field, robot: rbt.Robot, target: aux.Point
    ) -> Action:
        if field.is_ball_moves_to_point(robot.get_pos()):
            return StrategyActions.CatchBall()
        return Actions.BallGrab(aux.angle_to_point(field.ball.get_pos(), target))

    def pass_or_hold(
        self,
        field: fld.Field,
        actions: list[Optional[Action]],
        sender: rbt.Robot,
        targets: list[str],
    ) -> None:
        ball = field.ball.get_pos()
        if self.attack_passes >= MAX_ATTACK_PASSES:
            # End the possession with a shot or a safe forward clearance; do not
            # let a failed/blocked setup turn into an endless sixth pass.
            target = (
                self.choose_clear_target(field)
                if sender.r_id == field.gk_id
                else self.choose_shot_target(field)
            )
            if sender.r_id != field.gk_id and self.is_pass_blocked(field, ball, target):
                target = self.choose_clear_target(field)
            self.start_transfer(field, actions, sender, None, target, "RECOVER")
            return
        for role in targets:
            r_id = self.roles.get(role)
            if r_id is None or r_id == sender.r_id:
                continue
            receiver = field.allies[r_id]
            station = self.attack_points[role]
            if self.is_pass_blocked(field, ball, station) or self.is_pass_blocked(
                field, ball, receiver.get_pos()
            ):
                continue
            ready = (
                aux.dist(receiver.get_pos(), station) < RECEIVER_READY_DIST
                and receiver.get_vel().mag() < 250
                and abs(
                    aux.wind_down_angle(
                        aux.angle_to_point(receiver.get_pos(), ball)
                        - receiver.get_angle()
                    )
                )
                < 0.35
            )
            if ready:
                self.start_transfer(
                    field, actions, sender, r_id, receiver.get_pos(), "POSSESSION"
                )
                return
        waiting_role = next(
            (
                role
                for role in targets
                if self.roles.get(role) is not None
                and self.roles[role] != sender.r_id
                and not self.is_pass_blocked(field, ball, self.attack_points[role])
            ),
            None,
        )
        if waiting_role is None:
            lateral = aux.rotate(field.ally_goal.eye_forw, math.pi / 2)
            for role in targets:
                r_id = self.roles.get(role)
                if r_id is None or r_id == sender.r_id:
                    continue
                base = self.attack_points[role]
                alternatives = [
                    base + lateral * offset
                    for offset in (300, -300, 600, -600, 900, -900)
                ]
                station = next(
                    (
                        self.safe_point(field, point)
                        for point in alternatives
                        if not self.is_pass_blocked(
                            field, ball, self.safe_point(field, point)
                        )
                    ),
                    None,
                )
                if station is not None:
                    self.attack_points[role] = station
                    actions[r_id] = Actions.GoToPoint(
                        station, aux.angle_to_point(station, ball)
                    )
                    waiting_role = role
                    break
        if waiting_role is not None:
            actions[sender.r_id] = self.turn_with_ball(
                field, sender, self.attack_points[waiting_role]
            )
            return
        # No safe shot or pass yet: retain possession in place while the other
        # robots continue moving toward their open triangle stations.
        actions[sender.r_id] = self.turn_with_ball(
            field, sender, field.enemy_goal.center
        )

    def start_transfer(
        self,
        field: fld.Field,
        actions: list[Optional[Action]],
        sender: rbt.Robot,
        receiver: Optional[int],
        target: aux.Point,
        next_phase: str,
    ) -> None:
        self.transfer = Transfer(
            sender.r_id,
            receiver,
            target,
            next_phase,
            field.ball.get_pos(),
            self.clock(),
        )
        if receiver is not None:
            self.attack_passes += 1
        actions[sender.r_id] = self.kick_with_alignment(
            field, target, is_pass=receiver is not None
        )
        if receiver is not None:
            actions[receiver] = self.receive_ball(field, field.allies[receiver], target)

    def drive_transfer(self, field: fld.Field, actions: list[Optional[Action]]) -> bool:
        transfer = self.transfer
        if transfer is None:
            return False
        now = self.clock()
        sender = field.allies[transfer.sender]
        if transfer.receiver is not None and field.is_ball_in(
            field.allies[transfer.receiver]
        ):
            self.attack_phase = transfer.next_phase
            self.transfer = None
            return False
        holder = field.robot_with_ball
        if holder is not None and (
            holder.color != field.ally_color or holder.r_id != transfer.sender
        ):
            self.reset_attack()
            return False
        if transfer.released is None:
            if (
                not field.is_ball_in(sender)
                and aux.dist(field.ball.get_pos(), transfer.origin) > 160
            ):
                transfer.released = now
            elif now - transfer.started > KICK_TIMEOUT:
                self.reset_attack()
                return False
            else:
                actions[sender.r_id] = self.kick_with_alignment(
                    field, transfer.target, is_pass=transfer.receiver is not None
                )
        if transfer.released is not None:
            if now - transfer.released > FLIGHT_TIMEOUT:
                self.reset_attack()
                return False
            # Sender keeps its formation assignment; only the receiver follows the ball.
            if transfer.receiver is None:
                if (
                    now - transfer.released > 1.0
                    and not field.is_ball_moves_to_enemy_goal()
                ):
                    self.reset_attack()
                return True
        if transfer.receiver is not None:
            actions[transfer.receiver] = self.receive_ball(
                field, field.allies[transfer.receiver], transfer.target
            )
        return True

    def receive_ball(
        self, field: fld.Field, robot: rbt.Robot, station: aux.Point
    ) -> Action:
        ball = field.ball.get_pos()
        if field.robot_with_ball is None:
            if field.is_ball_moves_to_point(robot.get_pos()):
                return StrategyActions.CatchBall()
            if (
                self.transfer is not None
                and self.transfer.released is not None
                and field.is_ball_moves()
            ):
                # Long legal passes may stop short: meet the reachable part of their path.
                return StrategyActions.CatchBall()
            if aux.dist(ball, robot.get_pos()) < 400 or (
                self.transfer is not None
                and self.transfer.released is not None
                and not field.is_ball_moves()
            ):
                return Actions.BallGrab(aux.angle_to_point(robot.get_pos(), ball))
        return Actions.GoToPoint(station, aux.angle_to_point(station, ball)).compose(
            Actions.SetDribblerSpeed()
        )

    def run_defense(self, field: fld.Field, actions: list[Optional[Action]]) -> None:
        gk = field.allies[field.gk_id]
        if gk.is_used():
            actions[gk.r_id] = self.goalkeeper(field)
        forwards = field.active_allies(False)
        holder = field.robot_with_ball
        enemy_has_ball = holder is not None and holder.color != field.ally_color
        ball = field.ball.get_pos()
        if enemy_has_ball:
            target_robots = [holder]
            receivers = [
                robot for robot in field.active_enemies(False) if robot != holder
            ]
            receivers.sort(
                key=lambda robot: aux.dist(robot.get_pos(), field.ally_goal.center)
            )
            if receivers:
                target_robots.extend(receivers)
        else:
            opponents = field.active_enemies(False)
            target_robots = sorted(
                opponents, key=lambda robot: aux.dist(robot.get_pos(), ball)
            )
            if holder is None:
                collector = self.choose_collector(field)
                if collector is not None and not aux.is_point_inside_poly(
                    ball, field.ally_goal.big_hull
                ):
                    actions[collector.r_id] = self.collect_ball(
                        field, collector, field.enemy_goal.center
                    )
                    forwards = [
                        robot for robot in forwards if robot.r_id != collector.r_id
                    ]
        press_points = []
        for enemy in target_robots:
            direction = (field.ally_goal.center - enemy.get_pos()).unity()
            point = enemy.get_pos() + direction * (2 * const.ROBOT_R + 80)
            press_points.append(self.safe_point(field, point))
        self.assign_defenders(field, forwards, press_points, actions)

    def goalkeeper(self, field: fld.Field) -> Action:
        gk = field.allies[field.gk_id]
        keeper_can_collect = (
            aux.is_point_inside_poly(field.ball.get_pos(), field.ally_goal.big_hull)
            and not field.is_ball_moves()
        )
        if field.is_ball_in(gk) or keeper_can_collect:
            return KickActions.Straight(self.choose_clear_target(field), is_pass=True)
        if field.is_ball_moves_to_goal():
            return StrategyActions.CatchBall()
        point = self.keeper_point(field)
        return Actions.GoToPoint(point, aux.angle_to_point(point, field.ball.get_pos()))

    def keeper_point(self, field: fld.Field) -> aux.Point:
        """Slide along an arc in front of the posts to cover shots to either corner."""
        goal = field.ally_goal
        ball = field.ball.get_pos()
        forward = goal.eye_forw
        lateral = aux.Point(-forward.y, forward.x)
        radius = max(const.GK_FORW, const.GOAL_DY / 2 + const.ROBOT_R + 20)
        max_angle = math.atan2(const.GOAL_DY / 2, radius)
        relative = ball - goal.center
        lateral_distance = aux.scal_mult(relative, lateral)
        depth = aux.scal_mult(relative, forward)
        if field.is_ball_moves_to_goal() and abs(field.ball.get_vel().x) > 1e-6:
            time_to_goal = (goal.center.x - ball.x) / field.ball.get_vel().x
            if time_to_goal > 0:
                projected_y = ball.y + field.ball.get_vel().y * time_to_goal
                projected = aux.Point(goal.center.x, projected_y) - goal.center
                lateral_distance = aux.scal_mult(projected, lateral)
        angle = aux.minmax(
            math.atan2(lateral_distance, max(depth, 0)), -max_angle, max_angle
        )
        return (
            goal.center
            + forward * (radius * math.cos(angle))
            + lateral * (radius * math.sin(angle))
        )

    def assign_defenders(
        self,
        field: fld.Field,
        defenders: list[rbt.Robot],
        targets: list[aux.Point],
        actions: list[Optional[Action]],
    ) -> None:
        """Assign the closest defenders to opponent pressure/cover points."""
        if not defenders:
            return
        if not targets:
            ball = field.ball.get_pos()
            targets = [
                self.safe_point(
                    field, ball + (field.ally_goal.center - ball).unity() * dist
                )
                for dist in (450, 850)
            ]
        while len(targets) < len(defenders):
            distance = 650 + 350 * len(targets)
            targets.append(
                self.safe_point(
                    field,
                    aux.point_on_line(
                        field.ball.get_pos(), field.ally_goal.center, distance
                    ),
                )
            )
        targets = targets[: len(defenders)]
        assignment = min(
            permutations(range(len(targets)), len(defenders)),
            key=lambda order: sum(
                aux.dist(robot.get_pos(), targets[index])
                for robot, index in zip(defenders, order)
            ),
        )
        ball = field.ball.get_pos()
        for robot, index in zip(defenders, assignment):
            point = targets[index]
            actions[robot.r_id] = Actions.GoToPoint(
                point, aux.angle_to_point(point, ball)
            )

    def cover_pass_lines(
        self,
        field: fld.Field,
        defenders: list[rbt.Robot],
        actions: list[Optional[Action]],
    ) -> None:
        if not defenders:
            return
        ball = field.ball.get_pos()
        enemies = field.active_enemies(True)
        points: list[aux.Point] = []
        if enemies:
            holder = field.robot_with_ball
            if holder is None or holder.color == field.ally_color:
                holder = fld.find_nearest_robot(ball, enemies)
            receivers = [
                enemy for enemy in field.active_enemies(False) if enemy != holder
            ]
            receivers.sort(
                key=lambda enemy: aux.dist(enemy.get_pos(), field.ally_goal.center)
            )
            points = [
                aux.lerp(holder.get_pos(), enemy.get_pos(), COVER_RATIO)
                for enemy in receivers
            ]
        for index in range(len(defenders)):
            distance = min(
                700 + index * 400, max(0, aux.dist(ball, field.ally_goal.center) - 600)
            )
            points.append(aux.point_on_line(ball, field.ally_goal.center, distance))
        points = [self.safe_point(field, point) for point in points[: len(defenders)]]
        assignment = min(
            permutations(range(len(points))),
            key=lambda order: sum(
                aux.dist(robot.get_pos(), points[index])
                for robot, index in zip(defenders, order)
            ),
        )
        for robot, index in zip(defenders, assignment):
            actions[robot.r_id] = Actions.GoToPoint(
                points[index], aux.angle_to_point(points[index], ball)
            )

    def safe_point(self, field: fld.Field, point: aux.Point) -> aux.Point:
        point = aux.Point(
            aux.minmax(point.x, const.FIELD_DX - FIELD_MARGIN),
            aux.minmax(point.y, const.FIELD_DY - FIELD_MARGIN),
        )
        for goal in (field.ally_goal, field.enemy_goal):
            if aux.is_point_inside_poly(point, goal.big_hull):
                point = aux.Point(
                    goal.frw.x + goal.eye_forw.x * (1.2 * const.ROBOT_R + 10), point.y
                )
        return point

    def choose_shot_target(self, field: fld.Field) -> aux.Point:
        goal = field.enemy_goal
        low_y = min(goal.up.y, goal.down.y) + POST_MARGIN
        high_y = max(goal.up.y, goal.down.y) - POST_MARGIN
        # Evaluate the full mouth. A blocked far corner must not hide a clear
        # central lane or the opposite corner.
        targets = [
            aux.Point(goal.center.x, low_y + (high_y - low_y) * ratio)
            for ratio in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        enemies = field.active_enemies(True)
        ball = field.ball.get_pos()

        def clearance(target: aux.Point) -> float:
            return min(
                (
                    aux.dist_to_line(ball, target, enemy.get_pos(), "S")
                    for enemy in enemies
                ),
                default=math.inf,
            )

        # Use the open center first, then the widest remaining open lane.
        open_targets = [
            target for target in targets if not self.is_pass_blocked(field, ball, target)
        ]
        if targets[2] in open_targets:
            return targets[2]
        candidates = open_targets or targets
        center_y = goal.center.y
        return max(
            candidates,
            key=lambda target: (
                clearance(target),
                -abs(target.y - center_y),
            ),
        )

    def choose_clear_target(self, field: fld.Field) -> aux.Point:
        ball = field.ball.get_pos()
        enemies = [robot.get_pos() for robot in field.active_enemies(True)]
        points = [
            field.ally_goal.eye_forw * depth + aux.Point(0, y)
            for depth in CLEAR_DEPTHS
            for y in CLEAR_YS
        ]
        available = [
            point for point in points if not self.is_pass_blocked(field, ball, point)
        ]
        return max(
            available,
            key=lambda point: min(
                (aux.dist(point, enemy) for enemy in enemies), default=math.inf
            ),
            default=field.enemy_goal.center,
        )

    def is_pass_blocked(self, field: fld.Field, src: aux.Point, dst: aux.Point) -> bool:
        return any(
            aux.dist_to_line(src, dst, enemy.get_pos(), "S") < PASS_BLOCK_DIST
            for enemy in field.active_enemies(True)
        )

    def draw_state(self, field: fld.Field) -> None:
        field.strategy_image.print(
            aux.Point(0, const.FIELD_DY - 100), f"{self.mode} / {self.attack_phase}"
        )
        shot_target = self.choose_shot_target(field) if self.mode == "ATTACK" else None
        shot_open = shot_target is not None and not self.is_pass_blocked(
            field, field.ball.get_pos(), shot_target
        )
        if self.mode == "ATTACK":
            for role, r_id in self.roles.items():
                point = self.attack_points[role]
                field.strategy_image.draw_circle(point, (255, 128, 0), 60)
                field.strategy_image.print(point + aux.Point(0, 100), f"{role}: {r_id}")
        if self.transfer is not None:
            field.strategy_image.draw_line(
                field.ball.get_pos(), self.transfer.target, (0, 255, 128), 2
            )
        field.strategy_image.send_telemetry(
            "STRATEGY",
            f"game_state={field.game_state.name}; active_team={field.active_team.name}; "
            f"simulator={const.IS_SIMULATOR_USED}; {self.mode} {self.attack_phase}; roles={self.roles}; "
            f"holder={field.robot_with_ball.r_id if field.robot_with_ball else None}; "
            f"contacts={[r.r_id for r in field.active_allies(True) if field._is_ball_in(r)]}; "
            f"shot={shot_target}; shot_open={shot_open}; "
            f"last_touch={self.last_touch}; passes={self.attack_passes}/{MAX_ATTACK_PASSES}; transfer={self.transfer}",
        )
