"""
Межтактовая память стратегии (ТЗ, раздел 5.4)

Живёт в объекте Strategy (процесс стратегии). Все отметки времени - time.time() (R-5.8).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.strategy import params

if TYPE_CHECKING:
    from bridge.strategy.kick_model import KickModel
    from bridge.strategy.world import WorldModel


class Possession(Enum):
    """Владение мячом"""

    OURS = 0
    THEIRS = 1
    LOOSE = 2
    CONTESTED = 3


class Role(Enum):
    """Роли роботов"""

    GOALKEEPER = 0
    ATTACKER = 1
    SUPPORT = 2
    DEFENDER = 3


ROLE_LABELS = {
    Role.GOALKEEPER: "GK",
    Role.ATTACKER: "ATT",
    Role.SUPPORT: "SUP",
    Role.DEFENDER: "DEF",
}

# Игра остановлена: снимается запрет двойного касания и сбрасываются розыгрыши
STOPPED_STATES = (const.State.STOP, const.State.HALT, const.State.TIMEOUT, const.State.BALL_PLACEMENT)


@dataclass
class DribbleTrack:
    """Трекер ведения (G-6): смещение мяча считается от позиции в момент захвата"""

    start_ball_pos: aux.Point
    start_time: float
    last_seen: float


@dataclass
class PassInfo:
    """Пас в полёте"""

    from_id: int
    to_id: int
    target: aux.Point
    start_time: float
    expected_arrival: float


@dataclass
class Decision:
    """Решение Нападающего / розыгрыша, фиксируемое на время"""

    kind: str
    target: Optional[aux.Point]
    since: float
    utility: float = 0.0
    receiver_id: Optional[int] = None


@dataclass
class KickLog:
    """Измерение скорости мяча после нашего удара (G-7)"""

    robot_id: int
    voltage: int
    start_time: float
    max_speed: float = 0.0
    done: bool = False


@dataclass
class Memory:  # pylint: disable=too-many-instance-attributes
    """Память стратегии"""

    # Состояние игры
    state: const.State = const.State.HALT
    prev_state: const.State = const.State.HALT
    state_enter_time: float = 0.0
    we_active: bool = False

    # Роли и владение
    roles: dict[int, Role] = field(default_factory=dict)
    role_since: dict[int, float] = field(default_factory=dict)
    possession: Possession = Possession.LOOSE
    possession_since: float = 0.0
    possession_candidate: Possession = Possession.LOOSE
    possession_candidate_since: float = 0.0
    last_owner: Optional[tuple[const.Color, int]] = None
    last_touch: Optional[tuple[const.Color, int, float]] = None

    # Мяч и удары
    dribble: dict[int, DribbleTrack] = field(default_factory=dict)
    pass_in_flight: Optional[PassInfo] = None
    pending_pass: Optional[PassInfo] = None
    kick_log: Optional[KickLog] = None
    last_kick_max_speed: float = 0.0
    kick_voltage_issued: dict[int, int] = field(default_factory=dict)
    prev_ball_speed: float = 0.0

    # Стандарты и пенальти
    double_touch_banned_id: Optional[int] = None
    set_piece_kicker_id: Optional[int] = None
    set_piece_ball_start: Optional[aux.Point] = None
    set_piece_armed: bool = False
    set_piece_kicker_near_time: float = -1.0
    set_piece_last_time: float = 0.0
    set_piece_decision: Optional[Decision] = None
    penalty_ball_start: Optional[aux.Point] = None
    penalty_ball_in_play: bool = False
    penalty_tap_angle: Optional[float] = None
    placement_done_time: Optional[float] = None

    # Таймеры и прочее
    ball_in_our_area_since: Optional[float] = None
    attacker_decision: Optional[Decision] = None
    gk_decision: Optional[Decision] = None
    support_target: Optional[aux.Point] = None
    support_target_score: float = -1e9
    support_update_time: float = 0.0
    support_mode: str = "COVER"
    support_mode_since: float = 0.0
    push_contact_since: dict[int, float] = field(default_factory=dict)
    push_retreat_until: dict[int, float] = field(default_factory=dict)
    push_side: dict[int, float] = field(default_factory=dict)
    last_exception_print_time: float = 0.0
    last_telemetry_time: float = 0.0
    debug: dict[str, float] = field(default_factory=dict)

    # Метрики (15.3)
    metrics: dict[str, float] = field(default_factory=dict)

    def state_time(self, now: float) -> float:
        """Сколько длится текущее состояние"""
        return now - self.state_enter_time

    def update_state(self, state: const.State, now: float) -> None:
        """Таймеры состояний"""
        if state != self.state:
            self.prev_state = self.state
            self.state = state
            self.state_enter_time = now
            self.set_piece_decision = None
            self.placement_done_time = None
            if state in STOPPED_STATES:
                self.pass_in_flight = None
                self.pending_pass = None

    def dribble_disp(self, robot_id: int, ball: aux.Point) -> float:
        """Смещение мяча при ведении для робота (0, если не ведёт)"""
        track = self.dribble.get(robot_id)
        if track is None:
            return 0.0
        return aux.dist(ball, track.start_ball_pos)

    def metric_max(self, name: str, value: float) -> None:
        """Обновить максимум метрики"""
        self.metrics[name] = max(self.metrics.get(name, value), value)

    def metric_add(self, name: str, value: float = 1.0) -> None:
        """Увеличить счётчик метрики"""
        self.metrics[name] = self.metrics.get(name, 0.0) + value


def update_memory(memory: Memory, fld_: fld.Field, world: "WorldModel", now: float, kick_model: "KickModel") -> None:
    """Обновить трекеры по кадру (ТЗ 5.3, п. 2)"""
    _update_possession(memory, world, now)
    _update_touch(memory, fld_, world, now)
    _update_dribble(memory, fld_, world, now)
    _update_double_touch(memory, fld_, world, now)
    _update_ball_in_area(memory, fld_, world, now)
    _update_penalty(memory, fld_, world)
    _update_kicks(memory, fld_, world, now, kick_model)
    _update_pass(memory, fld_, world, now)


def _update_possession(memory: Memory, world: "WorldModel", now: float) -> None:
    raw = world.raw_possession
    if raw != memory.possession_candidate:
        memory.possession_candidate = raw
        memory.possession_candidate_since = now
    if raw != memory.possession and now - memory.possession_candidate_since >= params.POSSESSION_HYSTERESIS:
        memory.possession = raw
        memory.possession_since = now
    world.possession = memory.possession


def _update_touch(memory: Memory, fld_: fld.Field, world: "WorldModel", now: float) -> None:
    owner = fld_.robot_with_ball
    if owner is not None:
        memory.last_owner = (owner.color, owner.r_id)
        memory.last_touch = (owner.color, owner.r_id, now)
        return
    ball = world.ball
    for robot in world.allies + world.enemies:
        if aux.dist(robot.get_pos(), ball) < params.DOUBLE_TOUCH_OTHER_DIST:
            memory.last_touch = (robot.color, robot.r_id, now)


def _update_dribble(memory: Memory, fld_: fld.Field, world: "WorldModel", now: float) -> None:
    owner = fld_.robot_with_ball
    owner_id: Optional[int] = None
    if owner is not None and owner.color == fld_.ally_color:
        owner_id = owner.r_id
        track = memory.dribble.get(owner_id)
        if track is None:
            memory.dribble[owner_id] = DribbleTrack(world.ball, now, now)
        else:
            track.last_seen = now
            memory.metric_max("max_dribble_disp", aux.dist(world.ball, track.start_ball_pos))
        memory.metric_max("max_ang_vel_with_ball", abs(owner.get_anglevel()))
    for r_id in list(memory.dribble):
        if r_id != owner_id and now - memory.dribble[r_id].last_seen > params.BALL_LOST_TIMEOUT:
            del memory.dribble[r_id]


def _update_double_touch(memory: Memory, fld_: fld.Field, world: "WorldModel", now: float) -> None:
    ball = world.ball
    state = fld_.game_state

    if state in STOPPED_STATES:
        memory.double_touch_banned_id = None
        memory.set_piece_armed = False

    # Снятие запрета: мяча коснулся другой робот
    banned = memory.double_touch_banned_id
    if banned is not None:
        owner = fld_.robot_with_ball
        if owner is not None and not (owner.color == fld_.ally_color and owner.r_id == banned):
            memory.double_touch_banned_id = None
        else:
            for robot in world.allies + world.enemies:
                if robot.color == fld_.ally_color and robot.r_id == banned:
                    continue
                if aux.dist(robot.get_pos(), ball) < params.DOUBLE_TOUCH_OTHER_DIST:
                    memory.double_touch_banned_id = None
                    break

    # Взвод: наш KICKOFF / FREE_KICK
    in_set_piece = state in (const.State.KICKOFF, const.State.FREE_KICK)
    if in_set_piece and memory.we_active:
        memory.set_piece_last_time = now
        if not memory.set_piece_armed:
            memory.set_piece_armed = True
            memory.set_piece_ball_start = ball
            memory.set_piece_kicker_near_time = -1.0
    kicker_id = memory.set_piece_kicker_id
    if not memory.set_piece_armed or memory.set_piece_ball_start is None:
        return

    if kicker_id is not None and fld_.allies[kicker_id].is_used():
        if aux.dist(fld_.allies[kicker_id].get_pos(), ball) < params.DOUBLE_TOUCH_KICKER_DIST:
            memory.set_piece_kicker_near_time = now

    if aux.dist(ball, memory.set_piece_ball_start) > params.RULE_BALL_IN_PLAY_DIST:
        if kicker_id is not None and now - memory.set_piece_kicker_near_time < params.DOUBLE_TOUCH_NEAR_WINDOW:
            memory.double_touch_banned_id = kicker_id
            memory.metric_add("double_touch_bans")
        memory.set_piece_armed = False
    elif not in_set_piece and now - memory.set_piece_last_time > params.SET_PIECE_ARM_GRACE:
        # Мяч введён по таймеру 10 с без касания (окно - на запаздывание смещения мяча после RUN)
        memory.set_piece_armed = False


def _update_ball_in_area(memory: Memory, fld_: fld.Field, world: "WorldModel", now: float) -> None:
    in_area = aux.is_point_inside_poly(world.ball, fld_.ally_goal.hull)
    if in_area and not fld.is_ball_not_in_play(fld_.game_state):
        if memory.ball_in_our_area_since is None:
            memory.ball_in_our_area_since = now
        memory.metric_max("max_ball_in_area_time", now - memory.ball_in_our_area_since)
    else:
        memory.ball_in_our_area_since = None


def _update_penalty(memory: Memory, fld_: fld.Field, world: "WorldModel") -> None:
    state = fld_.game_state
    if state == const.State.PENALTY:
        if memory.penalty_ball_start is None:
            memory.penalty_ball_start = world.ball
            memory.penalty_ball_in_play = False
        elif aux.dist(world.ball, memory.penalty_ball_start) > params.RULE_BALL_IN_PLAY_DIST:
            memory.penalty_ball_in_play = True
    else:
        memory.penalty_ball_start = None
        memory.penalty_ball_in_play = False


def _update_kicks(memory: Memory, fld_: fld.Field, world: "WorldModel", now: float, kick_model: "KickModel") -> None:
    speed = world.ball_speed
    jump = speed - memory.prev_ball_speed
    memory.prev_ball_speed = speed

    log = memory.kick_log
    if log is not None and not log.done:
        log.max_speed = max(log.max_speed, speed)
        if now - log.start_time > params.KICK_LOG_TIME:
            log.done = True
            memory.last_kick_max_speed = log.max_speed
            memory.metric_max("max_ball_speed_after_kick", log.max_speed)
            kick_model.register_kick(log.voltage, log.max_speed)
            if log.max_speed > params.BALL_SPEED_ALERT:
                memory.metric_add("ball_speed_alerts")

    if jump < params.KICK_DETECT_SPEED_JUMP:
        return
    kicker = None
    best = float(params.KICK_DETECT_DIST)
    for robot in world.allies:
        dist = aux.dist(robot.get_pos(), world.ball)
        if dist < best:
            best = dist
            kicker = robot
    if kicker is None:
        return
    if log is None or log.done or now - log.start_time > params.KICK_LOG_TIME:
        memory.kick_log = KickLog(kicker.r_id, memory.kick_voltage_issued.get(kicker.r_id, 0), now, speed)
        memory.metric_add("our_kicks")
        pending = memory.pending_pass
        if pending is not None and pending.from_id == kicker.r_id:
            duration = pending.expected_arrival - pending.start_time
            memory.pass_in_flight = PassInfo(pending.from_id, pending.to_id, pending.target, now, now + duration)
            memory.pending_pass = None
            memory.metric_add("passes")


def _update_pass(memory: Memory, fld_: fld.Field, world: "WorldModel", now: float) -> None:
    info = memory.pass_in_flight
    if info is None:
        return
    owner = fld_.robot_with_ball
    if owner is not None:
        if owner.color == fld_.ally_color and owner.r_id == info.to_id:
            memory.metric_add("passes_received")
        memory.pass_in_flight = None
        return
    late = now > info.expected_arrival + params.PASS_LATE_TIME
    stopped = world.ball_speed < params.BALL_STOPPED_SPEED and now - info.start_time > params.PASS_STOPPED_TIME
    if late or stopped or fld_.game_state in STOPPED_STATES:
        memory.pass_in_flight = None
