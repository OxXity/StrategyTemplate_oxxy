"""
Упрощённый кинематический симулятор для офлайн-проверки стратегии (не замена grSim/ER-Force).

Стратегия -> действия роутера (100 Гц) -> кинематика роботов и мяча.
Мяч: качение с BALL_DECEL, захват дриблером в зоне перед роботом, удар по auto_kick/kick_forward
с перезарядкой кикера. Соперник: вратарь на линии ворот, полевые - простой прессинг.
Время подменяется (FakeClock), поэтому прогон идёт быстрее реального.
"""

from typing import Callable, Optional

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.router import safety
from bridge.router.actions import Action
from bridge.router.actions.action import ActionDomain, ActionValues
from bridge.strategy import params
from bridge.strategy.kick_model import KickModel
from bridge.strategy.strategy import Strategy
from tests.conftest import Pose, make_field

DT = 0.01  # шаг роутера и физики
FRONT = 90.0  # центр мяча в дриблере от центра робота
KICK_COOLDOWN = 0.5


class FakeClock:
    """Подменяемое время"""

    def __init__(self, start: float = 10000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class Sim:  # pylint: disable=too-many-instance-attributes
    """Мини-симулятор одного матча"""

    def __init__(
        self,
        clock: FakeClock,
        ball: tuple[float, float],
        allies: dict[int, Pose],
        enemies: dict[int, Pose],
        state: const.State = const.State.RUN,
        active: const.Color = const.Color.ALL,
        enemy_ai: bool = True,
    ) -> None:
        self.clock = clock
        self.field: fld.Field = make_field(ball=ball, allies=allies, enemies=enemies, state=state, active_team=active)
        self.strategy = Strategy()
        self.actions: list[Optional[Action]] = [None] * const.TEAM_ROBOTS_MAX_COUNT
        self.attached: Optional[tuple[bool, int]] = None
        self.enemy_ai = enemy_ai
        self.kick_model = KickModel()
        self.t = 0.0
        self.goals = {"ours": 0, "theirs": 0}
        self.events: list[tuple[float, str]] = []
        self.fouls: list[tuple[float, str]] = []
        self.kicks: list[tuple[float, bool, int, float]] = []  # (t, наш?, id, скорость)
        self.max_dribble = 0.0
        self.dribble_start: Optional[tuple[int, aux.Point]] = None
        self.cooldown: dict[tuple[bool, int], float] = {}

    # --- геометрия ---

    def robots(self) -> list[rbt.Robot]:
        """Все роботы на поле"""
        return [r for r in self.field.allies + self.field.enemies if r.is_used()]

    def front(self, robot: rbt.Robot) -> aux.Point:
        """Точка мяча в дриблере"""
        return robot.get_pos() + aux.rotate(aux.RIGHT, robot.get_angle()) * FRONT

    def in_front(self, robot: rbt.Robot) -> bool:
        """Мяч в зоне дриблера/кикера"""
        rel = aux.rotate(self.field.ball.get_pos() - robot.get_pos(), -robot.get_angle())
        return 60 < rel.x < 130 and abs(rel.y) < 45

    # --- соперник ---

    def enemy_command(self, robot: rbt.Robot) -> tuple[aux.Point, float, bool]:
        """Простой соперник: вратарь на линии, ближайший к мячу бьёт по нашим воротам"""
        field = self.field
        ball = field.ball.get_pos()
        if robot.r_id == field.enemy_gk_id:
            goal = field.enemy_goal
            target = goal.center + goal.eye_forw * 130 + aux.Point(0, aux.minmax(ball.y, -300, 300))
            return target, aux.angle_to_point(target, ball), False
        if field.game_state != const.State.RUN:
            target = robot.get_pos()
            if aux.dist(target, ball) < 700:
                target = ball + (target - ball).unity() * 700
            return target, aux.angle_to_point(target, ball), False
        enemies = field.active_enemies(False)
        closest = min(enemies, key=lambda e: aux.dist(e.get_pos(), ball))
        if robot is closest:
            angle = aux.angle_to_point(ball, field.ally_goal.center)
            behind = ball - aux.rotate(aux.RIGHT, angle) * 95
            if aux.dist(robot.get_pos(), behind) > 60 and aux.dist(robot.get_pos(), ball) > 120:
                behind = ball - aux.rotate(aux.RIGHT, angle) * 200
            return behind, angle, True
        target = aux.Point(ball.x * 0.5, -ball.y * 0.6)
        return target, aux.angle_to_point(target, ball), False

    # --- физика ---

    def move(self, robot: rbt.Robot, vel: aux.Point, angle: float, beep: int, vmax: float) -> None:
        """Голономный робот с ограничением ускорения и П-регулятором угла"""
        if vel.mag() > vmax:
            vel = vel.unity() * vmax
        dv = vel - robot.get_vel()
        if dv.mag() > const.MAX_ACCELERATION * DT:
            vel = robot.get_vel() + dv.unity() * const.MAX_ACCELERATION * DT
        robot._vel = vel
        robot._pos = robot.get_pos() + vel * DT
        omega = 0.0 if beep else aux.minmax(8 * aux.wind_down_angle(angle - robot.get_angle()), -10, 10)
        robot._anglevel = omega
        robot._angle = aux.wind_down_angle(robot.get_angle() + omega * DT)

    def kick(self, robot: rbt.Robot, voltage: int, ours: bool) -> None:
        """Удар с перезарядкой кикера"""
        key = (ours, robot.r_id)
        if self.t - self.cooldown.get(key, -10.0) < KICK_COOLDOWN:
            return
        self.cooldown[key] = self.t
        direction = aux.rotate(aux.RIGHT, robot.get_angle())
        speed = self.kick_model.speed_for_voltage(voltage) + max(0.0, aux.scal_mult(robot.get_vel(), direction))
        self.field.ball._vel = direction * speed
        self.field.ball._pos = self.front(robot) + direction * 25
        self.attached = None
        self.kicks.append((self.t, ours, robot.r_id, speed))
        if ours and speed > params.RULE_MAX_BALL_SPEED:
            self.fouls.append((self.t, "ball speed"))

    def router_values(self) -> dict[int, ActionValues]:
        """Действия наших роботов, как в CommandSink (с ограничителем сближения)"""
        field = self.field
        values: dict[int, ActionValues] = {}
        desired: dict[int, aux.Point] = {}
        for robot in field.active_allies(True):
            action = self.actions[robot.r_id]
            if action is None:
                continue
            we_active = field.active_team in (const.Color.ALL, field.ally_color)
            domain = ActionDomain(field, field.game_state, we_active, robot)
            value = ActionValues()
            value.vel = aux.Point(0, 0)
            action.process(domain, value)
            values[robot.r_id] = value
            desired[robot.r_id] = value.vel
        filtered = safety.limit_approach_speeds(field, desired, values)
        for r_id, value in values.items():
            value.vel = filtered[r_id]
        return values

    def step(self) -> None:
        """Шаг физики DT"""
        field = self.field
        values = self.router_values()
        for robot in field.active_allies(True):
            value = values.get(robot.r_id)
            if value is None:
                continue
            vmax = const.STOP_SPEED if field.game_state == const.State.STOP else const.MAX_SPEED
            self.move(robot, value.vel, value.angle, value.beep, vmax)
            touching = self.in_front(robot)
            if touching and (value.auto_kick in (1, 3) or value.kick_forward) and value.kicker_voltage > 0:
                self.kick(robot, value.kicker_voltage, True)
            elif touching and value.dribbler_speed > 0 and self.attached is None:
                self.attached = (True, robot.r_id)
            if self.attached == (True, robot.r_id) and (value.dribbler_speed == 0 or abs(robot.get_anglevel()) > 6):
                self.attached = None
        for robot in field.active_enemies(True):
            if not self.enemy_ai:
                robot._vel = aux.Point(0, 0)
                continue
            target, angle, want_kick = self.enemy_command(robot)
            self.move(robot, (target - robot.get_pos()) * 3, angle, 0, 1200)
            if want_kick and self.in_front(robot) and abs(aux.wind_down_angle(robot.get_angle() - angle)) < 0.3:
                self.kick(robot, 5, False)
        self.move_ball()
        self.separate_robots()

    def move_ball(self) -> None:
        """Мяч: в дриблере - перед роботом, иначе качение и отскок от корпусов"""
        ball = self.field.ball
        if self.attached is not None:
            ours, r_id = self.attached
            robot = (self.field.allies if ours else self.field.enemies)[r_id]
            ball._pos = self.front(robot)
            ball._vel = robot.get_vel()
        else:
            speed = ball.get_vel().mag()
            if speed > 0:
                ball._vel = ball.get_vel().unity() * max(0.0, speed - params.BALL_DECEL * DT)
            ball._pos = ball.get_pos() + ball.get_vel() * DT
            for robot in self.robots():
                delta = ball.get_pos() - robot.get_pos()
                if delta.mag() < const.ROBOT_R + const.BALL_R and not self.in_front(robot):
                    normal = delta.unity()
                    vn = aux.scal_mult(ball.get_vel() - robot.get_vel(), normal)
                    if vn < 0:
                        ball._vel = ball.get_vel() - normal * (1.5 * vn)
                    ball._pos = robot.get_pos() + normal * (const.ROBOT_R + const.BALL_R + 1)
        ball.last_update_ = self.clock.now

    def separate_robots(self) -> None:
        """Роботы не проходят друг сквозь друга"""
        robots = self.robots()
        for i, first in enumerate(robots):
            for second in robots[i + 1 :]:
                delta = second.get_pos() - first.get_pos()
                if 1e-6 < delta.mag() < 2 * const.ROBOT_R:
                    push = delta.unity() * ((2 * const.ROBOT_R - delta.mag()) / 2)
                    first._pos = first.get_pos() - push
                    second._pos = second.get_pos() + push

    def update_flags(self) -> None:
        """Признаки, которые в реальности считает FieldCreator"""
        field = self.field
        field.ball_start_point = field.ball.get_pos() - field.ball.get_vel() * 0.2
        owner = None
        for robot in field.enemies + field.allies:
            if robot.is_used() and field._is_ball_in(robot):  # pylint: disable=protected-access
                owner = robot
        field.robot_with_ball = owner

    def check_fouls(self) -> None:
        """Нарушения нашей команды, которые видит симулятор"""
        field = self.field
        ball = field.ball.get_pos()
        for robot in field.active_allies(False):
            for goal in (field.ally_goal, field.enemy_goal):
                depth = (robot.get_pos().x - goal.center.x) * goal.eye_forw.x
                if (
                    depth < const.GOAL_PEN_DX - const.ROBOT_R
                    and abs(robot.get_pos().y) < const.GOAL_PEN_DY / 2 - const.ROBOT_R
                ):
                    self.fouls.append((self.t, f"robot {robot.r_id} in area"))
        if self.attached is not None and self.attached[0]:
            if self.dribble_start is None or self.dribble_start[0] != self.attached[1]:
                self.dribble_start = (self.attached[1], ball)
            dist = aux.dist(ball, self.dribble_start[1])
            self.max_dribble = max(self.max_dribble, dist)
            if dist > params.RULE_MAX_DRIBBLE_DIST:
                self.fouls.append((self.t, "dribble"))
        else:
            self.dribble_start = None

    def ball_left_field(self) -> bool:
        """Гол или аут"""
        ball = self.field.ball.get_pos()
        if abs(ball.x) <= const.FIELD_DX and abs(ball.y) <= const.FIELD_DY:
            return False
        if abs(ball.x) > const.FIELD_DX and abs(ball.y) < const.GOAL_DY / 2:
            ours = (ball.x > 0) == (self.field.enemy_goal.center.x > 0)
            self.goals["ours" if ours else "theirs"] += 1
            self.events.append((self.t, "goal ours" if ours else "goal theirs"))
        else:
            self.events.append((self.t, "out"))
        return True

    def run(
        self,
        seconds: float,
        on_tick: Optional[Callable[["Sim"], None]] = None,
        on_out: Optional[Callable[["Sim"], None]] = None,
    ) -> None:
        """Прогон: такт стратегии 20 мс, два шага физики на такт"""
        for _ in range(int(seconds / 0.02)):
            self.update_flags()
            self.actions = self.strategy.process(self.field)
            if on_tick is not None:
                on_tick(self)
            for _ in range(2):
                self.clock.now += DT
                self.t += DT
                for robot in self.robots():
                    robot.last_update_ = self.clock.now
                self.step()
                self.check_fouls()
                if self.ball_left_field():
                    if on_out is None:
                        return
                    on_out(self)
