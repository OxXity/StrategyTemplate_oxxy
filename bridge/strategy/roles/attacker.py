"""
Нападающий (ТЗ, 8.2)
"""

import math
from dataclasses import dataclass
from typing import Optional

from bridge import const
from bridge.auxiliary import aux, rbt
from bridge.router.actions import Action, Actions, DribbleActions
from bridge.strategy import evaluation
from bridge.strategy import kick_model as km
from bridge.strategy import params
from bridge.strategy.context import (
    TickContext,
    angle_diff,
    angle_to,
    ball_near_sideline,
    carry_limits,
    clamp_angle_towards,
    rotation_time,
    unit,
)
from bridge.strategy.guards import go_to
from bridge.strategy.memory import Decision
from bridge.strategy.roles import common
from bridge.strategy.world import WorldModel

# Время исполнения удара после доворота, с (R-4.5)
KICK_EXEC_TIME = 0.15
# Мяч у борта: забирать вдоль борта, если ближе к боковой линии
SIDELINE_GRAB_MARGIN = 250
# Решения, которые заканчиваются ударом по мячу
KICK_KINDS = ("SHOOT", "PASS", "TAP", "CLEAR")
# Откидка: свободный коридор от соперников, мм
TAP_CORRIDOR_GAP = 40
# Ведение: шаг не короче, запас до лимита ведения, мм
DRIBBLE_MIN_STEP = 100
DRIBBLE_LIMIT_GAP = 30


@dataclass
class Option:
    """Вариант действия с мячом"""

    kind: str
    target: aux.Point
    utility: float
    receiver_id: Optional[int] = None
    t_arrive: float = 0.0
    p: float = 0.0  # вероятность: гол для удара, успех для паса


def next_direction(ctx: TickContext, robot: rbt.Robot, from_point: aux.Point) -> tuple[float, Option]:
    """Направление следующего действия до касания мяча (R-4.4)"""
    world = ctx.world
    shot = evaluation.eval_shot(world, from_point, robot.r_id)
    if shot.p_goal >= params.SHOT_MIN_P:
        return angle_to(from_point, shot.target), Option("SHOOT", shot.target, shot.p_goal)
    best: Optional[Option] = None
    for mate in world.field_allies:
        if mate.r_id == robot.r_id or ctx.is_banned(mate):
            continue
        pas = evaluation.eval_pass(world, from_point, mate, mate.get_pos())
        if (
            pas.ok
            and pas.p_success >= params.PASS_MIN_SUCCESS
            and pas.value > 0
            and (best is None or pas.value > best.utility)
        ):
            t_arrive = evaluation.ball_time(aux.dist(from_point, mate.get_pos()), pas.speed)
            best = Option("PASS", mate.get_pos(), pas.value, mate.r_id, t_arrive, pas.p_success)
    if best is not None:
        return angle_to(from_point, best.target), best
    goal = ctx.field.enemy_goal.center
    if ball_near_sideline(from_point, SIDELINE_GRAB_MARGIN):
        forward = 0.0 if world.attack_sign > 0 else math.pi
        return forward, Option("FORWARD", from_point + unit(forward) * 1000, 0.0)
    return angle_to(from_point, goal), Option("FORWARD", goal, 0.0)


class Attacker:
    """Нападающий"""

    def act(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Действие Нападающего в открытой игре"""
        world = ctx.world
        if world.ball_lost_vision:
            return go_to(ctx, robot, robot.get_pos(), robot.get_angle())
        if ctx.has_ball(robot):
            return self.with_ball(robot, ctx)
        ctx.memory.attacker_decision = None
        if world.owner_is_theirs:
            return self.vs_owner(robot, ctx)
        return self.loose_ball(robot, ctx)

    # --- A. Мяч свободен или летит ---

    def loose_ball(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Перехват по прогнозу, приём, подбор с предориентацией"""
        world = ctx.world
        ball = world.ball
        retreat = self._anti_push(robot, ctx)
        if retreat is not None:
            return retreat

        if world.in_area(ball, ally=True):
            # Мяч у нашего вратаря: не трогать, открыться под пас
            side = 1.0 if ball.y >= 0 else -1.0
            point = world.from_att(-const.FIELD_DX + const.GOAL_PEN_DX + 700, side * const.FIELD_DY * 0.6)
            return go_to(ctx, robot, point, angle_to(point, ball))
        if world.in_area(ball, ally=False):
            point = common.wait_point_outside_area(ctx, ball, False, params.ENEMY_AREA_WAIT_GAP)
            return go_to(ctx, robot, point, angle_to(point, ball))

        their = world.their_intercepts.get(world.their_best_id) if world.their_best_id is not None else None
        if world.ball_race < -params.RACE_LOSE_MARGIN and their is not None and not math.isinf(their.t):
            # Соперник явно первый: встать между его точкой перехвата и нашими воротами
            point = their.point + (ctx.field.ally_goal.center - their.point).unity() * params.PRESS_DIST
            return go_to(ctx, robot, point, angle_to(point, their.point))

        _, option = next_direction(ctx, robot, ball)
        if common.ball_coming_to(ctx, robot):
            face = option.target
            return common.receive(ctx, robot, face, common.one_touch_voltage(ctx, robot, face))

        inter = world.our_intercepts.get(robot.r_id)
        if world.ball_moving and world.ball_speed > params.RECEIVE_MIN_BALL_SPEED and inter is not None:
            if aux.dist(inter.point, ball) > const.ROBOT_R * 3:
                # Перехват по прогнозу, а не погоня: встать на путь мяча лицом к нему
                point = inter.point + world.ball_dir * const.ROBOT_R
                action = go_to(ctx, robot, point, angle_to(point, ball))
                return action.compose(Actions.SetDribblerSpeed(params.DRIBBLER_RECEIVE))

        angle, option = next_direction(ctx, robot, ball)
        if option.kind == "SHOOT":
            voltage = ctx.kick_model.shot_voltage(common.robot_vel_along(robot, angle))
            direct = common.kick_on_contact(ctx, robot, option.target, voltage, params.KICK_ALIGN_TOL_SHOT)
            if direct is not None:
                return direct
        elif option.kind == "PASS" and option.receiver_id is not None:
            voltage, t_arrive = common.pass_kick(ctx, robot, option.target)
            direct = common.kick_on_contact(ctx, robot, option.target, voltage, params.KICK_ALIGN_TOL_PASS)
            if direct is not None:
                common.register_pass(ctx, robot, option.receiver_id, option.target, t_arrive)
                return direct
        return common.grab(ctx, robot, angle)

    # --- B. Мяч у Нападающего ---

    def _options(self, robot: rbt.Robot, ctx: TickContext, pressure: bool, t_enemy: float) -> list[Option]:
        world = ctx.world
        ball = world.ball
        heading = robot.get_angle()
        options: list[Option] = []

        def turn_ok(angle: float) -> bool:
            if pressure and angle_diff(heading, angle) > params.PRESSURE_MAX_TURN:
                return False
            return rotation_time(robot, angle) + KICK_EXEC_TIME < t_enemy or angle_diff(heading, angle) < 0.2

        shot = evaluation.eval_shot(world, ball, robot.r_id)
        shot_angle = angle_to(ball, shot.target)
        shot_util = shot.p_goal - params.ROT_W * rotation_time(robot, shot_angle)
        if shot.p_goal > 0 and turn_ok(shot_angle):
            options.append(Option("SHOOT", shot.target, shot_util, p=shot.p_goal))
            ctx.draw.line(ball, shot.target, (int(230 * (1 - shot.p_goal)), int(220 * shot.p_goal), 0), 2)

        for mate in world.field_allies:
            if mate.r_id == robot.r_id or ctx.is_banned(mate):
                continue
            point = mate.get_pos()
            pass_angle = angle_to(ball, point)
            pas = evaluation.eval_pass(world, ball, mate, point, rotation_time(robot, pass_angle))
            ctx.draw.line(ball, point, (0, 200, 0) if pas.ok else (200, 0, 0), 1)
            if pas.ok and pas.p_success >= params.PASS_MIN_SUCCESS and turn_ok(pass_angle):
                t_arrive = evaluation.ball_time(aux.dist(ball, point), pas.speed)
                options.append(Option("PASS", point, pas.value, mate.r_id, t_arrive, pas.p_success))
        return options

    def _dribble_option(
        self, robot: rbt.Robot, ctx: TickContext, remaining: float, away_from: Optional[aux.Point] = None
    ) -> Optional[Option]:
        """Ведение (8.2-B3); away_from - уход от давящего соперника (защита мяча, 8.2-B5)"""
        world = ctx.world
        ball = world.ball
        best: Optional[Option] = None
        for k in range(params.DRIBBLE_DIRS):
            angle = 2 * math.pi * k / params.DRIBBLE_DIRS
            if away_from is not None:
                if angle_diff(angle, angle_to(away_from, ball)) > math.pi / 2:
                    continue
                attack_dir = 0.0 if world.attack_sign > 0 else math.pi
                if angle_diff(angle, attack_dir) > params.ESCAPE_MAX_BACK_ANGLE:
                    continue  # не уходить с мячом назад к своим воротам
            for dist in sorted({min(d, remaining - DRIBBLE_LIMIT_GAP) for d in params.DRIBBLE_DISTS}):
                if dist < DRIBBLE_MIN_STEP:
                    continue
                point = ball + unit(angle) * dist
                if not world.in_field(point, params.SUPPORT_BORDER_MARGIN):
                    continue
                if world.in_area(point, True, const.ROBOT_R * 2) or world.in_area(point, False, const.ROBOT_R * 2):
                    continue
                speed = params.BALL_HOLD_MAX_SPEED_FWD
                t_us = dist / speed + rotation_time(robot, angle) * 0.5
                t_en = world.enemy_time_to(point)
                keep = evaluation.clip01((t_en - t_us) / 0.5) if not math.isinf(t_en) else 1.0
                shot = evaluation.eval_shot(world, point, robot.r_id, extra_time=t_us)
                progress = (world.x_att(point) - world.x_att(ball)) / (2 * const.FIELD_DX)
                util = (shot.p_goal + params.PROGRESS_W * progress) * keep - params.DRIBBLE_TIME_W * t_us
                if best is None or util > best.utility:
                    best = Option("DRIBBLE", point, util)
        return best

    def _forced_release(
        self, robot: rbt.Robot, ctx: TickContext, options: list[Option], escape: Optional[Option] = None
    ) -> Option:
        """Принудительный выпуск мяча: удар, безопасный пас, (уход от давления), откидка, вынос"""
        world = ctx.world
        ball = world.ball
        shots = [o for o in options if o.kind == "SHOOT" and o.p >= params.SHOT_PRESSURE_MIN_P]
        if shots:
            return max(shots, key=lambda o: o.utility)
        passes = [o for o in options if o.kind == "PASS"]
        good_passes = [o for o in passes if o.utility > params.PASS_MIN_VALUE]
        if good_passes:
            return max(good_passes, key=lambda o: o.utility)
        if escape is not None:
            return escape
        tap = self._tap_option(robot, ctx)
        if tap is not None and tap.p > 0:
            return tap  # откидка вперёд лучше паса назад
        if passes:
            return max(passes, key=lambda o: o.utility)
        if tap is not None:
            return tap
        # R-4.6: под давлением вынос только в конусе допустимого поворота
        target, _ = evaluation.safe_clear_target(world, ball, cone=(robot.get_angle(), params.PRESSURE_MAX_TURN))
        return Option("CLEAR", target, 0.0)

    def _tap_option(self, robot: rbt.Robot, ctx: TickContext) -> Optional[Option]:
        """Откидка в конусе текущего курса в зону, где мы первые"""
        world = ctx.world
        ball = world.ball
        heading = robot.get_angle()
        best: Optional[Option] = None
        for k in range(-2, 3):
            angle = heading + k * params.PRESSURE_MAX_TURN / 2
            for dist in (params.TAP_DIST_MIN, params.TAP_DIST_MAX):
                point = ball + unit(angle) * dist
                if not world.in_field(point, params.CLEAR_BORDER_MARGIN):
                    continue
                if world.in_area(point, True, const.ROBOT_R) or world.in_area(point, False, const.ROBOT_R):
                    continue
                if not self._corridor_clear(world, ball, point):
                    continue
                t_us = world.time_to_point(robot, point)
                t_en = world.enemy_time_to(point)
                margin = t_en - t_us
                progress = (world.x_att(point) - world.x_att(ball)) / (2 * const.FIELD_DX)
                util = min(margin, 1.0) + progress
                if margin > 0 and (best is None or util > best.utility):
                    best = Option("TAP", point, util, p=progress)
        return best

    @staticmethod
    def _corridor_clear(world: WorldModel, start: aux.Point, end: aux.Point) -> bool:
        """Мяч пройдёт от start до end, не попав в соперника"""
        for enemy in world.enemies:
            on_line = aux.closest_point_on_line(start, end, enemy.get_pos(), "S")
            if aux.dist(on_line, enemy.get_pos()) < evaluation.HIT_R + TAP_CORRIDOR_GAP:
                return False
        return True

    def _decision_valid(self, ctx: TickContext, decision: Decision, disp: float) -> bool:
        """Решение держится DECISION_COMMIT_TIME; удар/пас/откидка/вынос - до KICK_DECISION_COMMIT"""
        age = ctx.now - decision.since
        is_kick = decision.kind in KICK_KINDS
        if decision.target is None or age > (params.KICK_DECISION_COMMIT if is_kick else params.DECISION_COMMIT_TIME):
            return False
        if not is_kick and disp >= params.DRIBBLE_RELEASE_DISP:
            return False
        if decision.kind == "PASS":
            receiver = decision.receiver_id
            if receiver is None:
                return False
            mate = ctx.field.allies[receiver]
            return bool(mate.is_used()) and not ctx.is_banned(mate)
        return decision.kind in ("SHOOT", "DRIBBLE", "TAP", "CLEAR", "HOLD")

    def with_ball(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Каждый такт - варианты с мячом, максимальная полезность (8.2-B)"""
        world = ctx.world
        memory = ctx.memory
        ball = world.ball
        now = ctx.now
        track = memory.dribble.get(robot.r_id)
        disp = memory.dribble_disp(robot.r_id, ball)
        hold_time = now - track.start_time if track is not None else 0.0
        nearest = world.nearest_enemy(ball)
        pressure = nearest is not None and aux.dist(nearest.get_pos(), ball) < params.PRESSURE_RADIUS
        t_enemy = world.enemy_time_to(ball)

        decision = memory.attacker_decision
        if decision is not None and self._decision_valid(ctx, decision, disp):
            option = Option(decision.kind, decision.target or ball, decision.utility, decision.receiver_id)
        else:
            options = self._options(robot, ctx, pressure, t_enemy)
            limits_hit = disp >= params.DRIBBLE_RELEASE_DISP or hold_time >= params.MAX_HOLD_TIME
            escape = None
            if pressure and not limits_hit and nearest is not None:
                escape = self._dribble_option(robot, ctx, params.DRIBBLE_SOFT_LIMIT - disp, nearest.get_pos())
            if limits_hit or (pressure and not options):
                option = self._forced_release(robot, ctx, options, escape)
            else:
                option = self._choose(robot, ctx, options, disp)
            memory.attacker_decision = Decision(option.kind, option.target, now, option.utility, option.receiver_id)
        ctx.draw.text(robot.get_pos() + aux.Point(0, -180), option.kind, (255, 255, 0), 60)
        memory.debug["att_util"] = option.utility
        return self._execute(robot, ctx, option, track.start_ball_pos if track else ball)

    def _choose(self, robot: rbt.Robot, ctx: TickContext, options: list[Option], disp: float) -> Option:
        shots = [o for o in options if o.kind == "SHOOT" and o.p >= params.SHOT_MIN_P]
        passes = [o for o in options if o.kind == "PASS" and o.utility > params.PASS_MIN_VALUE]
        best_shot = max(shots, key=lambda o: o.utility) if shots else None
        best_pass = max(passes, key=lambda o: o.utility) if passes else None
        best: Optional[Option] = best_shot
        if best_pass is not None and (best is None or best_pass.utility > best.utility + params.PASS_DECISION_GAIN):
            best = best_pass
        dribble = self._dribble_option(robot, ctx, params.DRIBBLE_SOFT_LIMIT - disp)
        if dribble is not None and (best is None or dribble.utility > best.utility + params.DRIBBLE_MIN_GAIN):
            best = dribble
        if best is None:
            # Ничего хорошего: довернуться с мячом к лучшей цели удара до принудительного выпуска
            shot = evaluation.eval_shot(ctx.world, ctx.world.ball, robot.r_id)
            best = Option("HOLD", shot.target, 0.0)
        return best

    def _execute(self, robot: rbt.Robot, ctx: TickContext, option: Option, start_ball: aux.Point) -> Action:
        world = ctx.world
        ball = world.ball
        angle = angle_to(ball, option.target)
        vel_along = common.robot_vel_along(robot, angle)
        if option.kind == "SHOOT":
            voltage = ctx.kick_model.shot_voltage(vel_along)
            return common.dribble_kick(ctx, robot, option.target, voltage, params.KICK_ALIGN_TOL_SHOT)
        if option.kind == "PASS":
            voltage, t_arrive = common.pass_kick(ctx, robot, option.target)
            if option.receiver_id is not None:
                common.register_pass(ctx, robot, option.receiver_id, option.target, t_arrive)
            return common.dribble_kick(ctx, robot, option.target, voltage, params.KICK_ALIGN_TOL_PASS)
        if option.kind == "HOLD":
            return DribbleActions.HoldBall(
                angle, carry_limits(), dribbler_speed=params.DRIBBLER_HOLD, hold_dist=params.BALL_HOLD_DIST
            )
        if option.kind == "TAP":
            dist = aux.dist(ball, option.target)
            voltage = ctx.kick_model.tap_voltage(dist)
            ctx.memory.kick_voltage_issued[robot.r_id] = voltage
            return DribbleActions.TapForward(
                angle, voltage, params.TAP_ALIGN_TOL, carry_limits(), hold_dist=params.BALL_HOLD_DIST
            )
        if option.kind == "CLEAR":
            voltage = ctx.kick_model.kick_voltage(km.stop_speed(aux.dist(ball, option.target)), vel_along)
            return common.dribble_kick(ctx, robot, option.target, voltage, params.KICK_ALIGN_TOL_PASS)
        # DRIBBLE: курс - к цели ведения, не круче PRESSURE_MAX_TURN от текущего
        face = clamp_angle_towards(robot.get_angle(), angle, params.PRESSURE_MAX_TURN)
        ctx.draw.circle(start_ball, (255, 200, 0), params.DRIBBLE_SOFT_LIMIT)
        ctx.draw.circle(start_ball, (255, 0, 0), params.RULE_MAX_DRIBBLE_DIST)
        return DribbleActions.DribbleTo(
            option.target,
            face,
            start_ball,
            params.DRIBBLE_SOFT_LIMIT,
            carry_limits(),
            dribbler_speed=params.DRIBBLER_HOLD,
            hold_dist=params.BALL_HOLD_DIST,
        )

    # --- C. Мяч у соперника ---

    def _anti_push(self, robot: rbt.Robot, ctx: TickContext) -> Optional[Action]:
        """G-10: долгий контакт без овладения - отъехать и зайти с другой стороны"""
        memory = ctx.memory
        now = ctx.now
        r_id = robot.r_id
        until = memory.push_retreat_until.get(r_id, 0.0)
        closest = ctx.world.nearest_enemy(robot.get_pos())
        if now < until and closest is not None:
            away = (robot.get_pos() - closest.get_pos()).unity()
            point = robot.get_pos() + away * params.PUSH_RETREAT_DIST
            return go_to(ctx, robot, point, angle_to(point, ctx.world.ball))

        in_contact = closest is not None and aux.dist(closest.get_pos(), robot.get_pos()) < params.PUSH_CONTACT_DIST
        if in_contact and not ctx.has_ball(robot):
            since = memory.push_contact_since.setdefault(r_id, now)
            if now - since > params.PUSH_MAX_CONTACT:
                memory.push_retreat_until[r_id] = now + params.PUSH_RETREAT_TIME
                memory.push_side[r_id] = -memory.push_side.get(r_id, 1.0)
                memory.push_contact_since.pop(r_id, None)
                ctx.memory.metric_add("push_retreats")
        else:
            memory.push_contact_since.pop(r_id, None)
        return None

    def vs_owner(self, robot: rbt.Robot, ctx: TickContext) -> Action:
        """Перекрыть удар у мяча; отбор, если соперник спиной к нашим воротам"""
        world = ctx.world
        ball = world.ball
        enemy = world.owner
        assert enemy is not None
        retreat = self._anti_push(robot, ctx)
        if retreat is not None:
            return retreat

        to_goal = angle_to(enemy.get_pos(), ctx.field.ally_goal.center)
        back_turned = angle_diff(enemy.get_angle(), to_goal) > math.pi / 2
        if back_turned and not world.in_area(ball, True) and not world.in_area(ball, False):
            side = ctx.memory.push_side.get(robot.r_id, 0.0)
            if side == 0.0:
                rel = aux.vec_mult(unit(enemy.get_angle()), robot.get_pos() - enemy.get_pos())
                side = 1.0 if rel >= 0 else -1.0
                ctx.memory.push_side[robot.r_id] = side
            # Подход сбоку от дриблера соперника
            return common.grab(ctx, robot, enemy.get_angle() - side * math.pi / 2)

        point = ball + (ctx.field.ally_goal.center - ball).unity() * (const.ROBOT_R + const.BALL_R + params.BLOCK_GAP)
        target = go_to(ctx, robot, point, angle_to(point, ball), ignore_ball=True)
        if isinstance(target, Actions.GoToPoint):
            target.ignore_robots = {enemy.color: [enemy.r_id]}
        return target.compose(Actions.SetDribblerSpeed(params.DRIBBLER_HOLD))
