"""
Модель удара: напряжение ↔ скорость мяча (ТЗ, раздел 12; G-7, R-0.5)

Любое напряжение удара стратегия берёт только отсюда. Целевая скорость мяча:
min(желаемая, BALL_SPEED_SAFE) - max(0, скорость робота вдоль направления удара).
"""

import math
from typing import Optional

from bridge import const
from bridge.strategy import params


class KickModel:
    """Таблица напряжений с адаптивной поправкой (KICK_ADAPTIVE)"""

    def __init__(self, table: Optional[list[tuple[int, float]]] = None, calibrated: Optional[bool] = None) -> None:
        if table is None:
            table = params.KICK_TABLE_SIM if const.IS_SIMULATOR_USED else params.KICK_TABLE_REAL
        self.calibrated = params.KICK_TABLE_CALIBRATED if calibrated is None else calibrated
        self.table = sorted(table)
        # Поправка «реальная скорость / табличная» по напряжениям (только растёт - безопасно)
        self.adapt: dict[int, float] = {}
        self.max_voltage = 15 if self.calibrated else const.VOLTAGE_SHOOT

    def warning(self) -> Optional[str]:
        """Предупреждение для телеметрии"""
        if not self.calibrated:
            return "KICK TABLE NOT CALIBRATED: conservative linear model, voltage <= VOLTAGE_SHOOT"
        return None

    def speed_for_voltage(self, voltage: float) -> float:
        """Скорость мяча при напряжении (кусочно-линейная интерполяция таблицы)"""
        table = self.table
        if voltage <= table[0][0]:
            base = table[0][1]
        elif voltage >= table[-1][0]:
            base = table[-1][1]
        else:
            base = table[-1][1]
            for (v0, s0), (v1, s1) in zip(table, table[1:]):
                if v0 <= voltage <= v1:
                    base = s0 + (s1 - s0) * (voltage - v0) / max(v1 - v0, 1e-9)
                    break
        return base * self.adapt.get(int(round(voltage)), 1.0)

    def voltage_for_speed(self, speed: float) -> int:
        """Наибольшее напряжение, не превышающее скорость speed (округление вниз)"""
        best = 0
        for voltage in range(1, self.max_voltage + 1):
            if self.speed_for_voltage(voltage) <= speed:
                best = voltage
        if best == 0 and self.speed_for_voltage(params.KICK_MIN_VOLTAGE) <= params.BALL_SPEED_SAFE:
            best = params.KICK_MIN_VOLTAGE
        return best

    def voltage_near(self, speed: float, cap: float) -> int:
        """Напряжение со скоростью мяча ближе всего к speed, но не выше cap (пасы, R-0.5)"""
        cap = min(cap, params.BALL_SPEED_SAFE)
        best = 0
        best_err = math.inf
        for voltage in range(params.KICK_MIN_VOLTAGE, self.max_voltage + 1):
            model_speed = self.speed_for_voltage(voltage)
            if model_speed > cap:
                continue
            err = abs(model_speed - speed)
            if err < best_err:
                best, best_err = voltage, err
        return best if best > 0 else self.voltage_for_speed(cap)

    @staticmethod
    def target_speed(desired: float, robot_vel_along: float) -> float:
        """Целевая скорость мяча с учётом потолка и собственной скорости робота"""
        return max(0.0, min(desired, params.BALL_SPEED_SAFE) - max(0.0, robot_vel_along))

    def shot_voltage(self, robot_vel_along: float = 0.0) -> int:
        """Удар по воротам"""
        return self.voltage_for_speed(self.target_speed(params.SHOT_SPEED, robot_vel_along))

    def kick_voltage(self, speed: float, robot_vel_along: float = 0.0) -> int:
        """Удар с заданной начальной скоростью мяча"""
        return self.voltage_for_speed(self.target_speed(speed, robot_vel_along))

    def pass_voltage(self, distance: float, robot_vel_along: float = 0.0, cap: float = math.inf) -> int:
        """Пас: мяч приходит со скоростью около PASS_ARRIVAL_SPEED; cap - потолок (остановка в поле)"""
        own = max(0.0, robot_vel_along)
        return self.voltage_near(pass_speed(distance) - own, min(cap, params.BALL_SPEED_SAFE) - own)

    def pass_voltage_for_speed(self, speed: float, robot_vel_along: float = 0.0, cap: float = math.inf) -> int:
        """Пас с заданной начальной скоростью мяча"""
        own = max(0.0, robot_vel_along)
        return self.voltage_near(speed - own, min(cap, params.BALL_SPEED_SAFE) - own)

    def tap_voltage(self, distance: float) -> int:
        """Откидка: мяч останавливается примерно через distance"""
        return self.kick_voltage(stop_speed(distance))

    def register_kick(self, voltage: int, measured_speed: float) -> None:
        """G-7: после удара измерена скорость; при превышении - понижающая поправка"""
        if not params.KICK_ADAPTIVE or voltage <= 0:
            return
        if measured_speed > params.BALL_SPEED_ALERT:
            predicted = self.speed_for_voltage(voltage)
            if predicted > 0:
                factor = max(measured_speed / predicted, 1.0 / params.KICK_ADAPT_STEP)
                self.adapt[voltage] = self.adapt.get(voltage, 1.0) * factor


def pass_speed(distance: float) -> float:
    """v0 = sqrt(PASS_ARRIVAL_SPEED² + 2·BALL_DECEL·d), в пределах [PASS_MIN_SPEED, BALL_SPEED_SAFE]"""
    v0 = math.sqrt(params.PASS_ARRIVAL_SPEED**2 + 2 * params.BALL_DECEL * max(distance, 0.0))
    return aux_clip(v0, params.PASS_MIN_SPEED, params.BALL_SPEED_SAFE)


def stop_speed(distance: float) -> float:
    """Начальная скорость, при которой мяч остановится через distance"""
    return min(math.sqrt(2 * params.BALL_DECEL * max(distance, 0.0)), params.BALL_SPEED_SAFE)


def stop_distance(speed: float) -> float:
    """Путь мяча до остановки"""
    return speed**2 / (2 * params.BALL_DECEL)


def ball_travel_time(distance: float, speed: float) -> float:
    """Время, за которое мяч со скоростью speed пройдёт distance (inf, если не дойдёт)"""
    if distance <= 0:
        return 0.0
    disc = speed**2 - 2 * params.BALL_DECEL * distance
    if disc < 0:
        return math.inf
    return (speed - math.sqrt(disc)) / params.BALL_DECEL


def ball_speed_after(distance: float, speed: float) -> float:
    """Скорость мяча после пути distance"""
    return math.sqrt(max(0.0, speed**2 - 2 * params.BALL_DECEL * max(distance, 0.0)))


def aux_clip(x: float, low: float, high: float) -> float:
    """Ограничить x отрезком [low, high]"""
    return max(low, min(high, x))
