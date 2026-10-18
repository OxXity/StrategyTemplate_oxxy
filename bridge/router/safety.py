"""
Ограничитель скорости сближения с соперниками (G-9, Регламент 8.4.2 «Столкновение»)

Для каждого нашего робота и соперника ближе COLLISION_GUARD_DIST проекция разности
скоростей на линию «мы → соперник» не превышает COLLISION_SAFE_REL_SPEED.
Отъезд от соперника не ограничивается.
"""

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.router.actions.action import ActionValues


def limit_approach_speed(
    our_pos: aux.Point,
    our_vel: aux.Point,
    enemies: list[tuple[aux.Point, aux.Point]],
    safe_rel_speed: float = const.COLLISION_SAFE_REL_SPEED,
    guard_dist: float = const.COLLISION_GUARD_DIST,
) -> aux.Point:
    """Ограничить командную скорость our_vel относительно соперников (позиция, скорость)"""
    vel = aux.Point(our_vel.x, our_vel.y)
    for enemy_pos, enemy_vel in enemies:
        delta = enemy_pos - our_pos
        if delta.mag() > guard_dist or delta.mag() < 1e-6:
            continue
        n = delta.unity()
        v_rel = aux.scal_mult(vel - enemy_vel, n)
        if v_rel > safe_rel_speed:
            vel = vel - n * (v_rel - safe_rel_speed)
    return vel


def limit_approach_speeds(
    field: fld.Field,
    desired_vels: dict[int, aux.Point],
    values: dict[int, ActionValues],
) -> dict[int, aux.Point]:
    """Phase 2 роутера: применить ограничитель ко всем командам с глобальной скоростью (beep=0)"""
    enemies = [(enemy.get_pos(), enemy.get_vel()) for enemy in field.enemies if enemy.is_used()]
    if not enemies:
        return desired_vels
    filtered: dict[int, aux.Point] = {}
    for idx, vel in desired_vels.items():
        action_values = values.get(idx)
        if action_values is None or action_values.beep:
            filtered[idx] = vel
            continue
        filtered[idx] = limit_approach_speed(field.allies[idx].get_pos(), vel, enemies)
    return filtered
