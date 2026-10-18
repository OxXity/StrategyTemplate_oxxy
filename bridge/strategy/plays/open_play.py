"""
Открытая игра: RUN / FORCE START (ТЗ, 9.7) и общий запуск ролей
"""

import traceback
from typing import Callable

from bridge.auxiliary import rbt
from bridge.router.actions import Action, Actions
from bridge.strategy import params
from bridge.strategy.context import RoleBehavior, TickContext
from bridge.strategy.guards import hold_position
from bridge.strategy.memory import Possession, Role
from bridge.strategy.roles.attacker import Attacker
from bridge.strategy.roles.defender import Defender
from bridge.strategy.roles.goalkeeper import Goalkeeper
from bridge.strategy.roles.support import Support

GOALKEEPER = Goalkeeper()
ATTACKER = Attacker()
SUPPORT = Support()
DEFENDER = Defender()

ROLE_BEHAVIORS: dict[Role, RoleBehavior] = {
    Role.GOALKEEPER: GOALKEEPER,
    Role.ATTACKER: ATTACKER,
    Role.SUPPORT: SUPPORT,
    Role.DEFENDER: DEFENDER,
}


def report_exception(ctx: TickContext, where: str, exc: BaseException) -> None:
    """R-5.6: исключение - в телеметрию и консоль не чаще раза в секунду"""
    message = f"{where}: {exc!r}"
    ctx.warnings.append(message)
    ctx.memory.metric_add("exceptions")
    if ctx.now - ctx.memory.last_exception_print_time > params.EXCEPTION_PRINT_PERIOD:
        ctx.memory.last_exception_print_time = ctx.now
        print(f"[STRATEGY] {message}\n{traceback.format_exc()}")


Behavior = Callable[[rbt.Robot, TickContext], Action]


def safe_act(ctx: TickContext, robot: rbt.Robot, where: str, func: Behavior) -> Action:
    """Выполнить поведение; в SAFE_MODE исключение даёт безопасное действие"""
    if not params.SAFE_MODE:
        return func(robot, ctx)
    try:
        return func(robot, ctx)
    except Exception as exc:  # pylint: disable=broad-except
        report_exception(ctx, f"{where} robot {robot.r_id}", exc)
        try:
            return hold_position(ctx, robot)
        except Exception:  # pylint: disable=broad-except
            return Actions.Stop()


def role_action(robot: rbt.Robot, ctx: TickContext) -> Action:
    """Действие робота по его роли"""
    role = ctx.roles.get(robot.r_id, Role.SUPPORT)
    world = ctx.world
    # Соло-режим: единственный полевой при владении соперника на нашей половине играет как защитник
    solo = len(world.field_allies) == 1 and role == Role.ATTACKER
    if solo and world.possession == Possession.THEIRS and world.in_our_half(world.ball):
        return SUPPORT.defend(robot, ctx, None)
    behavior = ROLE_BEHAVIORS[role]
    return behavior.act(robot, ctx)


def play(ctx: TickContext) -> dict[int, Action]:
    """Открытая игра по ролям"""
    actions: dict[int, Action] = {}
    for robot in ctx.world.allies:
        actions[robot.r_id] = safe_act(ctx, robot, "open_play", role_action)
    return actions
