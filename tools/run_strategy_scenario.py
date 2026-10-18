"""Exercise real Strategy -> router -> controller in a separate headless grSim.

Run from the repository root: venv/bin/python -m tools.run_strategy_scenario
Defaults use dedicated ports, not the interactive PAcmaCS instance.
"""

import argparse
import json
import math
import random
import select
import socket
import sys
import time

from strategy_bridge.pb import grsim_commands_pb2, grsim_replacement_pb2

# strategy-bridge 0.4.1 generated these two imports without the package prefix.
sys.modules.setdefault("grsim_commands_pb2", grsim_commands_pb2)
sys.modules.setdefault("grsim_replacement_pb2", grsim_replacement_pb2)
from strategy_bridge.pb.grsim_packet_pb2 import grSim_Packet
from strategy_bridge.pb.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket

from bridge import const
from bridge.auxiliary import aux, fld
from bridge.processors.router_processor import command_from_values_sim
from bridge.router.actions.action import ActionDomain, ActionValues
from bridge.strategy.strategy import Strategy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--vision-port", type=int, default=11006)
    parser.add_argument("--command-port", type=int, default=21011)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--expect-goal", action="store_true")
    parser.add_argument("--expect-combination", action="store_true")
    parser.add_argument("--color", choices=("BLUE", "YELLOW"), default="BLUE")
    parser.add_argument(
        "--scenario",
        choices=("attack", "defense", "keeper", "opponents", "two"),
        default="attack",
    )
    args = parser.parse_args()
    field = fld.Field(const.Color[args.color])
    field.game_state = const.State.RUN
    strategy = Strategy(rng=random.Random(args.seed))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", args.vision_port))
    sock.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_ADD_MEMBERSHIP,
        socket.inet_aton("224.5.23.2") + socket.inet_aton("0.0.0.0"),
    )
    sock.setblocking(False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    endpoint = ("127.0.0.1", args.command_port)
    forward = field.ally_goal.eye_forw.x
    packet = grSim_Packet()
    own_yellow = field.ally_color == const.Color.YELLOW
    for yellow in (False, True):
        for r_id, (x, y, angle) in enumerate(
            ((-2050, 0, 0), (400, -600, 0), (1400, 800, math.pi))
        ):
            own = yellow == own_yellow
            robot = packet.replacement.robots.add()
            robot.id, robot.yellowteam = r_id, yellow
            robot.x = (
                x * forward / 1000
                if own
                else (2.05 if r_id == 0 else 0.2 + r_id * 0.6) * forward
            )
            robot.y = y / 1000 if own else (0.25 if r_id == 0 else -1.1)
            robot.dir = math.degrees(angle if forward > 0 else math.pi - angle)
            enabled = (
                own and (args.scenario != "two" or r_id != 2)
            ) or args.scenario == "opponents"
            if not enabled:
                robot.x, robot.y = 5 + r_id, 5 if yellow else -5
    bx, by = {
        "attack": (600, 300),
        "two": (600, 300),
        "opponents": (600, 300),
        "defense": (-700, 300),
        "keeper": (-1900, 300),
    }[args.scenario]
    packet.replacement.ball.x = bx * forward / 1000
    packet.replacement.ball.y = by / 1000
    packet.replacement.ball.vx = packet.replacement.ball.vy = 0
    sender.sendto(packet.SerializeToString(), endpoint)
    time.sleep(0.1)
    while select.select([sock], [], [], 0)[0]:
        sock.recv(65535)
    started = time.monotonic()
    last_report = started
    previous = None
    trace = []
    peak_speed = 0
    peak_command_speed = 0
    received_frames = 0
    outcome = "timeout"
    while time.monotonic() - started < args.seconds:
        cycle = time.monotonic()
        balls, blue, yellow = [], {}, {}
        until = cycle + 0.02
        while time.monotonic() < until:
            if not select.select([sock], [], [], max(0, until - time.monotonic()))[0]:
                break
            detection = SSL_WrapperPacket.FromString(sock.recv(65535)).detection
            balls.extend(aux.Point(b.x, b.y) for b in detection.balls)
            blue.update({r.robot_id: r for r in detection.robots_blue})
            yellow.update({r.robot_id: r for r in detection.robots_yellow})
        now = time.time()
        if not balls:
            continue
        received_frames += 1
        for entity in [field.ball, *field.all_bots]:
            if not entity.last_update():
                entity.last_update_ = now - 0.02
        field.update_ball(sum(balls, aux.Point(0, 0)) / len(balls), now)
        for team, detections in ((field.b_team, blue), (field.y_team, yellow)):
            for r_id, detection in detections.items():
                if (
                    abs(detection.x) > const.FIELD_DX + 500
                    or abs(detection.y) > const.FIELD_DY + 500
                ):
                    continue
                team[r_id].update(
                    aux.Point(detection.x, detection.y), detection.orientation, now
                )
                team[r_id].used(1)
            for robot in team:
                if now - robot.last_update() > const.TIME_TO_DIE:
                    robot.used(0)
        field.update_active_allies(
            [r for r in field.allies if r.is_used() and r.r_id != field.gk_id]
        )
        field.update_active_enemies(
            [r for r in field.enemies if r.is_used() and r.r_id != field.enemy_gk_id]
        )
        holder = field.robot_with_ball
        if holder is None or not holder.is_used() or not field._is_ball_in(holder):
            holder = next(
                (r for r in field.all_bots if r.is_used() and field._is_ball_in(r)),
                None,
            )
        field.update_robot_with_ball(holder)
        field.update_ball_history()
        field.last_update = now
        actions = strategy.process(field)
        packet = grSim_Packet()
        packet.commands.isteamyellow = own_yellow
        packet.commands.timestamp = now
        for robot in field.active_allies(True):
            values = ActionValues()
            actions[robot.r_id].process(
                ActionDomain(field, const.State.RUN, True, robot), values
            )
            command = command_from_values_sim(field, robot, values)
            target = packet.commands.robot_commands.add()
            target.id = robot.r_id
            target.wheelsspeed = False
            target.veltangent = command.forward_vel / 1000
            target.velnormal = command.left_vel / 1000
            target.velangular = command.angular_vel
            target.spinner = command.dribbler_setting > 0
            target.kickspeedx = (
                command.kicker_setting * 0.4 if command.auto_kick_forward else 0
            )
            peak_command_speed = max(peak_command_speed, target.kickspeedx)
            target.kickspeedz = 0
        sender.sendto(packet.SerializeToString(), endpoint)
        peak_speed = max(peak_speed, field.ball.get_vel().mag())
        transfer = strategy.transfer
        shot_target = strategy.choose_shot_target(field)
        state = (
            strategy.mode,
            strategy.attack_phase,
            holder.r_id if holder else None,
            (
                (transfer.sender, transfer.receiver, transfer.released is not None)
                if transfer
                else None
            ),
            (round(shot_target.x), round(shot_target.y)),
        )
        elapsed = time.monotonic() - started
        if state != previous or time.monotonic() - last_report > 2:
            entry = dict(
                t=round(elapsed, 2),
                state=state,
                ball=[round(field.ball.get_pos().x), round(field.ball.get_pos().y)],
                shot_target=[round(shot_target.x), round(shot_target.y)],
                speed=round(field.ball.get_vel().mag()),
                robots={
                    r.r_id: [
                        round(r.get_pos().x),
                        round(r.get_pos().y),
                        round(r.get_angle(), 2),
                    ]
                    for r in field.active_allies(True)
                },
            )
            print(json.dumps(entry), flush=True)
            trace.append(entry)
            previous, last_report = state, time.monotonic()
        field.clear_images()
        if (
            field.ball.get_pos().x * forward > const.FIELD_DX + const.BALL_R
            and abs(field.ball.get_pos().y) < 400
        ):
            outcome = "goal"
            break
        if (
            field.ball.get_pos().x * forward < -const.FIELD_DX - const.BALL_R
            and abs(field.ball.get_pos().y) < 400
        ):
            outcome = "own_goal"
            break
    # Explicitly stop the test team at the end; no commands are left running.
    packet = grSim_Packet()
    packet.commands.isteamyellow = own_yellow
    packet.commands.timestamp = time.time()
    for robot in field.active_allies(True):
        target = packet.commands.robot_commands.add()
        target.id, target.wheelsspeed, target.spinner = robot.r_id, False, False
        target.veltangent = target.velnormal = target.velangular = target.kickspeedx = (
            target.kickspeedz
        ) = 0
    sender.sendto(packet.SerializeToString(), endpoint)
    phases = list(dict.fromkeys(entry["state"][1] for entry in trace))
    print(
        json.dumps(
            dict(
                result=outcome,
                peak_ball_speed=round(peak_speed),
                frames=received_frames,
                roles=strategy.roles,
                seconds=round(time.monotonic() - started, 2),
                phases=phases,
                peak_command_speed=round(peak_command_speed, 2),
            )
        ),
        flush=True,
    )
    if received_frames == 0:
        raise SystemExit("No vision frames received from test grSim")
    if args.expect_goal and outcome != "goal":
        raise SystemExit(f"Expected a goal, got {outcome}")
    expected = (
        ["RECOVER", "PASS_1", "SHOT"]
        if args.scenario == "two"
        else ["RECOVER", "PASS_1", "PASS_2", "SHOT"]
    )
    if args.expect_combination and phases != expected:
        raise SystemExit(f"Expected phases {expected}, got {phases}")


if __name__ == "__main__":
    main()
