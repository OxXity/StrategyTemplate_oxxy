"""
Модуль-прослойка между стратегией и отправкой пакетов на роботов
"""

from time import sleep, time
from typing import Optional

import attr
import attrs
import zmq
from cattrs import unstructure
from strategy_bridge.bus import DataBus, DataReader, DataWriter
from strategy_bridge.processors import BaseProcessor

from bridge import const, drawing
from bridge.auxiliary import aux, fld, rbt
from bridge.processors.python_controller import RobotCommand
from bridge.router import safety
from bridge.router.actions import Actions
from bridge.router.actions.action import Action, ActionDomain, ActionValues


@attr.s(auto_attribs=True)
class CommandSink(BaseProcessor):
    """
    Прослойка между стратегией и отправкой пакетов на роботов
    """

    processing_pause: Optional[float] = 1 / 100
    reduce_pause_on_process_time: bool = True

    def initialize(self, data_bus: DataBus) -> None:
        """
        Инициализация
        """
        super().initialize(data_bus)
        self.field_reader = DataReader(data_bus, const.FIELD_TOPIC)
        self.control_reader = DataReader(data_bus, const.CONTROL_TOPIC)
        self.image_writer = DataWriter(data_bus, const.IMAGE_TOPIC, 10)

        self.tmp_timer = time()

        self.field_b = fld.Field(const.Color.BLUE)
        self.field_y = fld.Field(const.Color.YELLOW)
        self.field: dict[const.Color, fld.Field] = {
            const.Color.BLUE: self.field_b,
            const.Color.YELLOW: self.field_y,
        }

        self.field[const.COLOR].router_image.timer = drawing.FeedbackTimer(time(), 10, 50)

        context = zmq.Context()
        self.s_control = context.socket(zmq.PUB)
        self.s_control.connect("tcp://127.0.0.1:5051")

        self.processed_actions: dict[const.Color, list[Action]] = {
            const.Color.BLUE: [Actions.Unused()] * const.TEAM_ROBOTS_MAX_COUNT,
            const.Color.YELLOW: [Actions.Unused()] * const.TEAM_ROBOTS_MAX_COUNT,
        }

        self.started = False

    def process(self) -> None:
        """
        Метод обратного вызова процесса
        """
        # control_data = DecoderTeamCommand(robot_commands=test_commands, isteamyellow=False)
        # self.s_control.send_json({"control": "actuate_robot", "data": unstructure(control_data)})
        # sleep(0.01)
        # return

        telemetry_message = drawing.get_wave() + "\n"
        new_field = self.field_reader.read_last()
        if new_field is not None:
            updated_field: fld.LiteField = new_field.content
            if self.field_b.last_update != updated_field.last_update:
                self.field_b.update_field(updated_field)
                self.field_y.update_field(updated_field)
                self.started = True

        cmds = self.control_reader.read_new()
        # if len(cmds) == 0:
        #     return
        for cmd in cmds:
            new_commands: list[RobotCommand] = cmd.content
            for command in new_commands:
                self.processed_actions[command.color][command.r_id] = command.action
                self.started = True

        if not self.started:
            return

        self.field[const.COLOR].router_image.timer.start(time())
        for color in [const.Color.BLUE, const.Color.YELLOW]:
            team_message: str = (
                f"TEAM {str(color)}\n"
                + "\tr_id\tvelFRW\tvelLEFT\tvelR\tangle\tkickUP\tkickFRW\tautoUP\tautoFRW\tvolt\tdrib\n"
            )
            team_commands: list[DecoderCommand] = []

            # Phase 1: Collect desired velocities and other action values
            desired_vels: dict[int, aux.Point] = {}
            action_results: dict[int, ActionValues] = {}

            for idx, action in enumerate(self.processed_actions[color]):
                if action.is_used and self.field[color].allies[idx].is_used():
                    domain = ActionDomain(
                        field=self.field[color],
                        game_state=self.field[color].game_state,
                        we_active=self.field[color].active_team in [const.Color.ALL, color],
                        robot=self.field[color].allies[idx],
                    )
                    values = ActionValues()
                    action.process(domain, values)

                    desired_vels[idx] = values.vel
                    action_results[idx] = values

            # Phase 2: Apply Barrier Filter (G-9: ограничение скорости сближения с соперниками)
            filtered_vels = safety.limit_approach_speeds(self.field[color], desired_vels, action_results)

            # Phase 3: Generate actual commands
            for idx, action in enumerate(self.processed_actions[color]):
                if action.is_used:
                    if self.field[color].allies[idx].is_used():
                        self.field[color].allies[idx].clear_fields()

                        values = action_results[idx]
                        values.vel = filtered_vels.get(idx, values.vel)

                        cur_command = command_from_values(self.field[color], self.field[color].allies[idx], values)
                        team_commands.append(cur_command)
                        team_message += create_telemetry(cur_command)
                    elif time() - self.field[color].allies[idx].last_update() > const.TIME_TO_DIE:
                        cur_command = stop_command(idx)
                        team_commands.append(cur_command)
                        team_message += create_telemetry(cur_command)
                        self.processed_actions[color][idx] = Actions.Unused()

            if len(team_commands) > 0:
                control_data = DecoderTeamCommand(
                    robot_commands=team_commands,
                    isteamyellow=(color == const.Color.YELLOW),
                )

                self.s_control.send_json({"control": "actuate_robot", "data": unstructure(control_data)})

            telemetry_message += team_message
            telemetry_message += "-" * 90 + "\n"

        self.field[const.COLOR].router_image.timer.end(time())
        self.field[const.COLOR].router_image.send_telemetry("COMMANDS TO ROBOTS", telemetry_message)
        self.image_writer.write(self.field[const.COLOR].router_image)

        # MANY useless spamming, only for debug with 1-3 robots
        self.image_writer.write(self.field[const.COLOR].path_image)

        self.field_b.clear_images()
        self.field_y.clear_images()

    def finalize(self) -> None:

        team_commands: list[DecoderCommand] = []
        for r_id in range(const.TEAM_ROBOTS_MAX_COUNT):
            team_commands.append(stop_command(r_id))

        for _ in range(5):  # NOTE not enough for sim
            for team in [True, False]:
                control_data = DecoderTeamCommand(robot_commands=team_commands, isteamyellow=team)
                self.s_control.send_json({"control": "actuate_robot", "data": unstructure(control_data)})
            sleep(0.002)

        self.s_control.close()


def command_from_values(field: fld.Field, robot: rbt.Robot, values: ActionValues) -> "DecoderCommand":
    """Turn ActionValues to commands for robots"""
    if const.IS_SIMULATOR_USED:
        return command_from_values_sim(field, robot, values)

    if values.beep == 0:
        robot.update_vel_xy(values.vel)
        aerr = aux.wind_down_angle(values.angle - robot.get_angle())
        robot.delta_angle = aerr

        reg_vel = aux.Point(robot.speed_x, -robot.speed_y)
        field.router_image.draw_line(
            robot.get_pos(),
            robot.get_pos() + aux.rotate(reg_vel, robot.get_angle()) / 5,
        )
    else:
        robot.speed_x = values.vel.x
        robot.speed_y = values.vel.y

        robot.speed_r = values.angle

    return DecoderCommand(
        robot_id=robot.r_id,
        kick_up=values.kick_up,
        kick_forward=values.kick_forward,
        auto_kick_up=values.auto_kick == 2,
        auto_kick_forward=values.auto_kick == 1,
        auto_kick_momentum=values.auto_kick == 3,
        kicker_setting=values.kicker_voltage,
        dribbler_setting=values.dribbler_speed,
        forward_vel=robot.speed_x,
        left_vel=-robot.speed_y,
        angular_vel=robot.speed_r if values.beep else None,
        angle=robot.delta_angle if not bool(values.beep) else None,
    )


def command_from_values_sim(field: fld.Field, robot: rbt.Robot, values: ActionValues) -> "DecoderCommand":

    if values.beep == 0:
        robot.update_vel_xy(values.vel)
        aerr = aux.wind_down_angle(values.angle - robot.get_angle())

        ang_vel = robot.angle_reg.process(aerr, -robot.get_anglevel())
        robot.speed_r = ang_vel

        reg_vel = aux.Point(robot.speed_x, -robot.speed_y)
        field.router_image.draw_line(
            robot.get_pos(),
            robot.get_pos() + aux.rotate(reg_vel, robot.get_angle()) / 5,
        )
    else:
        robot.speed_x = values.vel.x
        robot.speed_y = values.vel.y

        robot.speed_r = -values.angle

    return DecoderCommand(
        robot_id=robot.r_id,
        kick_up=values.kick_up,
        kick_forward=values.kick_forward,
        auto_kick_up=values.auto_kick == 2,
        auto_kick_forward=values.auto_kick in [1, 3],
        auto_kick_momentum=False,
        kicker_setting=values.kicker_voltage,
        dribbler_setting=values.dribbler_speed,
        forward_vel=robot.speed_x,
        left_vel=-robot.speed_y,
        angular_vel=robot.speed_r,
        angle=None,
    )


def create_telemetry(cmd: "DecoderCommand") -> str:
    """Create line for telemetry for single robot"""
    values = [
        str(cmd.robot_id),
        f"{cmd.forward_vel: .0f}",
        f"{cmd.left_vel: .0f}",
        f"{cmd.angular_vel: .0f}" if cmd.angular_vel is not None else "None",
        f"{cmd.angle: .0f}" if cmd.angle is not None else "None",
        str(cmd.kick_up),
        str(cmd.kick_forward),
        str(cmd.auto_kick_up),
        str(cmd.auto_kick_forward),
        str(cmd.kicker_setting),
        str(cmd.dribbler_setting),
    ]
    return "\t" + "\t".join(values) + "\n"


@attrs.define
class DecoderCommand:
    """Class with control command for one robot"""

    robot_id: int = attrs.field()

    kick_up: bool = attrs.field()
    kick_forward: bool = attrs.field()
    auto_kick_up: bool = attrs.field()
    auto_kick_forward: bool = attrs.field()
    auto_kick_momentum: bool = attrs.field()  # autokick for ricochet

    kicker_setting: int = attrs.field()  # 0-15 [popugi]
    dribbler_setting: float = attrs.field()  # 0-15 [popugi]

    forward_vel: float = attrs.field()  # [m/s]
    left_vel: float = attrs.field()  # [m/s]
    angular_vel: Optional[float] = attrs.field(default=None)  # [rad/s]
    angle: Optional[float] = attrs.field(default=None)  # [rad]


test_commands = [
    DecoderCommand(idx, True, False, False, False, False, 15, 15, 100, 0, 1, None)
    for idx in range(const.TEAM_ROBOTS_MAX_COUNT)
]


def stop_command(idx: int) -> DecoderCommand:
    """Command to stop a robot"""
    return DecoderCommand(idx, False, False, False, False, False, 0, 0, 0, 0, 0, None)


@attrs.define
class DecoderTeamCommand:
    """Class for sending control commands to robots of one team"""

    robot_commands: list[DecoderCommand] = attrs.field(factory=list)
    isteamyellow: bool = attrs.field(default=False)
