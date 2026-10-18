"""
Все настраиваемые параметры стратегии (ТЗ, раздел 13)

Единицы: мм, рад, мм/с, с. Где значения для симулятора и реальных роботов различаются,
используется sim_real(sim, real) по const.IS_SIMULATOR_USED.
Параметры не читаются из .env (R-5.2).
"""

# pylint: disable=invalid-name
from typing import TypeVar

from bridge import const

T = TypeVar("T", int, float)


def sim_real(sim: T, real: T) -> T:
    """Значение для симулятора или для реальных роботов"""
    return sim if const.IS_SIMULATOR_USED else real


ROBOT_R = const.ROBOT_R
BALL_R = const.BALL_R
GOAL_PEN_DX = const.GOAL_PEN_DX

# --- Правила (Регламент; не менять) ---
RULE_STOP_BALL_DIST = 500  # 5.1.1, 8.4.3
RULE_DEF_AREA_MARGIN = 200  # 8.4.1
RULE_PENALTY_OTHERS_DIST = 1000  # 5.3.5
RULE_MAX_DRIBBLE_DIST = 1000  # 8.4.1
RULE_MAX_BALL_SPEED = 3000  # 8.4.2
RULE_STOP_MAX_SPEED = 1500  # 8.4.3
RULE_COLLISION_REL_SPEED = 1500  # 8.4.2
RULE_BALL_IN_AREA_TIME = 10.0  # 8.4.1
RULE_SET_PIECE_TIME = 10.0  # 5.4
RULE_BALL_IN_PLAY_DIST = 50  # 5.4, 8.2
RULE_PENALTY_TIME = 10.0  # 5.3.5

# --- Запасы безопасности ---
SAFETY_MARGIN = 60
KEEP_BALL_DIST = const.KEEP_BALL_DIST  # 660, R-0.3
ENEMY_AREA_MARGIN_BALL_OUT = const.DEF_AREA_MARGIN_BALL_OUT  # = RULE_DEF_AREA_MARGIN + ROBOT_R + SAFETY_MARGIN = 360
PENALTY_OTHERS_DIST = RULE_PENALTY_OTHERS_DIST + ROBOT_R + 150  # 1250
DRIBBLE_SOFT_LIMIT = 700
DRIBBLE_HARD_LIMIT = 850
# DribbleTo не увозит мяч дальше DRIBBLE_SOFT_LIMIT - 50; выпуск мяча начинается чуть раньше этого
DRIBBLE_RELEASE_DISP = DRIBBLE_SOFT_LIMIT - 80
MAX_HOLD_TIME = 2.5
BALL_SPEED_SAFE = 2600
BALL_SPEED_ALERT = 2900
KICK_ADAPTIVE = False
COLLISION_SAFE_REL_SPEED = const.COLLISION_SAFE_REL_SPEED  # 1100, используется роутером
COLLISION_GUARD_DIST = const.COLLISION_GUARD_DIST  # 2 * ROBOT_R + 300, используется роутером
GK_CLEAR_SOFT_TIME = 3.0
GK_CLEAR_HARD_TIME = 6.0
SET_PIECE_MAX_WAIT = 4.0
PENALTY_SHOOT_DEADLINE = 7.0
NO_CROWD_RADIUS = 400
PUSH_MAX_CONTACT = 0.7
PUSH_CONTACT_DIST = 2 * ROBOT_R + 20  # контакт с соперником для анти-толкания
PUSH_RETREAT_DIST = 150
PUSH_RETREAT_TIME = 0.6
FIELD_OUT_MARGIN = const.FIELD_OUT_MARGIN  # 120, R-0.8
DOUBLE_TOUCH_KEEP_DIST = 300  # G-5: запрещённый робот держится дальше от мяча
DOUBLE_TOUCH_KICKER_DIST = ROBOT_R + BALL_R + 50  # G-5: исполнитель «у мяча» в момент ввода
DOUBLE_TOUCH_OTHER_DIST = ROBOT_R + BALL_R + 25  # G-5: другой робот коснулся мяча
DOUBLE_TOUCH_NEAR_WINDOW = 0.3  # с: исполнитель был у мяча не раньше этого до ввода
SET_PIECE_ARM_GRACE = 1.0  # с: после выхода из стандарта ещё ждём смещения мяча для G-5
STOP_TARGET_EXTRA = 50  # расстановка в STOP: KEEP_BALL_DIST + это

# --- Дриблер и работа с мячом (sim / real) ---
DRIBBLER_HOLD = 15
DRIBBLER_RECEIVE = 15
DRIBBLER_OFF_BEFORE_KICK = False
DRIBBLER_ON_DIST = 400  # R-4.9: включать дриблер ближе этого к мячу
BALL_HOLD_DIST = 90
BALL_HOLD_MAX_ANG_SPEED = sim_real(3.0, 1.2)  # рад/с - калибровать TURN_TEST
BALL_HOLD_MAX_ANG_ACC = sim_real(8.0, 4.0)  # рад/с²
BALL_HOLD_MAX_SPEED_FWD = sim_real(800, 600)
BALL_HOLD_MAX_SPEED_SIDE = sim_real(400, 300)
BALL_HOLD_MAX_SPEED_BACK = sim_real(300, 150)
BALL_HOLD_MAX_ACC = sim_real(1500, 1000)
BALL_LOST_TIMEOUT = 0.15
PRESSURE_RADIUS = 500
PRESSURE_MAX_TURN = 0.8  # ≈45°
RECEIVE_MAX_BIAS = 0.35  # ≈20°
RECEIVE_CUSHION_SPEED = 200
RECEIVE_CUSHION_DIST = 300
RECEIVE_CUSHION_SHIFT = 40
RECEIVE_LINE_DIST = 2 * ROBOT_R  # мяч идёт «на робота», если линия мяча ближе
RECEIVE_MIN_BALL_SPEED = 300  # приём, а не подбор, если мяч быстрее
OMEGA_FREE = 6.0  # рад/с без мяча, для оценки времени
MAX_GRAB_TURN = 0.52  # R-4.4: поворот после захвата не больше ≈30°
GRAB_FOLLOW_SPEED = 200  # мяч уходит от робота быстрее - не подбирать сразу (после откидки)
GRAB_FOLLOW_DIST = 400

# --- Мяч и удары ---
BALL_DECEL = 400  # мм/с², калибровать KICK_CALIBRATION
PASS_ARRIVAL_SPEED = sim_real(1000, 800)
PASS_MIN_SPEED = 700
PASS_MIN_ARRIVAL_SPEED = 300  # пас, замедленный ради остановки в поле, должен приходить не медленнее
PASS_MIN_VALUE = 0.0  # пас в открытой игре - только с положительной ценностью
SHOT_SPEED = BALL_SPEED_SAFE
KICK_ALIGN_TOL_SHOT = 0.08
KICK_ALIGN_TOL_PASS = 0.12
KICK_MAX_ANG_VEL = 0.5
SET_PIECE_APPROACH_SPEED = 300
SET_PIECE_ALIGN_TOL = 0.06
SET_PIECE_PREP_GAP = 100  # точка подготовки удара со стандарта: за мячом на GRAB_ALIGN_DIST + это
SET_PIECE_LATERAL_TOL = 30
SET_PIECE_DECISION_LOCK = 0.5  # решение фиксируется за это время до удара
USE_CHIP_KICK = False
USE_KICK_FORWARD = False  # если автоудар не срабатывает при прижатом мяче - True (kick_forward)
KICK_TABLE_CALIBRATED = False
ONE_TOUCH_ENABLED = False
ONE_TOUCH_MAX_ANGLE = 0.7  # ≈40°
TAP_DIST_MIN = 400
TAP_DIST_MAX = 700
TAP_ALIGN_TOL = 0.15
KICK_DETECT_SPEED_JUMP = 700  # мм/с: рывок скорости мяча = удар
KICK_DETECT_DIST = ROBOT_R + BALL_R + 80  # удар наш, если наш робот был ближе
KICK_LOG_TIME = 0.5  # с: окно измерения максимальной скорости мяча после удара (G-7)
KICK_ADAPT_STEP = 0.9  # понижающий коэффициент при превышении BALL_SPEED_ALERT

# Таблицы «напряжение → скорость мяча, мм/с» (заполнить по KICK_CALIBRATION).
# До калибровки - консервативная линейная оценка: VOLTAGE_SHOOT даёт не меньше RULE_MAX_BALL_SPEED.
KICK_LINEAR_SPEED_PER_VOLT = RULE_MAX_BALL_SPEED / const.VOLTAGE_SHOOT
KICK_TABLE_SIM: list[tuple[int, float]] = [(v, v * KICK_LINEAR_SPEED_PER_VOLT) for v in range(0, 16)]
KICK_TABLE_REAL: list[tuple[int, float]] = [(v, v * KICK_LINEAR_SPEED_PER_VOLT) for v in range(0, 16)]
KICK_MIN_VOLTAGE = 1

# --- Оценки ---
SHOT_MIN_P = 0.35
SHOT_PRESSURE_MIN_P = 0.15
SHOT_OWN_HALF_MIN_P = 0.6
SHOT_MAX_DIST = 3000
SHOT_FULL_DIST = 1200  # dist_factor = 1 ближе
SHOT_FAR_FACTOR = 0.3  # dist_factor на SHOT_MAX_DIST
SHOT_POST_MARGIN = 30
SHOT_REACH_K = 0.6
SHOT_REACH_CAP = 250
SHOT_WIDTH_FULL = 0.25  # рад: открытый угол, при котором фактор ширины = 1
SHOT_GK_SHIFT = 0.25  # цель сдвигается от вратаря на эту долю ширины интервала
SHOT_GK_MARGIN_OFFSET = 0.1
SHOT_GK_MARGIN_SCALE = 0.3
SHOT_ENEMY_GK_LINE_GAP = 30  # ожидаемая позиция вратаря соперника: ROBOT_R + это от линии ворот
PASS_SAFETY_MARGIN = 0.25
PASS_P_SCALE = 0.145  # гладкость p_success(min margin)
PASS_MIN_SUCCESS = 0.75
PASS_LONG_PENALTY = 0.7  # множитель оценки, если скорость паса упёрлась в потолок
PASS_SAMPLES = 8  # точек на отрезке паса для оценки риска
PASS_OWN_GOAL_FRONT = GOAL_PEN_DX + 400  # «перед нашими воротами» для запрета поперечного паса
GK_PASS_MIN_SUCCESS = 0.85
FREE_KICK_SHOT_MIN_P = 0.3
FREE_KICK_PASS_MIN_SUCCESS = 0.8
FREE_KICK_SAFE_MARGIN = 0.3  # безопасный ввод: наш робот первый с запасом
PENALTY_SHOT_MIN_P = 0.4
PROGRESS_W = 0.2
ROT_W = 0.3
ENEMY_MAX_SPEED = 1500
ENEMY_MAX_ACC = 3000
REACTION_TIME = 0.1
ENEMY_REACTION_TIME = 0.15
INTERCEPT_HORIZON = 3.0
INTERCEPT_STEP = 0.05
CONTEST_RADIUS = 250
POSSESSION_HYSTERESIS = 0.15
SUPPORT_GRID_STEP = 200
SUPPORT_MAX_CANDIDATES = 200  # R-5.5
SUPPORT_TOP_K = 12  # полная оценка паса и удара - только для лучших кандидатов
SUPPORT_UPDATE_PERIOD = 0.3  # с
SUPPORT_MIN_DIST_BALL = 700
SUPPORT_MAX_DIST_BALL = 2200  # открывание не дальше этого от мяча
SUPPORT_MAX_AHEAD = 1200  # и не дальше этого впереди мяча по направлению атаки
SUPPORT_MIN_DIST_MATE = 700
SUPPORT_MIN_DIST_ENEMY = 400
SUPPORT_BORDER_MARGIN = 200
SUPPORT_AREA_MARGIN = ROBOT_R + 100
SUPPORT_SWITCH_GAIN = 0.25
SUPPORT_W_GOAL = 1.0  # a
SUPPORT_W_SAFETY = 0.8  # b
SUPPORT_W_FREE = 0.4  # c
SUPPORT_W_TIME = 0.3  # d
SUPPORT_W_REBOUND = 0.2  # e
SUPPORT_W_PROGRESS = 0.6  # дополнение к 7.3: открывание вперёд (продвижение относительно мяча)
SUPPORT_COVER_BEHIND = 1000  # страховка позади Нападающего
SUPPORT_COVER_RACE = 0.3  # с: страховать, если гонка за мячом не в нашу пользу сильнее
SUPPORT_MODE_HOLD = 1.2  # с: режим Опорного держится минимум столько (кроме срочной обороны)
ESCAPE_MAX_BACK_ANGLE = 1.9  # рад: уход с мячом под давлением не круче этого от направления атаки
DECISION_COMMIT_TIME = 0.3
KICK_DECISION_COMMIT = 1.2  # решение «ударить» держится на время доворота с мячом
DRIBBLE_DIRS = 8
DRIBBLE_DISTS = (250, 450)
DRIBBLE_TIME_W = 0.15
DRIBBLE_MAX_ANGLE = 1.0  # рад: ведение без давления - только в этом конусе к чужим воротам
SHOT_FALLBACK_MIN_P = 0.02  # нет хорошего варианта: бить по воротам, если p_goal не ниже
DRIBBLE_MIN_GAIN = 0.05  # ведение выбирается, если улучшает p_goal хотя бы на это
CLEAR_BORDER_MARGIN = 300
CLEAR_GRID_STEP = 400
PASS_DECISION_GAIN = 0.1  # пас вместо удара, если ценнее на это
PASS_RECEIVER_SLACK = 0.3  # с: получатель может опоздать к мячу не больше
PASS_LATE_TIME = 1.0  # с: пас «в полёте» сбрасывается, если мяч не пришёл за это время после расчётного
PASS_STOPPED_TIME = 0.3  # с: ... или мяч остановился спустя это время после удара

# --- Роли ---
ROLE_SWITCH_MARGIN = 0.6
ROLE_MIN_HOLD = 1.5
ROLE_WRONG_SIDE_PENALTY = 0.3
RACE_LOSE_MARGIN = 0.2
INTERCEPT_STEAL_MARGIN = 0.2
PRESS_DIST = 300
BLOCK_GAP = 40  # Нападающий у мяча соперника: ROBOT_R + BALL_R + это
ENEMY_AREA_WAIT_GAP = 50  # мяч в чужой штрафной: ждать вне big_hull + это
DEF_DIST_K = 0.4
DEF_DIST_MIN = 500
DEF_DIST_MAX = 1200
DEF_AREA_GAP = ROBOT_R + 60
MARK_FRACTION = 0.35  # опека: доля отрезка «опекаемый → мяч»
MARK_GOAL_SHIFT = 100  # опека: смещение к нашим воротам
REBOUND_AREA_GAP = 100  # добивание: вне big_hull + это

# --- Вратарь ---
GK_LINE_OFFSET = ROBOT_R + 30
GK_MAX_OUT = GOAL_PEN_DX - ROBOT_R - 50  # 350
GK_COME_OUT_DIST = 1200
GK_SHOT_SPEED = 400
GK_BALL_SLOW_SPEED = 300  # мяч в штрафной «медленный»
GK_PENALTY_LINE_OFFSET = 40
GK_HEADING_PREDICT_WEIGHT = 0.7
GK_Y_MARGIN = 50  # |y| на линии вратаря <= GOAL_DY / 2 - это
GK_CORNER_ZONE = 400  # мяч ближе к линии ворот - прижаться к ближней штанге
GK_PENALTY_OUT_MIN = 600
GK_PENALTY_OUT_MAX = 900
GK_PENALTY_TAKE_MARGIN = 0.1

# --- Стандарты ---
KICKOFF_KICKER_DIST = ROBOT_R + 150
KICKOFF_RECEIVER_X = 250
KICKOFF_RECEIVER_Y = 900
KICKOFF_DIRECT_SHOT = False
KICKOFF_HALF_GAP = ROBOT_R + 50  # центр не ближе к центральной линии
KICKOFF_CIRCLE_R = 500
KICKOFF_ENEMY_BLOCK_DIST = 700
CORNER_RECEIVER_GAP = ROBOT_R + 150
PENALTY_KICKER_DIST = 250
PENALTY_TAP_DISP = 500  # пенальти: откидка уже при таком ведении (медленный «хвост» подъезда - потеря времени)
AUTO_BALL_PLACEMENT = False
PLACEMENT_MAX_SPEED = 400
PLACEMENT_TOLERANCE = 100
PLACEMENT_WAIT = 0.3
PLACEMENT_RETREAT = 300

# --- Прочее ---
SAFE_MODE = True
DEBUG_SCENARIO = "NONE"  # NONE | KICK_CALIBRATION | TURN_TEST | DRIBBLE_TEST | PASS_TEST | SHOOT_TEST | GK_TEST
DEBUG_DRAW = True
DEBUG_DRAW_MAX_PRIMITIVES = 60
BALL_LOST_VISION = 0.5
BALL_STOPPED_SPEED = 100
SPARRING_SECOND_TEAM = False
EXCEPTION_PRINT_PERIOD = 1.0
TELEMETRY_PERIOD = 0.1
