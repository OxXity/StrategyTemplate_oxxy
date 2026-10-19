"""Parameters of the «shadow, press, goalkeeper» strategy (all distances in mm, times in s)"""

import math
from typing import Optional

from bridge import const

# ROLES (goalkeeper id is GK from .env)
# Fixed ids of robots; None or an absent robot -> the role is given to a free robot
SHADOW_ID: Optional[int] = 1
ATTACKER_ID: Optional[int] = 4
# Robot that takes our penalty; None -> attacker
PENALTY_KICKER_ID: Optional[int] = None

# BALL MODEL
BALL_DECEL = 500  # ball slowdown, mm/s^2
ROLLING_SPEED = 300  # faster ball is "rolling": go to the interception point
ROBOT_SPEED = const.MAX_SPEED * 0.8  # expected robot speed for interception

# STABILIZATION OF ROTATION
FACE_MIN_DIST = 150  # closer points give a noisy direction -> keep the current angle
AIM_SMALL_ANGLE = 0.25  # change of the kick direction that is followed at once, rad
AIM_HOLD_TIME = 0.6  # a big change of the kick target is accepted not more often than this
CHASE_DIST = 600  # the ball rolls away from A closer than this -> chase it, don't intercept

# GOALKEEPER
GK_ARC_R = 350  # radius of the arc in front of the goal center
GK_ARC_MAX_ANGLE = math.radians(65)  # max deviation of the arc point from the goal axis
GK_LINE_FORW = const.ROBOT_R + 50  # distance from the goal line to the blocking line
GK_TAKE_SPEED = 300  # slower ball in our zone is taken by GK
GK_MAX_HOLD = 2.5  # max time to wait for the pass (by the rules < 3 s)

# SHADOW
SHADOW_MIN_D = 300  # P = F + d * unit(B - F), d = clamp(0.25 * |B - F|, MIN_D, MAX_D)
SHADOW_MAX_D = 700
SHADOW_K = 0.25

# ATTACKER
ATTACKER_GK_BLOCK_DIST = 600  # distance from the ball on the line "ball -> F" when the ball is at their GK
SHOT_DIST = 2500  # max distance to the goal for a shot
MIN_WINDOW = 2 * const.BALL_R + 80  # min width of a free window in the goal
DRIBBLE_STEP = 600  # length of one push when dribbling
STUCK_TIME = 2.0  # stuck longer -> shoot as is
STUCK_RADIUS = 200  # the ball moves less than this -> robot is stuck
STUCK_NEAR = 300  # robot is closer to the ball than this -> it is fighting for the ball
EXIT_X = 250  # exit point for the pass: distance from the center line to our side
EXIT_Y = 900  # exit point: distance from the field axis

# RECEIVING A PASS FROM S (A waits on the way to their goal)
RECEIVE_X_MIN = -300  # grid of points along x (to their goal): from this ...
RECEIVE_ZONE_GAP = 400  # ... to their zone minus this
RECEIVE_Y = 1100  # grid along y: [-RECEIVE_Y, RECEIVE_Y]
RECEIVE_AHEAD = 300  # the point is ahead of the ball at least by this (if possible)
RECEIVE_MIN_DIST = 1400  # distance from the ball to the point
RECEIVE_MIN_DIST_K = 5  # penalty for the point closer than RECEIVE_MIN_DIST (score per mm)
RECEIVE_MAX_DIST = 2200
RECEIVE_TRAVEL_K = 0.4  # penalty for the way of A to the point (score per mm)
RECEIVE_HYST = 300  # the point is changed only if a new one is better by this score
RECEIVE_READY_DIST = 1000  # A is closer to the point -> pass to the point on lead
RECEIVE_WAIT_DIST = 2500  # A is closer to the point -> S may wait for it
RECEIVE_SIDE_FREE = 700  # |y| of the receive point without penalty ...
RECEIVE_SIDE_K = 2.0  # ... farther to the side line: penalty per mm
RECEIVE_TURN_K = 600  # penalty of the receive point for the turn after receiving (score per rad)
RECEIVE_ANGLE_OFFSET = math.radians(15)  # A receives the ball turned to the goal by this angle (dribbler still catches)
RECEIVE_FACE_DIST = 1000  # the pass is closer than this to A -> face the ball straight (no turn to the goal)
RECEIVE_LINE_DIST = 800  # A shifts along the pass line to receive the ball, if it is closer than this
TURN_WITH_BALL_ANGLE = 0.35  # A with the ball turns smoothly (without kick) if the kick angle differs more
TURN_LOOKAHEAD = 0.3  # step of the smooth turn with the ball, rad (speed = angle gain * step)
S_MAX_HOLD = 1.5  # S holds the ball waiting for A not longer than this
S_HOLD_SAFE = 600  # S doesn't hold the ball if an enemy is closer than this

# EXCHANGE S <-> A
TAKER_HYST = 200  # hysteresis of choosing the robot for a free ball
S_TAKER_BONUS = 300  # S takes the free ball even if it is farther than A by this
S_TAKE_DIST = 800  # S takes the free ball closer than this also on their half
FORWARD_HYST = 300  # hysteresis of choosing the enemy forward F

# PASSES
PASS_SAFE_DIST = 800  # pass to A only if A is farther from F than this
PASS_MIN_DIST = 900  # don't pass to A closer than this (too fast ball at a short distance)
PASS_LINE_MARGIN = const.ROBOT_R + const.BALL_R + 30  # robot closer to the pass line blocks it
CLEAR_X_K = 0.55  # x of the clearance point = enemy_goal.x * CLEAR_X_K
CLEAR_Y_OFFSET = 400  # clearance point: distance from the side line

# SET PIECES
SET_PIECE_DIST = const.KEEP_BALL_DIST + 150  # distance to the ball for the enemy set piece
KICKOFF_BEHIND = const.KEEP_BALL_DIST + 100  # distance behind the ball before our kickoff
PENALTY_SHOOT_DIST = 1200  # push the ball closer to the goal before the penalty shot
PENALTY_PUSH = 500  # length of the penalty push
PENALTY_OTHERS_DIST = 1200  # other robots stay behind the ball at this distance
CORNER_OFFSET = const.BALL_R + 80  # shot point: distance from the post
