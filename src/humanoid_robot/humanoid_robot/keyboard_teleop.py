import math
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Twist
from std_msgs.msg import String, Bool
from tf2_ros import TransformBroadcaster

# Unitree G1 (23-DOF) actuated joints, in the order defined by g1_23dof.urdf.
JOINT_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
]

JOINT_LIMITS = {
    'left_hip_pitch_joint': (-2.5307, 2.8798),
    'left_hip_roll_joint': (-0.5236, 2.9671),
    'left_hip_yaw_joint': (-2.7576, 2.7576),
    'left_knee_joint': (-0.0873, 2.8798),
    'left_ankle_pitch_joint': (-0.8727, 0.5236),
    'left_ankle_roll_joint': (-0.2618, 0.2618),
    'right_hip_pitch_joint': (-2.5307, 2.8798),
    'right_hip_roll_joint': (-2.9671, 0.5236),
    'right_hip_yaw_joint': (-2.7576, 2.7576),
    'right_knee_joint': (-0.0873, 2.8798),
    'right_ankle_pitch_joint': (-0.8727, 0.5236),
    'right_ankle_roll_joint': (-0.2618, 0.2618),
    'waist_yaw_joint': (-2.6180, 2.6180),
    'left_shoulder_pitch_joint': (-3.0892, 2.6704),
    'left_shoulder_roll_joint': (-1.5882, 2.2515),
    'left_shoulder_yaw_joint': (-2.6180, 2.6180),
    'left_elbow_joint': (-1.0472, 2.0944),
    'left_wrist_roll_joint': (-1.9722, 1.9722),
    'right_shoulder_pitch_joint': (-3.0892, 2.6704),
    'right_shoulder_roll_joint': (-2.2515, 1.5882),
    'right_shoulder_yaw_joint': (-2.6180, 2.6180),
    'right_elbow_joint': (-1.0472, 2.0944),
    'right_wrist_roll_joint': (-1.9722, 1.9722),
}

# Root link of the URDF (the pelvis floats free and is moved by this node's TF broadcast).
BASE_FRAME = 'pelvis'

# --- Walking cruise speeds -------------------------------------------------
WALK_SPEED = 0.35       # m/s, forward/backward
STRAFE_SPEED = 0.2      # m/s, sideways
TURN_SPEED = 0.6        # rad/s

# --- Procedural walking-gait tuning ---------------------------------------
# One full gait cycle (phase 0..2*pi) covers this much forward travel and
# contains one swing of each leg. Everything below is an open-loop cosmetic
# gait (no balance/dynamics) driven purely by this phase.
STRIDE_LENGTH = 0.6                # metres per full gait cycle
TURN_TO_GAIT_SPEED = 0.3           # how much turning-in-place also drives the gait

# Hip/knee/ankle sagittal-plane keyframes across one gait cycle (0.0-1.0 =
# 0-100% of the cycle for a single leg), taken from standard clinical gait
# analysis figures (e.g. Perry's "Gait Analysis", Kenhub's gait-cycle summary,
# PM&R KnowledgeNow's biomechanics-of-normal-gait reference): hip flexion at
# initial contact, the knee's double-flexion bump (a small one absorbing
# impact at loading response, a big one clearing the foot during swing), and
# the ankle's push-off plantarflexion spike. Values are (fraction, hip_deg,
# knee_deg, ankle_deg), hip/knee positive = flexion, ankle positive =
# dorsiflexion (toes up).
GAIT_KEYFRAMES_DEG = [
    (0.00, 20, 5, 0),     # initial contact (heel strike)
    (0.10, 20, 18, -3),   # loading response - knee flexes to absorb impact
    (0.30, 0, 5, 3),      # midstance - hip vertical, knee nearly straight
    (0.50, -10, 5, 10),   # terminal stance - hip hyperextension, heel rising
    (0.60, -5, 15, -10),  # pre-swing - weight releasing, knee starts folding
    (0.62, 0, 40, -20),   # toe-off - ankle push-off plantarflexion peak
    (0.73, 15, 60, 0),    # initial swing - peak knee flexion, foot clears ground
    (0.87, 25, 30, 0),    # midswing - hip nearing peak flexion, knee extending
    (1.00, 20, 5, 0),     # terminal swing, wraps back to initial contact
]
GAIT_KEYFRAMES = [
    (frac, math.radians(hip), math.radians(knee), math.radians(ankle))
    for frac, hip, knee, ankle in GAIT_KEYFRAMES_DEG
]

HIP_ROLL_SWAY = 0.05               # side-to-side weight shift
ARM_SWING = 0.45                   # opposite-arm-to-leg counter-swing (~25 deg, typical natural gait)
# shoulder_pitch=0 on this URDF is NOT "arm hanging at the side" - it's a
# forward-reaching pose (the joint's own mounting is tilted). Verified
# numerically against the URDF: +0.082 rad (~4.7 deg) is what actually swings
# the arm down to hang naturally, so every arm pose below is built around
# this rest angle instead of a bare 0 - otherwise the arms look frozen,
# raised up front rather than hanging down like a standing person's.
ARM_REST_PITCH = 0.082
# Likewise elbow=0 is NOT a straight arm - it's a sharp ~100 deg bend
# (confirmed numerically: the upper-arm/forearm interior angle at elbow=0 is
# only ~80 deg, a tight "robot arm" fold). The joint's own geometry can't
# reach a true 180 deg straight arm, but close to its extension limit
# (-1.0472 rad) gives ~137 deg - a natural, relaxed, barely-bent arm, the way
# a standing person's arm actually looks, not locked at a hard right angle.
ELBOW_REST = -1.0
ELBOW_BASE_BEND = 0.25             # a bit more bend added on top while walking

# Standing/base pose bias applied under every other offset (trim, gait) for
# joints whose zero position isn't a natural rest pose on this URDF.
REST_POSE = {
    'left_shoulder_pitch_joint': ARM_REST_PITCH,
    'right_shoulder_pitch_joint': ARM_REST_PITCH,
    'left_elbow_joint': ELBOW_REST,
    'right_elbow_joint': ELBOW_REST,
}
WAIST_YAW_COUNTER = 0.06
PELVIS_BOB_AMPLITUDE = 0.02        # small vertical bob, two bobs per gait cycle
GAIT_BLEND_RATE = 4.0              # 1/s, how fast the gait fades in/out at start/stop

# --- Pick-and-place full-body animation -------------------------------------
# Driven by navigate_to_goal over /manipulation_cmd ("pick"/"place") once it
# walks the robot up to a box or a drop point: a real crouch-reach-grasp-rise
# sequence using both legs (squat), the pelvis height, and both arms - not
# just an arm wiggle - so it reads as a person bending down to lift something
# with both hands, not a box teleporting into a static pose. Offsets here are
# added on top of REST_POSE/the gait, same sign convention as elsewhere
# (hip_pitch/shoulder_pitch negative-forward, see the notes above).
PICK_DURATION = 2.8     # seconds for the full crouch -> grasp -> rise sequence
PLACE_DURATION = 3.2    # a little slower/gentler setting the box down
# Metres the pelvis sinks during the crouch - a deep squat (knee/hip angles
# below are picked to match: at these leg joint limits the foot stays almost
# exactly planted, verified by FK, for this much pelvis drop) because this
# URDF has no waist/torso pitch joint to lean forward with - squatting the
# legs is the only way to bring the shoulders down toward the ground at all.
SQUAT_HEIGHT_DROP = 0.50

MANIP_POSES = {
    'rest': {},
    'bend': {
        # Both legs squat deeply and symmetrically (pelvis height drop is
        # applied separately, see SQUAT_HEIGHT_DROP) - with no waist pitch
        # joint on this URDF, this is the only way to bring the shoulders
        # down near the ground. The arms stay close to a natural hang rather
        # than swinging the shoulder forward (shoulder_pitch pivots the arm
        # on an arc centred on the shoulder, and hanging straight down is
        # already the lowest point that arc reaches - swinging it forward
        # only raises the hand). The elbow bend below isn't "straighten the
        # arm to reach further" either - a full sweep of the joint's FK found
        # the true lowest-reaching elbow angle is this bent, not straight.
        'left_hip_pitch_joint': -1.327,
        'left_knee_joint': 2.653,
        'left_ankle_pitch_joint': 0.464,
        'right_hip_pitch_joint': -1.327,
        'right_knee_joint': 2.653,
        'right_ankle_pitch_joint': 0.464,
        'left_shoulder_pitch_joint': 0.0,
        'left_shoulder_roll_joint': 0.1,
        'left_elbow_joint': 2.39,
        'right_shoulder_pitch_joint': 0.0,
        'right_shoulder_roll_joint': -0.1,
        'right_elbow_joint': 2.39,
    },
    'carry': {
        # standing tall again, both elbows bent, holding the box against
        # the chest with both hands - legs/pelvis back to neutral so normal
        # walking resumes cleanly once this is the held pose
        'left_shoulder_pitch_joint': -0.9,
        'left_shoulder_roll_joint': 0.3,
        'left_elbow_joint': 1.6,
        'right_shoulder_pitch_joint': -0.9,
        'right_shoulder_roll_joint': -0.3,
        'right_elbow_joint': 1.6,
    },
}
MANIP_HEIGHT_DELTA = {'rest': 0.0, 'bend': -SQUAT_HEIGHT_DROP, 'carry': 0.0}

# (fraction of animation, pose name). The hold at 'bend' stands in for the
# grasp/release moment - GRASP_FRACTION/RELEASE_FRACTION below mark the
# instant within that hold where the box actually changes hands, so it moves
# with the robot's hands through the crouch and the stand-back-up, not just
# at the very start/end of the whole animation.
PICK_TIMELINE = [(0.0, 'rest'), (0.35, 'bend'), (0.55, 'bend'), (1.0, 'carry')]
PLACE_TIMELINE = [(0.0, 'carry'), (0.25, 'bend'), (0.45, 'bend'), (1.0, 'rest')]
GRASP_FRACTION = 0.55    # within PICK_TIMELINE: box leaves the ground, into both hands
RELEASE_FRACTION = 0.45  # within PLACE_TIMELINE: box leaves both hands, onto the ground


def _blend_manip_poses(name_a, name_b, f):
    a, b = MANIP_POSES[name_a], MANIP_POSES[name_b]
    ease = 0.5 - 0.5 * math.cos(math.pi * f)
    joints = {n: a.get(n, 0.0) + (b.get(n, 0.0) - a.get(n, 0.0)) * ease for n in set(a) | set(b)}
    height = MANIP_HEIGHT_DELTA[name_a] + (MANIP_HEIGHT_DELTA[name_b] - MANIP_HEIGHT_DELTA[name_a]) * ease
    return joints, height


def _manip_timeline_lookup(timeline, progress):
    progress = max(0.0, min(1.0, progress))
    for i in range(len(timeline) - 1):
        t0, p0 = timeline[i]
        t1, p1 = timeline[i + 1]
        if t0 <= progress <= t1:
            f = (progress - t0) / (t1 - t0) if t1 > t0 else 0.0
            return _blend_manip_poses(p0, p1, f)
    name = timeline[-1][1]
    return dict(MANIP_POSES[name]), MANIP_HEIGHT_DELTA[name]


def _gait_keyframe_lookup(frac):
    """Interpolate (hip, knee, ankle) radians at a 0..1 fraction of the gait
    cycle, easing between keyframes so the curve has no sharp corners."""
    frac = frac % 1.0
    for i in range(len(GAIT_KEYFRAMES) - 1):
        t0, h0, k0, a0 = GAIT_KEYFRAMES[i]
        t1, h1, k1, a1 = GAIT_KEYFRAMES[i + 1]
        if t0 <= frac <= t1:
            local = (frac - t0) / (t1 - t0) if t1 > t0 else 0.0
            ease = 0.5 - 0.5 * math.cos(math.pi * local)
            hip = h0 + (h1 - h0) * ease
            knee = k0 + (k1 - k0) * ease
            ankle = a0 + (a1 - a0) * ease
            return hip, knee, ankle
    return GAIT_KEYFRAMES[-1][1:]


INSTRUCTIONS = """
Unitree G1 keyboard teleop - human-like walking gait
-------------------------------------------------
Base movement (cruise style: press once, keeps moving until changed):
  w / s : walk forward / backward
  a / d : turn left / right (arcs while walking too)
  q / e : sidestep left / right
  space : stop all base motion
  z / x : crouch down / stand up

Fine-tune a joint on top of the walking gait (e.g. head, arms):
  [ / ] : select previous / next joint
  j / l : decrease / increase the selected joint's trim
  r     : clear all manual joint trims

Autonomous navigation: this node also listens on /cmd_vel - run the
navigate_to_goal node and send it a goal (RViz "2D Nav Goal" tool, or publish
geometry_msgs/PoseStamped on /goal_pose) and it will walk itself there,
planning a path around obstacles seen by the LiDAR. Pressing any movement key
here takes manual control back at any time.

Pick-and-place: this node also plays a right-arm reach/carry animation on
/manipulation_cmd ("pick"/"place"), driven automatically by navigate_to_goal
when it walks the robot up to a box or a drop point.

CTRL-C to quit
-------------------------------------------------
"""


def get_key(settings, timeout=0.05):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class HumanoidTeleop(Node):

    def __init__(self):
        super().__init__('humanoid_keyboard_teleop')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Manual per-joint trim, adjustable with [ / ] and j / l. The walking
        # gait (computed each tick) is added on top of this for leg/arm/waist
        # joints, so the two controls never fight each other.
        self.joint_trim = {name: 0.0 for name in JOINT_NAMES}
        self.selected_index = 0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.base_height = 0.79
        self.min_height = 0.55
        self.max_height = 0.79

        # Persistent cruise-control velocities (body frame). Driven either by
        # keyboard keys or by a navigation node publishing cmd_vel.
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self.height_step = 0.02
        self.joint_step = 0.1

        self.gait_phase = 0.0
        self.gait_blend = 0.0  # 0 = standing still, 1 = full walking gait

        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.on_cmd_vel, 10)

        # Pick-and-place full-body animation state.
        self.manip_action = None       # None | 'pick' | 'place' - animation in progress
        self.manip_elapsed = 0.0
        self.manip_done_sent = True
        self.manip_event_sent = True   # grasp/release event already fired for this action
        self.manip_hold_pose = 'rest'  # pose held once an animation finishes
        self.holding_box = False       # suppresses the gait's arm swing while True
        self.manip_cmd_sub = self.create_subscription(
            String, 'manipulation_cmd', self.on_manipulation_cmd, 10)
        self.manip_done_pub = self.create_publisher(Bool, 'manipulation_done', 10)
        self.manip_event_pub = self.create_publisher(String, 'manipulation_event', 10)

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(0.02, self.timer_callback)

    # --- input handlers ----------------------------------------------------

    def set_velocity(self, key):
        if key == 'w':
            self.vx = WALK_SPEED
        elif key == 's':
            self.vx = -WALK_SPEED
        elif key == 'a':
            self.wz = TURN_SPEED
        elif key == 'd':
            self.wz = -TURN_SPEED
        elif key == 'q':
            self.vy = STRAFE_SPEED
        elif key == 'e':
            self.vy = -STRAFE_SPEED
        elif key == ' ':
            self.vx = 0.0
            self.vy = 0.0
            self.wz = 0.0
        elif key == 'z':
            self.base_height = max(self.min_height, self.base_height - self.height_step)
        elif key == 'x':
            self.base_height = min(self.max_height, self.base_height + self.height_step)

    def on_cmd_vel(self, msg):
        # A navigation node (or any other external controller) drives the
        # robot exactly the same way the keyboard does - same vx/vy/wz, same
        # walking gait underneath. Pressing a keyboard key simply overwrites
        # these again next tick, so manual input always takes control back.
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z

    def on_manipulation_cmd(self, msg):
        if msg.data not in ('pick', 'place'):
            return
        self.manip_action = msg.data
        self.manip_elapsed = 0.0
        self.manip_done_sent = False
        self.manip_event_sent = False

    def select_joint(self, key):
        if key == '[':
            self.selected_index = (self.selected_index - 1) % len(JOINT_NAMES)
        else:
            self.selected_index = (self.selected_index + 1) % len(JOINT_NAMES)
        self.get_logger().info(f'Selected joint: {JOINT_NAMES[self.selected_index]}')

    def adjust_joint(self, key):
        name = JOINT_NAMES[self.selected_index]
        lower, upper = JOINT_LIMITS[name]
        delta = self.joint_step if key == 'l' else -self.joint_step
        new_val = self.joint_trim[name] + delta
        self.joint_trim[name] = max(lower, min(upper, new_val))

    def reset_joints(self):
        for name in JOINT_NAMES:
            self.joint_trim[name] = 0.0

    # --- per-tick update -----------------------------------------------------

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.0 or dt > 0.5:
            dt = 0.02

        self.integrate_base(dt)
        offsets, pelvis_bob = self.step_gait(dt)
        manip_offsets, manip_height_delta = self.compute_manip_offsets(dt)
        for name, value in manip_offsets.items():
            offsets[name] = offsets.get(name, 0.0) + value

        self.publish_tf(pelvis_bob, manip_height_delta)
        self.publish_joint_states(offsets)

    def integrate_base(self, dt):
        self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
        self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt
        self.yaw += self.wz * dt

    def step_gait(self, dt):
        gait_speed = math.hypot(self.vx, self.vy) + abs(self.wz) * TURN_TO_GAIT_SPEED
        walking = gait_speed > 1e-3

        target_blend = 1.0 if walking else 0.0
        blend_delta = GAIT_BLEND_RATE * dt
        if self.gait_blend < target_blend:
            self.gait_blend = min(target_blend, self.gait_blend + blend_delta)
        else:
            self.gait_blend = max(target_blend, self.gait_blend - blend_delta)

        if walking:
            phase_rate = 2.0 * math.pi * gait_speed / STRIDE_LENGTH
            self.gait_phase = (self.gait_phase + phase_rate * dt) % (2.0 * math.pi)

        phase = self.gait_phase
        scale = self.gait_blend

        left_phase = phase
        right_phase = phase + math.pi

        def leg_offsets(leg_phase):
            hip_pitch, knee, ankle_pitch = _gait_keyframe_lookup(leg_phase / (2.0 * math.pi))
            # The keyframe table's angles are anatomical (positive = flexion,
            # thigh forward). Verified numerically against this URDF's actual
            # joint chain: this hip_pitch joint's positive direction swings
            # the leg backward, the opposite of that anatomical convention -
            # so it has to be negated here to actually swing forward when the
            # table says "flexion".
            hip_pitch = -hip_pitch
            hip_roll = HIP_ROLL_SWAY * math.sin(leg_phase)
            return hip_pitch, hip_roll, knee, ankle_pitch

        l_hip_pitch, l_hip_roll, l_knee, l_ankle = leg_offsets(left_phase)
        r_hip_pitch, r_hip_roll, r_knee, r_ankle = leg_offsets(right_phase)

        offsets = {name: 0.0 for name in JOINT_NAMES}
        offsets['left_hip_pitch_joint'] = l_hip_pitch * scale
        offsets['left_hip_roll_joint'] = l_hip_roll * scale
        offsets['left_knee_joint'] = l_knee * scale
        offsets['left_ankle_pitch_joint'] = l_ankle * scale
        offsets['right_hip_pitch_joint'] = r_hip_pitch * scale
        offsets['right_hip_roll_joint'] = r_hip_roll * scale
        offsets['right_knee_joint'] = r_knee * scale
        offsets['right_ankle_pitch_joint'] = r_ankle * scale

        # Opposite-arm-to-leg counter swing, plus a small counter-rotation of
        # the waist, the way a person naturally balances while walking.
        # (Negated for the same reason as hip_pitch above - this shoulder
        # joint's positive direction is also anatomically backward.)
        # Both arms are reserved for carrying while a pick/place animation is
        # running or a box is held in both hands - the manipulation pose
        # (added in timer_callback) would otherwise fight the gait's
        # counter-swing on top of it.
        manipulating = self.manip_action is not None or self.holding_box
        if not manipulating:
            offsets['left_shoulder_pitch_joint'] = -ARM_SWING * math.sin(right_phase) * scale
            offsets['left_elbow_joint'] = ELBOW_BASE_BEND * scale
            offsets['right_shoulder_pitch_joint'] = -ARM_SWING * math.sin(left_phase) * scale
            offsets['right_elbow_joint'] = ELBOW_BASE_BEND * scale
        else:
            offsets['left_shoulder_pitch_joint'] = 0.0
            offsets['left_elbow_joint'] = 0.0
            offsets['right_shoulder_pitch_joint'] = 0.0
            offsets['right_elbow_joint'] = 0.0
        offsets['waist_yaw_joint'] = WAIST_YAW_COUNTER * math.sin(left_phase) * scale

        pelvis_bob = PELVIS_BOB_AMPLITUDE * abs(math.sin(phase)) * scale
        return offsets, pelvis_bob

    def compute_manip_offsets(self, dt):
        """Full-body pick/place animation: crouch (both legs + pelvis height)
        -> reach with both arms -> grasp/release -> stand back up. Runs once
        per /manipulation_cmd, then holds its end pose (carrying, or back at
        rest - both standing tall, legs neutral) until the next command,
        independent of the walking gait above."""
        if self.manip_action is not None:
            self.manip_elapsed += dt
            duration = PICK_DURATION if self.manip_action == 'pick' else PLACE_DURATION
            progress = min(1.0, self.manip_elapsed / duration)
            timeline = PICK_TIMELINE if self.manip_action == 'pick' else PLACE_TIMELINE
            offsets, height_delta = _manip_timeline_lookup(timeline, progress)

            event_fraction = GRASP_FRACTION if self.manip_action == 'pick' else RELEASE_FRACTION
            if progress >= event_fraction and not self.manip_event_sent:
                self.manip_event_sent = True
                event = 'grasped' if self.manip_action == 'pick' else 'released'
                self.manip_event_pub.publish(String(data=event))

            if progress >= 1.0 and not self.manip_done_sent:
                self.manip_done_sent = True
                self.holding_box = (self.manip_action == 'pick')
                self.manip_hold_pose = 'carry' if self.manip_action == 'pick' else 'rest'
                self.manip_action = None
                self.manip_done_pub.publish(Bool(data=True))
            return offsets, height_delta
        if self.manip_hold_pose == 'carry':
            return dict(MANIP_POSES['carry']), 0.0
        return {}, 0.0

    # --- publishing ----------------------------------------------------------

    def publish_tf(self, pelvis_bob, manip_height_delta=0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = BASE_FRAME
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = max(0.3, self.base_height + pelvis_bob + manip_height_delta)
        t.transform.rotation.z = math.sin(self.yaw / 2.0)
        t.transform.rotation.w = math.cos(self.yaw / 2.0)
        self.tf_broadcaster.sendTransform(t)

    def publish_joint_states(self, gait_offsets):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        positions = []
        for name in JOINT_NAMES:
            lower, upper = JOINT_LIMITS[name]
            value = REST_POSE.get(name, 0.0) + self.joint_trim[name] + gait_offsets[name]
            positions.append(max(lower, min(upper, value)))
        msg.position = positions
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanoidTeleop()
    settings = termios.tcgetattr(sys.stdin)
    print(INSTRUCTIONS)
    try:
        while rclpy.ok():
            key = get_key(settings, timeout=0.02)
            if key:
                if key in ('w', 's', 'a', 'd', 'q', 'e', 'z', 'x', ' '):
                    node.set_velocity(key)
                elif key in ('[', ']'):
                    node.select_joint(key)
                elif key == 'r':
                    node.reset_joints()
                elif key in ('j', 'l'):
                    node.adjust_joint(key)
                elif key == '\x03':
                    break
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
