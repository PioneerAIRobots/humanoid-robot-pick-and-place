import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

LIDAR_FRAME = 'lidar32_link'
WORLD_FRAME = 'odom'

# Mechanical 32-beam spinning LiDAR. Tilted further down than a typical
# vehicle-roof Velodyne mount: this one is mounted above the head specifically
# to see close, low obstacles right in front of the robot's feet, which needs
# a much steeper downward tilt than a car's forward-looking placement.
NUM_CHANNELS = 32
VERTICAL_FOV_MIN = math.radians(-60.0)
VERTICAL_FOV_MAX = math.radians(15.0)
HORIZONTAL_STEP_DEG = 2.0
MAX_RANGE = 10.0
MIN_RANGE = 0.2
SCAN_RATE_HZ = 5.0

# Static box obstacles placed in a zigzag/slalom course down the walking
# path, in the odom (world) frame: (center_x, center_y, center_z, size_x,
# size_y, size_z) - alternating left/right so a straight-line path would hit
# one of them, forcing the planner to weave around each in turn.
OBSTACLES = [
    (1.6, 0.55, 0.2, 0.4, 0.4, 0.4),
    (2.9, -0.55, 0.2, 0.4, 0.4, 0.4),
    (4.2, 0.55, 0.2, 0.4, 0.4, 0.4),
    (5.5, -0.55, 0.2, 0.4, 0.4, 0.4),
    (6.8, 0.55, 0.2, 0.4, 0.4, 0.4),
]


def ray_box_intersection(origin, direction, box):
    cx, cy, cz, sx, sy, sz = box
    box_min = (cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0)
    box_max = (cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0)

    t_min, t_max = 0.0, MAX_RANGE
    for i in range(3):
        o, d = origin[i], direction[i]
        lo, hi = box_min[i], box_max[i]
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
            continue
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min if t_min > MIN_RANGE else None


def ray_ground_intersection(origin, direction):
    if direction[2] >= -1e-6:
        return None
    t = -origin[2] / direction[2]
    return t if MIN_RANGE < t <= MAX_RANGE else None


def quat_rotate(q, v):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # standard quaternion-vector rotation: v' = v + 2*qw*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
    ux, uy, uz = qy * vz - qz * vy, qz * vx - qx * vz, qx * vy - qy * vx
    uux = qy * uz - qz * uy
    uuy = qz * ux - qx * uz
    uuz = qx * uy - qy * ux
    return (
        vx + 2.0 * qw * ux + 2.0 * uux,
        vy + 2.0 * qw * uy + 2.0 * uuy,
        vz + 2.0 * qw * uz + 2.0 * uuz,
    )


class LidarSimulator(Node):

    def __init__(self):
        super().__init__('lidar32_simulator')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cloud_pub = self.create_publisher(PointCloud2, 'lidar_points', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 1)

        self.vertical_angles = [
            VERTICAL_FOV_MIN + i * (VERTICAL_FOV_MAX - VERTICAL_FOV_MIN) / (NUM_CHANNELS - 1)
            for i in range(NUM_CHANNELS)
        ]
        num_h = int(360.0 / HORIZONTAL_STEP_DEG)
        self.horizontal_angles = [math.radians(i * HORIZONTAL_STEP_DEG) for i in range(num_h)]

        self.create_timer(1.0 / SCAN_RATE_HZ, self.scan_callback)
        self.create_timer(1.0, self.publish_obstacle_markers)

    def scan_callback(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, LIDAR_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except TransformException:
            return

        t = tf.transform.translation
        r = tf.transform.rotation
        origin = (t.x, t.y, t.z)
        quat = (r.x, r.y, r.z, r.w)

        points = []
        for v_angle in self.vertical_angles:
            cos_v = math.cos(v_angle)
            sin_v = math.sin(v_angle)
            for h_angle in self.horizontal_angles:
                local_dir = (cos_v * math.cos(h_angle), cos_v * math.sin(h_angle), sin_v)
                direction = quat_rotate(quat, local_dir)

                best_range = None
                ground_range = ray_ground_intersection(origin, direction)
                if ground_range is not None:
                    best_range = ground_range
                for box in OBSTACLES:
                    box_range = ray_box_intersection(origin, direction, box)
                    if box_range is not None and (best_range is None or box_range < best_range):
                        best_range = box_range

                if best_range is None:
                    continue
                points.append((
                    origin[0] + direction[0] * best_range,
                    origin[1] + direction[1] * best_range,
                    origin[2] + direction[2] * best_range,
                ))

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = WORLD_FRAME
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        self.cloud_pub.publish(cloud_msg)

    def publish_obstacle_markers(self):
        array = MarkerArray()
        for i, (cx, cy, cz, sx, sy, sz) in enumerate(OBSTACLES):
            marker = Marker()
            marker.header.frame_id = WORLD_FRAME
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz
            marker.pose.orientation.w = 1.0
            marker.scale.x = sx
            marker.scale.y = sy
            marker.scale.z = sz
            marker.color.r = 0.9
            marker.color.g = 0.35
            marker.color.b = 0.1
            marker.color.a = 0.9
            array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarSimulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
