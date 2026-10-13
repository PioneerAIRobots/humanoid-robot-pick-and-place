import heapq
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener, TransformException

WORLD_FRAME = 'odom'
ROBOT_FRAME = 'pelvis'

# --- Occupancy grid --------------------------------------------------------
# A simple local grid built fresh from each LiDAR scan (the simulated
# obstacles are static, so there's no need to accumulate/fuse scans over
# time - the latest scan already has everything the sensor can see).
GRID_RESOLUTION = 0.1          # metres per cell
GRID_SIZE_M = 20.0             # grid covers this many metres on a side
GRID_CELLS = int(GRID_SIZE_M / GRID_RESOLUTION)
GRID_ORIGIN_X = -GRID_SIZE_M / 2.0
GRID_ORIGIN_Y = -GRID_SIZE_M / 2.0
OBSTACLE_MIN_HEIGHT = 0.06     # metres above ground - excludes ground-plane returns
# How far the obstacles are inflated so the planner keeps the robot's body
# clear of them, not just its centre point.
ROBOT_RADIUS = 0.35
INFLATE_CELLS = max(1, math.ceil(ROBOT_RADIUS / GRID_RESOLUTION))

# --- Path following ---------------------------------------------------------
WALK_SPEED = 0.35
TURN_SPEED = 0.6
WAYPOINT_TOLERANCE = 0.3        # metres - close enough to advance to the next waypoint
GOAL_TOLERANCE = 0.2            # metres - close enough to call the goal reached
TURN_IN_PLACE_THRESHOLD = 0.6   # radians - heading error above this: stop and turn first
HEADING_KP = 1.2
REPLAN_PERIOD = 2.0             # seconds between replans while navigating
CONTROL_PERIOD = 0.1            # seconds between control-loop ticks


def world_to_cell(x, y):
    cx = int((x - GRID_ORIGIN_X) / GRID_RESOLUTION)
    cy = int((y - GRID_ORIGIN_Y) / GRID_RESOLUTION)
    return cx, cy


def cell_to_world(cx, cy):
    x = GRID_ORIGIN_X + (cx + 0.5) * GRID_RESOLUTION
    y = GRID_ORIGIN_Y + (cy + 0.5) * GRID_RESOLUTION
    return x, y


def in_bounds(cx, cy):
    return 0 <= cx < GRID_CELLS and 0 <= cy < GRID_CELLS


NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


def astar(grid, start_cell, goal_cell):
    """8-connected A* over a 2D occupancy grid (1 = occupied). Returns a list
    of (x, y) world waypoints from start to goal, or None if unreachable."""
    if not in_bounds(*goal_cell) or grid[goal_cell[1]][goal_cell[0]]:
        return None
    if not in_bounds(*start_cell):
        return None

    def heuristic(c):
        return math.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1])

    open_heap = [(heuristic(start_cell), 0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}
    visited = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_cell:
            break
        cx, cy = current
        for dx, dy, cost in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny) or grid[ny][nx]:
                continue
            neighbor = (nx, ny)
            if neighbor in visited:
                continue
            tentative = g + cost
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + heuristic(neighbor), tentative, neighbor))

    if goal_cell not in came_from and goal_cell != start_cell:
        return None

    cell_path = [goal_cell]
    while cell_path[-1] != start_cell:
        cell_path.append(came_from[cell_path[-1]])
    cell_path.reverse()
    return [cell_to_world(cx, cy) for cx, cy in cell_path]


def line_of_sight_clear(grid, p0, p1):
    """True if the straight segment p0->p1 (world coords) crosses no
    occupied cell, sampled at roughly half a cell's resolution."""
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    steps = max(1, int(dist / (GRID_RESOLUTION * 0.5)))
    for i in range(steps + 1):
        f = i / steps
        x = p0[0] + (p1[0] - p0[0]) * f
        y = p0[1] + (p1[1] - p0[1]) * f
        cx, cy = world_to_cell(x, y)
        if not in_bounds(cx, cy) or grid[cy][cx]:
            return False
    return True


def simplify_path(grid, waypoints):
    """Greedy line-of-sight pruning: keep only waypoints where the straight
    line to the furthest-still-visible point would otherwise be blocked."""
    if len(waypoints) <= 2:
        return waypoints
    simplified = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        j = len(waypoints) - 1
        while j > i + 1 and not line_of_sight_clear(grid, waypoints[i], waypoints[j]):
            j -= 1
        simplified.append(waypoints[j])
        i = j
    return simplified


class NavigateToGoal(Node):

    def __init__(self):
        super().__init__('navigate_to_goal')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 1)
        self.grid_pub = self.create_publisher(OccupancyGrid, 'nav_occupancy_grid', 1)

        self.create_subscription(PointCloud2, 'lidar_points', self.on_lidar_points, 5)
        self.create_subscription(PoseStamped, 'goal_pose', self.on_goal_pose, 10)

        self.grid = [[0] * GRID_CELLS for _ in range(GRID_CELLS)]
        self.goal = None           # (x, y) in odom frame
        self.waypoints = []        # remaining path waypoints, [0] is the current target
        self.last_plan_time = None

        self.create_timer(CONTROL_PERIOD, self.control_tick)
        self.get_logger().info(
            'navigate_to_goal ready - send a goal on /goal_pose '
            '(e.g. RViz "2D Nav Goal") to start navigating.')

    # --- sensing -------------------------------------------------------------

    def on_lidar_points(self, msg):
        grid = [[0] * GRID_CELLS for _ in range(GRID_CELLS)]
        for px, py, pz in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            pz = float(pz)
            if pz < OBSTACLE_MIN_HEIGHT:
                continue  # ground-plane return
            cx, cy = world_to_cell(float(px), float(py))
            if in_bounds(cx, cy):
                grid[cy][cx] = 1

        if INFLATE_CELLS > 0:
            occupied = [(x, y) for y in range(GRID_CELLS) for x in range(GRID_CELLS) if grid[y][x]]
            for cx, cy in occupied:
                for dy in range(-INFLATE_CELLS, INFLATE_CELLS + 1):
                    for dx in range(-INFLATE_CELLS, INFLATE_CELLS + 1):
                        if dx * dx + dy * dy <= INFLATE_CELLS * INFLATE_CELLS:
                            nx, ny = cx + dx, cy + dy
                            if in_bounds(nx, ny):
                                grid[ny][nx] = 1
        self.grid = grid
        self.publish_grid()

    def on_goal_pose(self, msg):
        if msg.header.frame_id not in ('', WORLD_FRAME):
            self.get_logger().warn(
                f'Ignoring goal in frame "{msg.header.frame_id}" - expected "{WORLD_FRAME}".')
            return
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
        self.replan()

    # --- planning --------------------------------------------------------------

    def get_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, ROBOT_FRAME, rclpy.time.Time(), timeout=Duration(seconds=0.1))
        except TransformException:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        yaw = 2.0 * math.atan2(r.z, r.w)
        return t.x, t.y, yaw

    def replan(self):
        if self.goal is None:
            return
        pose = self.get_robot_pose()
        if pose is None:
            self.get_logger().warn('No robot pose yet (TF not ready) - cannot plan.')
            return
        x, y, _ = pose
        start_cell = world_to_cell(x, y)
        goal_cell = world_to_cell(*self.goal)

        path = astar(self.grid, start_cell, goal_cell)
        self.last_plan_time = self.get_clock().now()
        if path is None:
            self.get_logger().warn('No path found to goal (blocked or out of bounds).')
            self.waypoints = []
            self.publish_path([])
            return

        path = simplify_path(self.grid, path)
        # Drop the start point itself; keep the rest (including the goal) as
        # the waypoints to walk through in order.
        self.waypoints = path[1:] if len(path) > 1 else path
        self.publish_path(path)
        self.get_logger().info(f'Planned path with {len(self.waypoints)} waypoint(s).')

    # --- control ---------------------------------------------------------------

    def control_tick(self):
        if self.goal is None:
            return

        now = self.get_clock().now()
        if (self.last_plan_time is None or
                (now - self.last_plan_time).nanoseconds / 1e9 > REPLAN_PERIOD):
            self.replan()

        pose = self.get_robot_pose()
        if pose is None:
            return
        x, y, yaw = pose

        if not self.waypoints:
            dist_to_goal = math.hypot(self.goal[0] - x, self.goal[1] - y)
            if dist_to_goal <= GOAL_TOLERANCE:
                return  # already there, nothing to do
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        target = self.waypoints[0]
        dist = math.hypot(target[0] - x, target[1] - y)
        tolerance = GOAL_TOLERANCE if len(self.waypoints) == 1 else WAYPOINT_TOLERANCE
        if dist <= tolerance:
            self.waypoints.pop(0)
            if not self.waypoints:
                self.get_logger().info('Goal reached.')
                self.publish_cmd(0.0, 0.0, 0.0)
                return
            target = self.waypoints[0]
            dist = math.hypot(target[0] - x, target[1] - y)

        desired_heading = math.atan2(target[1] - y, target[0] - x)
        heading_error = math.atan2(math.sin(desired_heading - yaw), math.cos(desired_heading - yaw))

        wz = max(-TURN_SPEED, min(TURN_SPEED, HEADING_KP * heading_error))
        if abs(heading_error) > TURN_IN_PLACE_THRESHOLD:
            vx = 0.0
        else:
            slow_factor = 1.0 - 0.6 * (abs(heading_error) / TURN_IN_PLACE_THRESHOLD)
            vx = WALK_SPEED * max(0.3, slow_factor)
            vx = min(vx, dist / CONTROL_PERIOD * 0.5)  # don't overshoot a close waypoint
            vx = max(vx, 0.0)

        self.publish_cmd(vx, 0.0, wz)

    def publish_cmd(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self.cmd_pub.publish(msg)

    # --- visualization -----------------------------------------------------

    def publish_path(self, waypoints):
        msg = Path()
        msg.header.frame_id = WORLD_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        for wx, wy in waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = Point(x=wx, y=wy, z=0.0)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def publish_grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id = WORLD_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = GRID_RESOLUTION
        msg.info.width = GRID_CELLS
        msg.info.height = GRID_CELLS
        msg.info.origin.position.x = GRID_ORIGIN_X
        msg.info.origin.position.y = GRID_ORIGIN_Y
        msg.info.origin.orientation.w = 1.0
        data = [0] * (GRID_CELLS * GRID_CELLS)
        for cy in range(GRID_CELLS):
            row = self.grid[cy]
            base = cy * GRID_CELLS
            for cx in range(GRID_CELLS):
                if row[cx]:
                    data[base + cx] = 100
        msg.data = data
        self.grid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NavigateToGoal()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
