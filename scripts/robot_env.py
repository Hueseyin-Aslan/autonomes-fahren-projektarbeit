#!/usr/bin/env python3
import subprocess
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


class RobotEnv(gym.Env):
    def __init__(self):
        super().__init__()

        rclpy.init()
        self.node = Node('robot_env')

        self.lidar_data = np.full(24, 3.5, dtype=np.float32)
        self.odom_data = None
        self.collided = False
        self.lidar_fresh = False
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        self.node.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
        self.node.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)
        self.cmd_pub = self.node.create_publisher(
            TwistStamped, '/cmd_vel', 10)

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.5], dtype=np.float32),
            high=np.array([0.22, 1.5], dtype=np.float32),
        )

        # 24 LiDAR-Sektoren + x + y + sin/cos(Zielwinkel) + lin.Geschwindigkeit + ang.Geschwindigkeit = 30
        self.observation_space = spaces.Box(
            low=np.array([0.0]*24 + [-30.0, -3.0, -1.0, -1.0, -0.3, -3.0], dtype=np.float32),
            high=np.array([3.5]*24 + [30.0, 3.0, 1.0, 1.0, 0.3, 3.0], dtype=np.float32),
        )

        self.goal_x = -3.0
        self.goal_y = 0.0
        self.goal_radius = 0.35
        self.start_x = -7.0
        self.max_steps = 1200
        self.current_step = 0
        self.prev_distance_to_goal = None

        self.odom_offset_x = 0.0
        self.odom_offset_y = 0.0
        self.burger_id = None

        # Stillstands-Erkennung: Position alle ~1s (10 Schritte) speichern
        self.STALL_WINDOW_STEPS = 150   # ca. 15 Sekunden bei 0.1s/Schritt
        self.STALL_MIN_PROGRESS = 0.15  # Meter
        self.position_history = []
        self.stuck_position_history = []
        self.stuck_distance_history = []

        self.ACTION_SMOOTHING = 0.25  # 0 = keine Glättung, 1 = nur alte Aktion

        # Stuck-Erkennung (Zoegern beenden)
        self.STUCK_WINDOW = 100  # ca. 6 Sekunden bei 0.1s/Schritt
        self.STUCK_MIN_MOVE = 0.10
        self.STUCK_MIN_GOAL_PROGRESS = 0.03
        self.stuck_position_history = []
        self.stuck_distance_history = []

    def _scan_callback(self, msg):
        self.lidar_fresh = True
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=0.0)
        n = len(ranges)
        sector_size = n / 24.0
        sectors = np.zeros(24, dtype=np.float32)
        for i in range(24):
            start = int(i * sector_size)
            end = int((i + 1) * sector_size)
            if end <= start:
                end = start + 1
            sectors[i] = np.min(ranges[start:end])
        self.lidar_data = sectors
        if np.min(self.lidar_data) < 0.15:
            self.collided = True

    def _odom_callback(self, msg):
        self.odom_data = msg

    def _get_yaw(self):
        if self.odom_data is None:
            return 0.0
        q = self.odom_data.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _get_obs(self):
        if self.odom_data is not None:
            raw_x = self.odom_data.pose.pose.position.x
            raw_y = self.odom_data.pose.pose.position.y
            x = self.start_x + (raw_x - self.odom_offset_x)
            y = 0.0 + (raw_y - self.odom_offset_y)
            lin_vel = self.odom_data.twist.twist.linear.x
            ang_vel = self.odom_data.twist.twist.angular.z
        else:
            x, y = self.start_x, 0.0
            lin_vel, ang_vel = 0.0, 0.0

        yaw = self._get_yaw()
        angle_to_goal = math.atan2(self.goal_y - y, self.goal_x - x)
        relative_angle = angle_to_goal - yaw
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))

        extras = np.array(
            [x, y, math.sin(relative_angle), math.cos(relative_angle), lin_vel, ang_vel],
            dtype=np.float32
        )
        return np.concatenate([self.lidar_data, extras]).astype(np.float32)

    def _spin_briefly(self, duration=0.1):
        end_time = time.time() + duration
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def _get_burger_entity_id(self):
        cmd = 'gz topic -e -t /world/lawn/pose/info -n 1'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if 'name: "burger"' in line:
                for j in range(i + 1, min(i + 3, len(lines))):
                    if 'id:' in lines[j]:
                        return int(lines[j].split(':')[1].strip())
        return None

    def _reset_position(self, x=-7.0, y=0.0, z=0.05):
        id_part = f'id: {self.burger_id}, ' if self.burger_id is not None else ''
        cmd = (
            'gz service -s /world/lawn/set_pose '
            '--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean '
            '--timeout 5000 '
            f'--req \'name: "burger", {id_part}position: {{x: {x}, y: {y}, z: {z}}}, '
            'orientation: {x: 0, y: 0, z: 0, w: 1}\''
        )
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            print("RESET: subprocess Timeout")
            return False
        ok = result.returncode == 0 and "true" in result.stdout.lower()
        if not ok:
            print(f"RESET RESULT: {result.stdout.strip()} | ERR: {result.stderr.strip()}")
        return ok

    def set_goal(self, new_goal_x):
        self.goal_x = new_goal_x

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.burger_id = self._get_burger_entity_id()

        self._send_cmd(0.0, 0.0)
        self._spin_briefly(0.2)  # sicherstellen dass der Roboter wirklich anhaelt

        self.collided = False
        self.current_step = 0
        self.odom_data = None
        self.lidar_fresh = False
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.position_history = []
        self.stuck_position_history = []
        self.stuck_distance_history = []

        reset_ok = False
        for attempt in range(3):
            success = self._reset_position(x=self.start_x, y=0.0)
            self._spin_briefly(0.3)

            if success and self.odom_data is not None:
                actual_x = self.odom_data.pose.pose.position.x
                actual_y = self.odom_data.pose.pose.position.y
                if attempt == 0 or (abs(actual_x) < 0.5 and abs(actual_y) < 0.5):
                    reset_ok = True
                    break
            print(f"RESET-Versuch {attempt + 1} unsicher, wiederhole...")
            self._spin_briefly(0.2)

        if not reset_ok:
            print("WARNUNG: Reset nach 3 Versuchen nicht sicher bestaetigt.")

        # Auf mindestens einen frischen LiDAR-Scan nach dem Reset warten
        self.lidar_fresh = False
        wait_start = time.time()
        while not self.lidar_fresh and time.time() - wait_start < 1.0:
            self._spin_briefly(0.05)

        # Falls der Teleport-Uebergang faelschlich eine Kollision ausgeloest hat,
        # hier zuruecksetzen (Uebergangsphase, keine echte Kollision)
        self.collided = False

        if self.odom_data is not None:
            self.odom_offset_x = self.odom_data.pose.pose.position.x
            self.odom_offset_y = self.odom_data.pose.pose.position.y
        else:
            self.odom_offset_x = 0.0
            self.odom_offset_y = 0.0

        self.prev_distance_to_goal = self.goal_x - self.start_x
        self.max_x_reached = self.start_x
        obs = self._get_obs()
        return obs, {}

    def _send_cmd(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def step(self, action):
        raw_action = np.array(action, dtype=np.float32)
        smoothed = self.ACTION_SMOOTHING * self.prev_action + (1 - self.ACTION_SMOOTHING) * raw_action
        self.prev_action = smoothed
        linear_x, angular_z = float(smoothed[0]), float(smoothed[1])

        self._send_cmd(linear_x, angular_z)
        self._spin_briefly(0.1)

        self.current_step += 1
        obs = self._get_obs()
        current_x = obs[24]

        distance_to_goal = self.goal_x - current_x
        progress = self.prev_distance_to_goal - distance_to_goal
        self.prev_distance_to_goal = distance_to_goal

        reward = -0.02
        reward += progress * 5.0

        terminated = False
        stuck = False

        self.stuck_position_history.append((obs[24], obs[25]))
        self.stuck_distance_history.append(distance_to_goal)
        if len(self.stuck_position_history) > self.STUCK_WINDOW:
            self.stuck_position_history.pop(0)
            self.stuck_distance_history.pop(0)
            px0, py0 = self.stuck_position_history[0]
            px1, py1 = self.stuck_position_history[-1]
            moved = math.hypot(px1 - px0, py1 - py0)
            goal_progress = self.stuck_distance_history[0] - self.stuck_distance_history[-1]
            if moved < self.STUCK_MIN_MOVE and goal_progress < self.STUCK_MIN_GOAL_PROGRESS:
                stuck = True

        if current_x > self.max_x_reached:
            self.max_x_reached = current_x

        if self.collided:
            reward = -100.0
            terminated = True
        elif current_x >= self.goal_x - self.goal_radius:
            reward = 150.0
            terminated = True
        elif stuck:
            reward = -40.0
            terminated = True

        truncated = self.current_step >= self.max_steps

        info = {}
        if terminated or truncated:
            info = {
                "max_x": self.max_x_reached,
                "goal_reached": 1.0 if current_x >= self.goal_x - self.goal_radius else 0.0,
                "collided": 1.0 if self.collided else 0.0,
                "stuck": 1.0 if stuck else 0.0,
            }

        return obs, reward, terminated, truncated, info

    def close(self):
        self._send_cmd(0.0, 0.0)
        self.node.destroy_node()
        rclpy.shutdown()


