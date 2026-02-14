#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from rclpy.qos import qos_profile_sensor_data # <--- THE MAGIC FIX

class DepthMissionController(Node):
    def __init__(self):
        super().__init__('depth_mission_controller')
        self.get_logger().info('--- PHASE 14: QOS NUCLEAR FIX ---')
        self.bridge = CvBridge()
        
        self.latest_depth = None
        self.pitch = 0.0
        self.state = "DRIVE_TO_EDGE"
        self.descent_start_time = 0
        self.ground_counter = 0

        # Apply the standard Sensor Data QoS profile (Best Effort + Volatile)
        self.create_subscription(Imu, '/imu', self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth_image', self.depth_cb, qos_profile_sensor_data)
        
        self.vel_pub_main = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vel_pub_front = self.create_publisher(Twist, '/cmd_vel_front', 10)
        
        self.timer = self.create_timer(0.05, self.run_mission)
        self.get_logger().info("Node initialized. Listening for Depth...")

    def imu_cb(self, msg):
        q = msg.orientation
        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)

    def depth_cb(self, msg):
        # DEBUG: Print once to prove we are getting data
        if self.latest_depth is None:
            self.get_logger().info(f"FIRST DEPTH RECEIVED! Encoding: {msg.encoding}")

        try:
            # Handle standard float depth buffers
            if msg.encoding == "32FC1":
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            # Handle potential 16-bit integers (sometimes Gazebo defaults to this)
            elif msg.encoding == "16UC1":
                raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.latest_depth = raw.astype(np.float32) / 1000.0 # Convert mm to m
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Fail: {e}")

    def run_mission(self):
        # Just return if no data, don't spam logs
        if self.latest_depth is None:
            return

        h, w = self.latest_depth.shape
        center_patch = self.latest_depth[h//2-30:h//2+30, w//2-30:w//2+30]
        valid = center_patch[np.isfinite(center_patch)]
        dist = np.mean(valid) if len(valid) > 0 else 99.9
        
        cmd = Twist()

        if self.state == "DRIVE_TO_EDGE":
            cmd.linear.x = 0.15
            if self.pitch < -0.08:
                self.state = "DESCEND_6WD"
                self.descent_start_time = time.time()
                self.get_logger().info("Edge Detected. Descending.")
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(cmd)

        elif self.state == "DESCEND_6WD":
            if (time.time() % 1.0) < 0.2: cmd.linear.x = 0.05
            else: cmd.linear.x = 0.0

            if (time.time() - self.descent_start_time) > 1.0 and abs(self.pitch) < 0.12:
                self.ground_counter += 1
                if self.ground_counter > 2:
                    self.state = "ROTATE_TO_FIND"
                    self.get_logger().info("Touchdown. Searching for Ramp.")
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(cmd)

        elif self.state == "ROTATE_TO_FIND":
            if 0.5 < dist < 1.8:
                self.state = "CLIMB"
                self.get_logger().info(f"Ramp Found at {dist:.2f}m. CLIMBING!")
            else:
                cmd.angular.z = -0.4
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(Twist())

        elif self.state == "CLIMB":
            cmd.linear.x = 0.8
            if dist < 0.6: self.state = "FINISHED"
            self.vel_pub_main.publish(cmd)
            self.vel_pub_front.publish(cmd)

        elif self.state == "FINISHED":
            self.vel_pub_main.publish(Twist())
            self.vel_pub_front.publish(Twist())

        # Visualization
        vis = np.clip(self.latest_depth, 0, 5) / 5.0
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.putText(vis, f"STATE: {self.state}", (20, 40), 0, 0.7, (255,255,255), 2)
        cv2.imshow("Depth View", vis)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DepthMissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()
