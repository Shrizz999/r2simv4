#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist
from ros_gz_interfaces.msg import Contacts
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math

# --- STATES ---
STATE_SEARCH = 0      
STATE_APPROACH = 1    
STATE_ALIGNING = 2    
STATE_CLIMBING = 3    
STATE_WAIT = 4 

class LevelManager(Node):
    def __init__(self):
        super().__init__('level_manager')
        self.get_logger().info('--- R2KRISHNA: LOOP FIX (HYSTERESIS + LEVEL LOCK) ---')

        # --- SETTINGS ---
        self.invert_steering = True   
        self.linear_speed = 0.4       
        self.search_speed = 0.5       
        self.steering_gain = 0.015    
        self.max_detection_range = 2.0  

        # --- CLIMB SETTINGS ---
        self.climb_torque = 1.0       
        self.alignment_tolerance = 40 
        self.virtual_bumper_area = 45000 

        # --- TRACKING VARS ---
        self.current_level = 0        
        self.target_level_seen = None 
        self.valid_target_locked = False 
        self.align_start_time = 0.0
        self.bumper_hit = False
        self.current_area = 0
        
        # --- LIDAR VARS ---
        self.lidar_error = 0.0
        self.lidar_available = False
        self.lidar_min_dist = 99.9
        
        # --- WAIT STATE ---
        self.wait_start_time = 0.0
        self.wait_duration = 3.0  # Increased to 3s to allow simulation to stabilize

        # --- SENSORS ---
        self.current_pitch = 0.0
        self.pitch_threshold = 0.12 
        self.is_tilting = False
        self.sonar_dist = 99.9
        self.state = STATE_SEARCH
        self.block_x = None

        # --- COLOR RANGES ---
        self.color_200_lower = np.array([35, 150, 50]); self.color_200_upper = np.array([85, 255, 255])
        self.color_400_lower = np.array([90, 100, 50]); self.color_400_upper = np.array([130, 255, 255])
        self.color_600_lower = np.array([0, 0, 0]); self.color_600_upper = np.array([180, 255, 40])
        self.red1_lower = np.array([0, 120, 50]); self.red1_upper = np.array([10, 255, 255])
        self.red2_lower = np.array([170, 120, 50]); self.red2_upper = np.array([180, 255, 255])
        
        # --- ROS ---
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/ultrasonic', self.sonar_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Contacts, '/bumper_states', self.bumper_callback, 10)
        
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_front = self.create_publisher(Twist, '/cmd_vel_front', 10)

        self.br = CvBridge()
        self.create_timer(0.05, self.control_loop) 

    def drive(self, fwd, rot, engage_climbers=False):
        cmd = Twist()
        cmd.linear.x = float(fwd)
        cmd.angular.z = float(rot)
        self.pub_vel.publish(cmd)
        if engage_climbers: self.pub_front.publish(cmd)
        else: self.pub_front.publish(Twist())

    def bumper_callback(self, msg):
        self.bumper_hit = True if len(msg.contacts) > 0 else False

    def sonar_callback(self, msg):
        valid = [r for r in msg.ranges if not np.isinf(r) and not np.isnan(r)]
        if len(valid) > 0: self.sonar_dist = min(valid)

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        count = len(ranges)
        mid_idx = count // 2
        fov_idx = int(count * (60.0 / 360.0)) // 2 
        start = max(0, mid_idx - fov_idx)
        end = min(count, mid_idx + fov_idx)
        front_ranges = ranges[start:end]
        valid_mask = (front_ranges < 2.5) & (front_ranges > 0.1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 5: 
            min_idx = valid_indices[0]
            max_idx = valid_indices[-1]
            span_ratio = (max_idx - min_idx) / len(front_ranges)
            if span_ratio > 0.4:
                self.lidar_available = False
                return
            min_local_idx = valid_indices[np.argmin(front_ranges[valid_indices])]
            global_idx = start + min_local_idx
            angle_per_idx = (msg.angle_max - msg.angle_min) / count
            target_angle = msg.angle_min + (global_idx * angle_per_idx)
            self.lidar_error = target_angle
            self.lidar_min_dist = front_ranges[min_local_idx]
            self.lidar_available = True
        else:
            self.lidar_available = False
            self.lidar_error = 0.0

    def imu_callback(self, msg):
        q = msg.orientation
        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.current_pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)

    def get_contour_data(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 1000: 
                x, y, w, h = cv2.boundingRect(largest)
                aspect_ratio = float(w)/h
                if aspect_ratio < 1.8: 
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        return int(M["m10"] / M["m00"]), area
        return None, 0

    def image_callback(self, msg):
        try: frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        except: return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape
        center_x = width // 2

        # Ignore visual data during Wait/Hysteresis state
        if self.state == STATE_WAIT:
            cv2.imshow("Main Vision", frame); cv2.waitKey(1)
            return

        x_200, a_200 = self.get_contour_data(cv2.inRange(hsv, self.color_200_lower, self.color_200_upper))
        x_400, a_400 = self.get_contour_data(cv2.inRange(hsv, self.color_400_lower, self.color_400_upper))
        x_600, a_600 = self.get_contour_data(cv2.inRange(hsv, self.color_600_lower, self.color_600_upper))
        x_red, a_red = self.get_contour_data(cv2.inRange(hsv, self.red1_lower, self.red1_upper) + cv2.inRange(hsv, self.red2_lower, self.red2_upper))

        self.valid_target_locked = False; self.block_x = None; self.target_level_seen = None; self.current_area = 0
        
        # Only look for the NEXT level
        target_height = self.current_level + 200

        if target_height == 200 and x_200: self.block_x = x_200; self.target_level_seen = 200; self.current_area = a_200
        elif target_height == 400 and x_400: self.block_x = x_400; self.target_level_seen = 400; self.current_area = a_400
        elif target_height == 600 and x_600: self.block_x = x_600; self.target_level_seen = 600; self.current_area = a_600
        elif target_height > 600 and x_red: self.block_x = x_red; self.target_level_seen = 999; self.current_area = a_red

        if self.block_x is not None:
            if self.lidar_available and self.lidar_min_dist > self.max_detection_range:
                self.valid_target_locked = False
                self.block_x = None
            else:
                self.valid_target_locked = True
                cv2.circle(frame, (self.block_x, height//2), 10, (0, 255, 0), 3)

        cv2.imshow("Main Vision", frame); cv2.waitKey(1)

    def control_loop(self):
        if self.state == STATE_WAIT:
            # Pivot Clockwise to look away from the block we just finished
            self.drive(0.0, -self.search_speed, False)
            if time.time() - self.wait_start_time > self.wait_duration:
                self.state = STATE_SEARCH; self.is_tilting = False
            return

        if self.state == STATE_CLIMBING:
            self.drive(self.climb_torque, 0.0, True) 
            if abs(self.current_pitch) > self.pitch_threshold: self.is_tilting = True
            # Level out detection
            if self.is_tilting and abs(self.current_pitch) < 0.05:
                self.current_level = self.target_level_seen
                self.get_logger().info(f"✅ LEVEL COMPLETE: {self.current_level}. Switching to Wait state.")
                self.state = STATE_WAIT; self.wait_start_time = time.time()
            return

        if self.state == STATE_ALIGNING:
            if not self.valid_target_locked: 
                self.drive(0.0, 0.0, False) 
                self.state = STATE_SEARCH; return
            
            err = 320 - self.block_x
            if self.bumper_hit or (abs(err) < self.alignment_tolerance and self.current_area > self.virtual_bumper_area): 
                self.state = STATE_CLIMBING; return
            
            if self.lidar_available and self.lidar_min_dist < 1.2:
                raw_turn = self.lidar_error * 1.8 
                final_turn = -raw_turn if self.invert_steering else raw_turn
            else:
                turn_cmd = self.steering_gain * err 
                min_turn = 0.4
                if abs(turn_cmd) < min_turn: turn_cmd = math.copysign(min_turn, turn_cmd)
                final_turn = -turn_cmd if self.invert_steering else turn_cmd

            self.drive(0.0, final_turn, False)
            if time.time() - self.align_start_time > 8.0: self.state = STATE_CLIMBING; return
            return

        if self.valid_target_locked:
            err = 320 - self.block_x
            turn_cmd = self.steering_gain * err
            final_turn = -turn_cmd if self.invert_steering else turn_cmd
            self.drive(self.linear_speed, final_turn, False)
            
            if (self.lidar_available and self.lidar_min_dist < 0.5) or self.sonar_dist < 0.6 or self.current_area > 20000:
                self.drive(0.0, 0.0, False) 
                self.state = STATE_ALIGNING; self.align_start_time = time.time()
        else:
            self.drive(0.0, -self.search_speed, False)

def main(args=None):
    rclpy.init(args=args); node = LevelManager()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__': main()
