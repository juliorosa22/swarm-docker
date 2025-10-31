import math
import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from gymnasium import spaces

from .airsim_env import AirSimEnv
from .uav_config import UAVConfig  # Add this import
import sys
sys.path.append('C:\\Users\\julio\\OneDrive\\Documents\\Programming\\Drones\\Airsim\\PythonClient')
import airsim


class UAVGymEnv(AirSimEnv):
    """Single UAV environment using Gymnasium interface that interfaces with AirSim."""
    
    def __init__(
        self,
        client: airsim.client,
        config: UAVConfig
    ):
        super().__init__()
        
        # Set AirSim client
        self.client = client
        
        # Unpack config into instance attributes
        self.drone_name = config.drone_name
        self.start_position = np.array(config.start_position, dtype=np.float32)
        self.end_position = np.array(config.end_position, dtype=np.float32)
        self.action_type = config.action_type
        self.observation_img_size = config.observation_img_size
        self.obstacle_threshold = config.obstacle_threshold
        self.goal_threshold = config.goal_threshold
        self.max_target_distance = (
            1.5 * np.linalg.norm(self.start_position - self.end_position)
            if config.max_target_distance is None
            else config.max_target_distance
        )
        self.max_obstacle_distance = config.max_obstacle_distance
        self.max_steps = config.max_steps
        
        # Initialize collision counter
        self.collision_counter = 0
        
        # Initialize current_state to None
        self.current_state = None
        
        print(f"Drone:{self.drone_name} - Gymnasium UAV Environment")

        # State tracking
        self.last_position = self.start_position.copy()
        self.done = False
        self.truncated = False
        self.collision_detected = False
        self.step_count = 0
        

        # Define Gymnasium spaces
        self._define_spaces()
        
        # Initialize AirSim connection
        self._setup_flight()
        
    def _define_spaces(self):
        """Define observation and action spaces using Gymnasium format."""
        
        # Define observation space as Dict space
        h, w = self.observation_img_size
        
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(
                low=0.0, 
                high=255.0, 
                shape=(h, w), 
                dtype=np.float32
            ),
            'position': spaces.Box(
                low=-1000.0, 
                high=1000.0, 
                shape=(3,), 
                dtype=np.float32
            ),
            'rotation': spaces.Box(
                low=-np.pi, 
                high=np.pi, 
                shape=(3,), 
                dtype=np.float32
            ),
            'velocity': spaces.Box(
                low=-10.0, 
                high=10.0, 
                shape=(3,), 
                dtype=np.float32
            ),
            'target_distance': spaces.Box(
                low=0.0, 
                high=2*self.max_target_distance, 
                shape=(1,), 
                dtype=np.float32
            ),
            'front_obs_distance': spaces.Box(
                low=0.0, 
                high=self.max_obstacle_distance, 
                shape=(1,), 
                dtype=np.float32
            ),
        })
        
        # Define action space
        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(8)  # 0-7 valid actions
        else:  # continuous
            self.action_space = spaces.Box(
                low=np.array([-10.0, -10.0, -1.0, -10.0], dtype=np.float32),
                high=np.array([10.0, 10.0, 1.0, 10.0], dtype=np.float32),
                dtype=np.float32
            )
    
    def _enable_api_control(self):
        """Enable API control for the drone."""
        if self.client is None:
            raise RuntimeError("AirSim client not set.")
        else:
            if not self.client.isApiControlEnabled(vehicle_name=self.drone_name):
                #print(f"Enabling API control for {self.drone_name}")
                self.client.enableApiControl(True, self.drone_name)
                self.client.armDisarm(True, self.drone_name)
        
    def _setup_flight(self):
        """Setup the flight for the drone."""
        
        #new_start_position = np.array(reset_position, dtype=np.float32) if reset_position is not None else self.start_position
        
        # Reset and position the drone at starting location
        x, y, z = self.start_position.tolist()
        position = airsim.Vector3r(x, y, z)
        orientation = airsim.Quaternionr(0, 0, -0.707, 0.707)  # Default orientation
        pose = airsim.Pose(position, orientation)
        self.client.simSetVehiclePose(pose, True, self.drone_name)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self._enable_api_control()
        #self.client.hoverAsync(vehicle_name=self.drone_name).join()
        # Takeoff and move to position
        #self.client.takeoffAsync(vehicle_name=self.drone_name).join()
        #self.client.moveToPositionAsync(float(x), float(y), float(z), 5, vehicle_name=self.drone_name).join()
        
        # Store initial position
        self.last_position = self.start_position.copy()

    def act(self, action):
        """Execute action for this drone synchronously, structured like act_no_join but with join() calls."""
        duration_cmd = 1.0  # Standardized duration to match act_no_join
        # Update current_state if not set
        if self.current_state is None:
            self.current_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        
        if self.action_type == "discrete":
            action_idx = int(action)
            if action_idx < 6:
                act_vec = self.map_int_act(action_idx)
                vx, vy, vz = float(act_vec[0]), float(act_vec[1]), float(act_vec[2])
                return self.client.moveByVelocityAsync(vx, vy, vz, duration_cmd, vehicle_name=self.drone_name).join()
            elif action_idx == 6:
                current_orientation = self.current_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                return self.client.rotateToYawAsync(current_yaw + math.radians(45), duration_cmd, vehicle_name=self.drone_name).join()
            elif action_idx == 7:
                current_orientation = self.current_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                return self.client.rotateToYawAsync(current_yaw - math.radians(45), duration_cmd, vehicle_name=self.drone_name).join()
        else:  # Continuous
            vx, vy, vz, yaw_increment = action.astype(np.float64)
            
            # Apply continuous actions with join
            if abs(yaw_increment) > 0.1:
                current_orientation = self.current_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                self.client.rotateToYawAsync(current_yaw + math.radians(yaw_increment), duration_cmd, vehicle_name=self.drone_name).join()
            else:
                self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration_cmd, vehicle_name=self.drone_name).join()
        
        # No return value, as it's synchronous

    def map_int_act(self, action):
        """Map integer action to continuous action space."""
        # Removed the action_type check to allow use in discrete mode
        mapping = {
            0: np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 0.0, 3.0, 0.0], dtype=np.float32),
            3: np.array([-3.0, 0.0, 0.0, 0.0], dtype=np.float32),
            4: np.array([0.0, -3.0, 0.0, 0.0], dtype=np.float32),
            5: np.array([0.0, 0.0, -3.0, 0.0], dtype=np.float32),
            6: np.array([0.0, 0.0, 0.0, 45.0], dtype=np.float32),
            7: np.array([0.0, 0.0, 0.5, -45.0], dtype=np.float32),
        }
        return mapping.get(action, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def act_no_join(self, action):
        """Execute action for this drone asynchronously, returning a single future based on the action condition."""
        duration_cmd=1.0
        # Update current_state if not set
        if self.current_state is None:
            self.current_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        
        if self.action_type == "discrete":
            action_idx = int(action)
            if action_idx < 6:
                act_vec = self.map_int_act(action_idx)
                vx, vy, vz = float(act_vec[0]), float(act_vec[1]), float(act_vec[2])
                return self.client.moveByVelocityAsync(vx, vy, vz, duration_cmd, vehicle_name=self.drone_name)
            elif action_idx == 6:
                current_orientation = self.current_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                return self.client.rotateToYawAsync(current_yaw + math.radians(45), duration_cmd, vehicle_name=self.drone_name)
            elif action_idx == 7:
                current_orientation = self.current_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                return self.client.rotateToYawAsync(current_yaw - math.radians(45), duration_cmd, vehicle_name=self.drone_name)
        else:  # Continuous
            vx, vy, vz, yaw_increment = action.astype(np.float64)
            
            # Return future based on condition: rotate if yaw_increment is significant, else move
            if abs(yaw_increment) > 0.1:
                current_orientation = self.current_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                return self.client.rotateToYawAsync(current_yaw + math.radians(yaw_increment), duration_cmd, vehicle_name=self.drone_name)
            else:
                return self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration_cmd, vehicle_name=self.drone_name)

        # If no condition matches (shouldn't happen), return None
        return None

    def end_last_action(self):
        """ends the last action for this drone."""
        if self.client is not None:
            self.client.cancelLastTask(vehicle_name=self.drone_name)
            #print(f"Cancelled last task for {self.drone_name}")
        else:
            print("AirSim client not set, cannot cancel last task.")

    def _get_target_distance(self, current_position):
        """Calculate L2 norm distance between current position and target position."""
        try:
            distance = np.linalg.norm(current_position - self.end_position)
            # Clamp to maximum expected distance
            #distance = np.clip(distance, 0.0, self.max_target_distance)
            return np.array([distance], dtype=np.float32)
        except Exception as e:
            print(f"Error calculating target distance for {self.drone_name}: {e}")
            return np.array([self.max_target_distance], dtype=np.float32)

    def _get_front_obstacle_distance(self, use_lidar: bool = False):
        """Get distance to nearest front obstacle using distance sensors or Lidar."""
        if use_lidar:
            # Use Lidar data (original implementation)
            lidar_data = self.client.getLidarData(vehicle_name=self.drone_name)
            point_cloud = np.array(lidar_data.point_cloud, dtype=np.float32)
            
            if len(point_cloud) >= 3:
                point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0]/3), 3))
                
                # Filter points in front of drone (positive x direction)
                front_points = point_cloud[point_cloud[:, 0] > 0]
                
                if len(front_points) > 0:
                    distances = np.linalg.norm(front_points, axis=1)
                    min_distance = min(np.min(distances), self.max_obstacle_distance)
                else:
                    min_distance = self.max_obstacle_distance
            else:
                min_distance = self.max_obstacle_distance
            
            return np.array([min_distance], dtype=np.float32)
        else:
            # Use three distance sensors and return the minimum
            distances = []
            sensor_names = ["DistanceSensor1", "DistanceSensor2", "DistanceSensor3"]
            
            for sensor_name in sensor_names:
                try:
                    # Get distance sensor data
                    distance_data = self.client.getDistanceSensorData(sensor_name, vehicle_name=self.drone_name)
                    
                    if distance_data.distance > 0:
                        # Clamp to max distance
                        dist = min(distance_data.distance, self.max_obstacle_distance)
                    else:
                        dist = self.max_obstacle_distance
                    
                    distances.append(dist)
                except Exception as e:
                    print(f"Error getting data from {sensor_name} for {self.drone_name}: {e}")
                    distances.append(self.max_obstacle_distance)
            
            # Return the minimum distance as a single value
            min_distance = min(distances) if distances else self.max_obstacle_distance
            return np.array([min_distance], dtype=np.float32)

    def _get_obs(self):
        """Get observation for this UAV using numpy arrays."""
        try:
            #self.client.simPause(True)
            
            # Get depth image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ], vehicle_name=self.drone_name)
            
            if not responses or len(responses) == 0:
                print(f"Warning: Failed to get depth image for {self.drone_name}, using zeros")
                depth_img = np.zeros(self.observation_img_size, dtype=np.float32)
            else:
                # Process depth image
                depth_data = np.array(responses[0].image_data_float, dtype=np.float32)
                if len(depth_data) == 0:
                    depth_img = np.zeros(self.observation_img_size, dtype=np.float32)
                else:
                    depth_img = depth_data.reshape(responses[0].height, responses[0].width)
                    depth_img = np.clip(depth_img, 0.0, 100.0)

            # Update current state
            self.current_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = self.current_state.kinematics_estimated.position
            orientation = self.current_state.kinematics_estimated.orientation
            vel = self.current_state.kinematics_estimated.linear_velocity
            
            # Check collision using both AirSim built-in and sensor data
            #collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
            front_obs_distance = self._get_front_obstacle_distance()  # Use distance sensors by default
            
            # Check for collision based on sensor distance < 0.5m
            if front_obs_distance[0] < 0.5:
                self.collision_detected = True
                self.collision_counter += 1
            
            
            # Convert state to numpy arrays
            position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
            
            # Convert quaternion to euler angles
            pitch, roll, yaw = airsim.to_eularian_angles(orientation)
            rotation = np.array([roll, pitch, yaw], dtype=np.float32)
            
            velocity = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
            
            # Calculate new observation components
            target_distance = self._get_target_distance(position)
            
            
            # Create observation dictionary
            obs = {
                'depth_image': depth_img,
                'position': position,
                'rotation': rotation,
                'velocity': velocity,
                'target_distance': target_distance,
                'front_obs_distance': front_obs_distance,
            }
            
            #self.client.simPause(False)
            return obs
            
        except Exception as e:
            print(f"Error getting observation for {self.drone_name}: {e}")
            # Return default observation
            return self._get_default_obs()

    def _compute_reward(self, obs):
        """Compute reward with optimized scaling."""
        try:
            current_position = obs['position']
            target_distance = obs['target_distance'][0]
            front_obs_distance = obs['front_obs_distance'][0]
            
            # Calculate distance improvement
            if self.last_position is not None:
                prev_dist_to_goal = np.linalg.norm(self.last_position - self.end_position)
                curr_dist_to_goal = target_distance
                dist_improvement = prev_dist_to_goal - curr_dist_to_goal
            else:
                dist_improvement = 0.0
            
            # Base progress reward (slightly increased for more positive feedback)
            progress_reward = float(dist_improvement * 3.5)  # Increased from 3.0
            
            # Safety factor for obstacle proximity
            safety_factor = min(1.0, front_obs_distance / self.obstacle_threshold)
            safety_factor = max(0.2, safety_factor)  # Increased minimum from 0.1
            
            # Apply safety scaling to progress
            progress_reward *= safety_factor
            
            # Obstacle penalty (slightly reduced for better balance)
            obstacle_penalty = 0.0
            if front_obs_distance < self.obstacle_threshold:
                violation_ratio = (self.obstacle_threshold - front_obs_distance) / self.obstacle_threshold
                obstacle_penalty = violation_ratio ** 2 * 1.2  # Reduced from 1.5
        
            # Step penalty (slightly reduced)
            step_penalty = 0.003  # Reduced from 0.005
        
            # Combine components
            reward = progress_reward - obstacle_penalty - step_penalty
        
            return reward

        except Exception as e:
            print(f"Error computing reward for {self.drone_name}: {e}")
            return -0.003  # Reduced default penalty

    def reset(self, seed=None, options=None):
        """Reset the UAV to initial state. Returns (observation, info) to comply with Gym standards."""
        super().reset(seed=seed)
        
        try:
            # Reset state tracking
            self.done = False
            self.truncated = False
            self.collision_detected = False
            self.collision_counter = 0
            self.last_position = None
            self.step_count = 0
            self.current_state = None  # Reset current_state
            
            # Handle reset positions from options
            if options and 'reset_positions' in options:
                reset_pos = options['reset_positions']
                self.start_position = np.array(reset_pos['start'], dtype=np.float32)
                self.end_position = np.array(reset_pos['end'], dtype=np.float32)
            
            
            # Setup flight
            self._setup_flight()
            
            # Get initial observation
            obs = self._get_obs()
            
            # Info dictionary (standard Gym format)
            info = {
                'collisions': self.collision_counter,
                'episode_step': self.step_count,
                'terminated': self.done,
                'truncated': self.truncated,
                'target_distance': float(obs['target_distance'][0])
            }
            return obs, info
            
        except Exception as e:
            print(f"Error during reset for {self.drone_name}: {e}")
            # Return safe fallback
            obs = self._get_default_obs()
            info = {'collisions': 0, 'episode_step': 0,'error': True}
            return obs, 0.0, self.done, self.truncated, info

    def step(self, action):
        """Execute one step. Returns (observation, reward, terminated, truncated, info)."""
        try:

            # Skip if the drone is done or truncated (drone has reached the target or has reached the step limit)
            if self.done or self.truncated:
                obs = self._get_obs()
                info = {
                    'collisions': self.collision_counter,
                    'episode_step': self.step_count,
                    'terminated': self.done,
                    'truncated': self.truncated,
                    'target_distance': float(obs['target_distance'][0])
                }
                return obs, 0.0, self.done, self.truncated, info
            
            self.step_count += 1
            # Execute action
            self.act(action)
            
            # Get observation
            obs = self._get_obs()
            target_distance = obs['target_distance'][0]
            current_position = obs['position']
            
            # Calculate reward (simplified)
            reward = self._compute_reward(obs)
            
            # Check if goal reached (no bonus here)
            terminated = False
            if target_distance < self.goal_threshold:
                self.done = True
                terminated = True
                print(f"Goal reached by {self.drone_name}!")
            
            # Check for truncation (episode too long)
            self.truncated = self.step_count > self.max_steps
            
            # Update last position
            self.last_position = current_position.copy()
            
            # Info dictionary
            info = {
                'collisions': self.collision_counter,
                'episode_step': self.step_count,
                'terminated': terminated,
                'truncated': self.truncated,
                'target_distance': float(target_distance)
            }

            return obs, reward, terminated, self.truncated, info

        except Exception as e:
            print(f"Error during step for {self.drone_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return safe fallback
            obs = self._get_default_obs()
            info = {'collisions': self.collision_counter, 'episode_step': self.step_count, 'error': True}
            return obs, 0.0, True, False, info
        
    def close(self):
        """Clean up resources."""
        try:
            if self.client:
                self.client.enableApiControl(False, self.drone_name)
                self.client.armDisarm(False, self.drone_name)
        except:
            pass

    def _get_default_obs(self):
        """Return a default observation dictionary for error cases."""
        return {
            'depth_image': np.ones(self.observation_img_size, dtype=np.float32)*255.0,
            'position': self.start_position.copy(),
            'rotation': np.zeros(3, dtype=np.float32),
            'velocity': np.zeros(3, dtype=np.float32),
            'target_distance': np.array([self.max_target_distance], dtype=np.float32),
            'front_obs_distance': np.array([self.max_obstacle_distance], dtype=np.float32),
        }



