import math
import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from gymnasium import spaces
from ..airsim_env import AirSimEnv
import sys
import asyncio  # Ensure this is imported for async handling
sys.path.append('C:\\Users\\julio\\OneDrive\\Documents\\Programming\\Drones\\Airsim\\PythonClient')
import airsim

class AsyncUAVGymEnv(AirSimEnv):
    """Single UAV environment using Gymnasium interface that interfaces with AirSim, with async I/O for performance."""

    def __init__(
        self,
        client: airsim.client,
        drone_name: str,
        start_position: Tuple[float, float, float],
        end_position: Tuple[float, float, float],
        action_type: str = "continuous",
        observation_img_size: Tuple[int, int] = (64, 64),
        obstacle_threshold: float = 2.0,
        goal_threshold: float = 1.0,
        max_target_distance: Optional[float] = None,
        max_obstacle_distance: float = 50.0,
    ):
        super().__init__()
        
        self.client = client
        self.drone_name = drone_name
        self.collision_counter = 0
        
        # Store positions as numpy arrays for Gym compatibility
        self.start_position = np.array(start_position, dtype=np.float32)
        self.end_position = np.array(end_position, dtype=np.float32)

        
        self.action_type = action_type
        self.observation_img_size = observation_img_size
        self.obstacle_threshold = obstacle_threshold
        self.goal_threshold = goal_threshold
        
        self.max_target_distance = 1.5 * np.linalg.norm(self.start_position - self.end_position) if max_target_distance is None  else max_target_distance
        
        self.max_obstacle_distance=max_obstacle_distance
        
        
        print(f"Drone:{self.drone_name} - Async Gymnasium UAV Environment")

        # State tracking
        self.last_position = None
        self.done = False
        self.truncated = False
        self.collision_detected = False
        self.step_count = 0
        self.max_steps = 300  # Add episode truncation
        
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
            self.action_space = spaces.Discrete(10)
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
                print(f"Enabling API control for {self.drone_name}")
                self.client.enableApiControl(True, self.drone_name)
                self.client.armDisarm(True, self.drone_name)
        
    def _setup_flight(self, reset_position: Optional[Tuple[float, float, float]] = None):
        """Setup the flight for the drone."""
        self._enable_api_control()
        new_start_position = np.array(reset_position, dtype=np.float32) if reset_position is not None else self.start_position
        
        # Reset and position the drone at starting location
        x, y, z = new_start_position.tolist()
        position = airsim.Vector3r(x, y, z)
        orientation = airsim.Quaternionr(0, 0, 0, 1)  # Default orientation
        pose = airsim.Pose(position, orientation)
        self.client.simSetVehiclePose(pose, True, self.drone_name)
        
        # Takeoff and move to position (async, wrapped for asyncio)
        future = self.client.takeoffAsync(vehicle_name=self.drone_name)
        asyncio.run(asyncio.wrap_future(future))  # Wrap and run
        future = self.client.moveToPositionAsync(float(x), float(y), float(z), 5, vehicle_name=self.drone_name)
        asyncio.run(asyncio.wrap_future(future))
        
        # Store initial position
        self.last_position = new_start_position.copy()

    async def act(self, action):  # Made async
        """Execute action for this drone asynchronously."""
        if self.action_type == "discrete":
            action_idx = int(action)
            
            # Discrete action mapping (async)
            if action_idx == 0:
                future = self.client.moveByVelocityAsync(1.0, 0.0, 0.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 1:
                future = self.client.moveByVelocityAsync(3.0, 0.0, 0.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 2:
                future = self.client.moveByVelocityAsync(5.0, 0.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 3:
                future = self.client.moveByVelocityAsync(0.0, -1.0, 0.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 4:
                future = self.client.moveByVelocityAsync(0.0, 1.0, 0.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 5:
                future = self.client.moveByVelocityAsync(-1.0, 0.0, 0.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 6:
                future = self.client.moveByVelocityAsync(0.0, 0.0, -1.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 7:
                future = self.client.moveByVelocityAsync(0.0, 0.0, 1.0, 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 8:
                future = self.client.getMultirotorStateAsync(vehicle_name=self.drone_name)
                state = await asyncio.wrap_future(future)
                current_orientation = state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                future = self.client.rotateToYawAsync(current_yaw + math.radians(45), 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
            elif action_idx == 9:
                future = self.client.getMultirotorStateAsync(vehicle_name=self.drone_name)
                state = await asyncio.wrap_future(future)
                current_orientation = state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                future = self.client.rotateToYawAsync(current_yaw - math.radians(45), 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)
        else:  # Continuous
            vx, vy, vz, yaw_increment = action.astype(np.float64)
            
            # Apply continuous actions (async)
            future = self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), 0.5, vehicle_name=self.drone_name)
            await asyncio.wrap_future(future)
            
            # Apply yaw rotation if needed (async)
            if abs(yaw_increment) > 0.1:
                future = self.client.getMultirotorStateAsync(vehicle_name=self.drone_name)
                state = await asyncio.wrap_future(future)
                current_orientation = state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                future = self.client.rotateToYawAsync(current_yaw + math.radians(yaw_increment), 0.5, vehicle_name=self.drone_name)
                await asyncio.wrap_future(future)

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

    def _get_front_obstacle_distance(self,use_lidar: bool = True):
        """Get distance to nearest front obstacle using UAV distance sensor."""
        if use_lidar:
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
        # Get distance sensor data from AirSim
            distance_data = self.client.getDistanceSensorData("Distance", vehicle_name=self.drone_name)
            
            if distance_data.distance > 0:
                front_distance = min(distance_data.distance, self.max_obstacle_distance)
            else:
                front_distance = self.max_obstacle_distance
            
            return np.array([front_distance], dtype=np.float32)
            
        

    async def _get_obs(self):  # Made async
        """Get observation for this UAV using numpy arrays asynchronously."""
        try:
            #self.client.simPause(True)
            
            # Get depth image (async)
            future = self.client.simGetImagesAsync([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ], vehicle_name=self.drone_name)
            responses = await asyncio.wrap_future(future)
            
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
            
            # Get drone state (async)
            future = self.client.getMultirotorStateAsync(vehicle_name=self.drone_name)
            state = await asyncio.wrap_future(future)
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            vel = state.kinematics_estimated.linear_velocity
            
            # Check collision (async)
            future = self.client.simGetCollisionInfoAsync(vehicle_name=self.drone_name)
            collision_info = await asyncio.wrap_future(future)
            if collision_info.has_collided:
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
            front_obs_distance = self._get_front_obstacle_distance()
            
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
            return {
                'depth_image': np.zeros(self.observation_img_size, dtype=np.float32),
                'position': np.zeros(3, dtype=np.float32),
                'rotation': np.zeros(3, dtype=np.float32),
                'velocity': np.zeros(3, dtype=np.float32),
                'target_distance': np.array([self.max_target_distance], dtype=np.float32),
                'front_obs_distance': np.array([self.max_obstacle_distance], dtype=np.float32),
            }

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

    async def reset(self, seed=None, options=None):  # Made async
        """Reset the UAV to initial state. Returns (observation, info)."""
        super().reset(seed=seed)
        
        try:
            # Reset state tracking
            self.done = False
            self.truncated = False
            self.collision_detected = False
            self.collision_counter = 0
            self.last_position = None
            self.step_count = 0
            
            # Handle reset position from options
            reset_position = None
            if options and 'reset_position' in options:
                reset_position = options['reset_position']
            
            # Setup flight
            self._setup_flight(reset_position=reset_position)
            
            # Get initial observation
            obs = await self._get_obs()
            
            # Info dictionary
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
            obs = {
                'depth_image': np.zeros(self.observation_img_size, dtype=np.float32),
                'position': np.zeros(3, dtype=np.float32),
                'rotation': np.zeros(3, dtype=np.float32),
                'velocity': np.zeros(3, dtype=np.float32),
                'target_distance': np.array([self.max_target_distance], dtype=np.float32),
                'front_obs_distance': np.array([self.max_obstacle_distance], dtype=np.float32),
            }
            info = {'collisions': 0, 'episode_step': 0,'error': True}
            return obs, info
        
    async def step(self, action):  # Made async
        """Execute one step. Returns (observation, reward, terminated, truncated, info)."""
        try:

            # Skip if the drone is done or truncated (drone has reached the target or has reached the step limit)
            if self.done or self.truncated:
                obs = await self._get_obs()
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
            await self.act(action)
            
            # Get observation
            obs = await self._get_obs()
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
            obs = {
                'depth_image': np.zeros(self.observation_img_size, dtype=np.float32),
                'position': np.zeros(3, dtype=np.float32),
                'rotation': np.zeros(3, dtype=np.float32),
                'velocity': np.zeros(3, dtype=np.float32),
                'target_distance': np.array([self.max_target_distance], dtype=np.float32),
                'front_obs_distance': np.array([self.max_obstacle_distance], dtype=np.float32),
            }
            info = {'collisions': self.collision_counter, 'episode_step': self.step_count, 'error': True}
            return obs, -1.0, True, False, info
        
    def close(self):
        """Clean up resources."""
        try:
            if self.client:
                self.client.enableApiControl(False, self.drone_name)
                self.client.armDisarm(False, self.drone_name)
        except:
            pass