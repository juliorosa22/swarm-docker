import torch
import math
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Dict, List, Tuple, Optional, Union
from torchrl.data import Composite, Bounded, Unbounded
import numpy as np

import sys
sys.path.append('C:\\Users\\julio\\OneDrive\\Documents\\Programming\\Drones\\Airsim\\PythonClient')
import airsim

class UAVEnv(EnvBase):
    """Single UAV environment that interfaces with AirSim using tensor operations."""
    
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
        max_target_distance: float = 100.0,  # Maximum expected distance to target
        max_obstacle_distance: float = 50.0,  # Maximum distance sensor range
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size: Optional[torch.Size] = None,
    ):
        # Set batch size before calling super().__init__()
        self.batch_size = batch_size or torch.Size([])
        super().__init__(device=device, batch_size=self.batch_size)
        
        self.drone_name = drone_name
        self.collision_counter = 0
        
        # Convert positions to tensors and store on device
        self.start_position = torch.tensor(start_position, dtype=torch.float32, device=device)
        self.end_position = torch.tensor(end_position, dtype=torch.float32, device=device)
        
        self.action_type = action_type
        self.observation_img_size = observation_img_size
        self.obstacle_threshold = obstacle_threshold
        self.goal_threshold = goal_threshold
        
        print(f"Drone:{self.drone_name} - device:{self.device}")

        # State tracking (using tensors)
        self.last_position = None
        self.done = False
        self.collision_detected = False
        self.max_target_distance = max_target_distance
        self.max_obstacle_distance = max_obstacle_distance

        # Define observation and action spaces
        self._define_spaces()
        
        # AirSim client will be set by the SwarmEnv
        self.client = client
        self._set_seed()  # Set random seed if needed
        self._setup_flight()  # Setup flight for the drone
        
    def _define_spaces(self):
        """Define observation and action spaces for a single UAV."""
        # Image space
        h, w = self.observation_img_size
        img_shape = (1, h, w)  # Single channel for depth image
        
        # Define observation components with proper batch dimensions
        self.observation_spec = Composite({
            "depth_image": Bounded(
                shape=(*self.batch_size, *img_shape), 
                dtype=torch.float32, 
                low=0.0, 
                high=100.0,
                device=self.device
            ),
            "position": Unbounded(
                shape=(*self.batch_size, 3), 
                dtype=torch.float32,
                device=self.device
            ),
            "rotation": Unbounded(
                shape=(*self.batch_size, 3), 
                dtype=torch.float32,
                device=self.device
            ),
            "velocity": Unbounded(
                shape=(*self.batch_size, 3), 
                dtype=torch.float32,
                device=self.device
            ),
            # NEW: Distance to target position (scalar)
            "target_distance": Bounded(
                shape=(*self.batch_size, 1), 
                dtype=torch.float32,
                low=0.0,
                high=2*self.max_target_distance,
                device=self.device
            ),
            # NEW: Distance to nearest front obstacle (scalar)
            "front_obs_distance": Bounded(
                shape=(*self.batch_size, 1), 
                dtype=torch.float32,
                low=0.0,
                high=self.max_obstacle_distance,
                device=self.device
            ),
            "collisions": Unbounded(
                shape=(*self.batch_size, 1), 
                dtype=torch.float32,
                device=self.device
            )
        }, device=self.device)
        
        # Define action space
        if self.action_type == "discrete":
            self.action_spec = Bounded(
                shape=(*self.batch_size, 1), 
                dtype=torch.int64, 
                low=0, 
                high=9,
                device=self.device
            )
        else:  # continuous
            self.action_spec = Bounded(
                shape=(*self.batch_size, 4), 
                dtype=torch.float32, 
                low=torch.tensor([-1.0, -1.0, -1.0, -10.0], device=self.device),
                high=torch.tensor([10.0, 10.0, 1.0, 10.0], device=self.device),
                device=self.device
            )
            
        # Define done spec
        self.done_spec = Bounded(
            shape=(*self.batch_size, 1), 
            dtype=torch.bool, 
            low=0, 
            high=1,
            device=self.device
        )
        
        # Add reward spec
        self.reward_spec = Unbounded(
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
    
    def _enable_api_control(self):
        """Enable API control for the drone."""
        if self.client is None:
            raise RuntimeError("AirSim client not set. UAVEnv should be used within SwarmEnv.")
        else:
            if not self.client.isApiControlEnabled(vehicle_name=self.drone_name):
                print(f"Enabling API control for {self.drone_name}")
                self.client.enableApiControl(True, self.drone_name)
                self.client.armDisarm(True, self.drone_name)
        
    def _setup_flight(self,reset_position: Optional[Tuple[float, float, float]] = None):
        """Setup the flight for the drone."""
        self._enable_api_control()
        new_start_position = torch.tensor(reset_position, dtype=torch.float32, device=self.device)  if reset_position is not None  else self.start_position
        # Reset and position the drone at starting location
        
        x, y, z = new_start_position.cpu().tolist()
        position = airsim.Vector3r(x, y, z)
        orientation = airsim.Quaternionr(0, 0, 0, 1)  # Default orientation
        pose = airsim.Pose(position, orientation)
        self.client.simSetVehiclePose(pose, True, self.drone_name)
        
        # Takeoff and move to position
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()
        self.client.moveToPositionAsync(float(x), float(y), float(z), 5, vehicle_name=self.drone_name).join()
        
        # Store initial position
        self.last_position = new_start_position.clone()
        

    def _execute_action(self, action):
        """Execute action for this drone synchronously."""
        # Handle batched actions - take first element if batched
        if action.dim() > 1:
            action = action.squeeze(0)
            
        if self.action_type == "discrete":
            action_idx = action.item()
            
            # Discrete action mapping
            if action_idx == 0:
                self.client.moveByVelocityAsync(1.0, 0.0, 0.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 1:
                self.client.moveByVelocityAsync(3.0, 0.0, 0.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 2:
                self.client.moveByVelocityAsync(5.0, 0.0, 0.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 3:
                self.client.moveByVelocityAsync(0.0, -1.0, 0.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 4:
                self.client.moveByVelocityAsync(0.0, 1.0, 0.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 5:
                self.client.moveByVelocityAsync(-1.0, 0.0, 0.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 6:
                self.client.moveByVelocityAsync(0.0, 0.0, -1.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 7:
                self.client.moveByVelocityAsync(0.0, 0.0, 1.0, 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 8:
                drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                current_orientation = drone_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                self.client.rotateToYawAsync(current_yaw + math.radians(45), 0.5, vehicle_name=self.drone_name).join()
            elif action_idx == 9:
                drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                current_orientation = drone_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                self.client.rotateToYawAsync(current_yaw - math.radians(45), 0.5, vehicle_name=self.drone_name).join()
        else:  # Continuous
            # Convert to CPU for AirSim API
            vx, vy, vz, yaw_increment = action.cpu().tolist()
            
            # Apply continuous actions
            self.client.moveByVelocityAsync(vx, vy, vz, 0.5, vehicle_name=self.drone_name).join()
            
            # Apply yaw rotation if needed
            if abs(yaw_increment) > 0.1:
                drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
                current_orientation = drone_state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(current_orientation)[2]
                self.client.rotateToYawAsync(current_yaw + math.radians(yaw_increment), 0.5, vehicle_name=self.drone_name).join()
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for the UAV environment."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    ##observation handler 

    def _get_target_distance(self, current_position):
        """Calculate L2 norm distance between current position and target position."""
        try:
            distance = torch.norm(current_position - self.end_position)
            # Clamp to maximum expected distance
            distance = torch.clamp(distance, 0.0, self.max_target_distance)
            return distance.unsqueeze(0)  # Add dimension to match spec shape (1,)
        except Exception as e:
            print(f"Error calculating target distance for {self.drone_name}: {e}")
            return torch.tensor([self.max_target_distance], dtype=torch.float32, device=self.device)

    def _get_front_obstacle_distance(self):
        """Get distance to nearest front obstacle using UAV distance sensor."""
        try:
            # Get distance sensor data from AirSim
            distance_data = self.client.getDistanceSensorData("Distance", vehicle_name=self.drone_name)
            
            if distance_data.distance > 0:
                # Distance sensor returns distance in meters
                front_distance = min(distance_data.distance, self.max_obstacle_distance)
            else:
                # If no valid reading, assume no obstacle in range
                front_distance = self.max_obstacle_distance
            
            return torch.tensor([front_distance], dtype=torch.float32, device=self.device)
            
        except Exception as e:
            # Fallback: try to use lidar data as alternative
            try:
                lidar_data = self.client.getLidarData(vehicle_name=self.drone_name)
                point_cloud = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
                
                if len(point_cloud) >= 3:
                    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0]/3), 3))
                    
                    # Filter points in front of drone (positive x direction)
                    front_points = point_cloud[point_cloud[:, 0] > 0]
                    
                    if len(front_points) > 0:
                        # Calculate distances and find minimum
                        distances = np.linalg.norm(front_points, axis=1)
                        min_distance = min(np.min(distances), self.max_obstacle_distance)
                    else:
                        min_distance = self.max_obstacle_distance
                else:
                    min_distance = self.max_obstacle_distance
                
                return torch.tensor([min_distance], dtype=torch.float32, device=self.device)
                
            except Exception as lidar_error:
                print(f"Warning: Could not get obstacle distance for {self.drone_name}: {e}, {lidar_error}")
                # Return maximum distance as safe fallback
                return torch.tensor([self.max_obstacle_distance], dtype=torch.float32, device=self.device)

    def get_error_observation(self):
        """Get a safe observation in case of an error."""
        return TensorDict({
            "depth_image": torch.zeros((1, *self.observation_img_size), device=self.device, dtype=torch.float32),
            "position": torch.zeros(3, device=self.device, dtype=torch.float32),
            "rotation": torch.zeros(3, device=self.device, dtype=torch.float32),
            "velocity": torch.zeros(3, device=self.device, dtype=torch.float32),
            "target_distance": torch.tensor([self.max_target_distance], device=self.device, dtype=torch.float32),
            "front_obs_distance": torch.tensor([self.max_obstacle_distance], device=self.device, dtype=torch.float32),
            "collisions": torch.tensor([self.collision_counter], dtype=torch.int32, device=self.device)
        }, batch_size=self.batch_size)

    def _get_observation(self):
        """Get observation for this UAV using tensor operations."""
        try:
            self.client.simPause(True)
            # Get depth image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ], vehicle_name=self.drone_name)
            
            if not responses or len(responses) == 0:
                print("error on getting image")
                # Fallback to zeros if image request fails
                print(f"Warning: Failed to get depth image for {self.drone_name}, using zeros")
                depth_tensor = torch.zeros((1, *self.observation_img_size), device=self.device, dtype=torch.float32)
            else:
                # Process depth image
                depth_img = np.array(responses[0].image_data_float, dtype=np.float32)
                print(f"shape HxW:{responses[0].height} X {responses[0].width}")
                if len(depth_img) == 0:
                    depth_tensor = torch.zeros((1, *self.observation_img_size), device=self.device, dtype=torch.float32)
                else:
                    depth_img = depth_img.reshape(responses[0].height, responses[0].width)
                    depth_tensor = torch.from_numpy(depth_img.copy())
                    depth_tensor = depth_tensor.unsqueeze(0)  # Add channel dimension
                    depth_tensor = depth_tensor.to(self.device)
                    depth_tensor = torch.clamp(depth_tensor, 0.0, 100.0)
            
            # Get drone state
            state = self.client.getMultirotorState(vehicle_name=self.drone_name)
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            vel = state.kinematics_estimated.linear_velocity
            
            # Check collision
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
            if collision_info.has_collided:
                self.collision_detected = True
                self.collision_counter += 1
            
            # Convert state to tensors
            position = torch.tensor(
                [pos.x_val, pos.y_val, pos.z_val], 
                dtype=torch.float32, device=self.device
            )
            
            # Convert quaternion to euler angles
            pitch, roll, yaw = airsim.to_eularian_angles(orientation)
            rotation = torch.tensor(
                [roll, pitch, yaw], 
                dtype=torch.float32, device=self.device
            )
            
            velocity = torch.tensor(
                [vel.x_val, vel.y_val, vel.z_val], 
                dtype=torch.float32, device=self.device
            )
            
            # Calculate new observation components
            target_distance = self._get_target_distance(position)
            front_obs_distance = self._get_front_obstacle_distance()
            
            # Create observation dictionary - ensure proper batch dimensions
            obs = TensorDict({
                "depth_image": depth_tensor,
                "position": position,
                "rotation": rotation,
                "velocity": velocity,
                "target_distance": target_distance,
                "front_obs_distance": front_obs_distance,
                "collisions": torch.tensor([self.collision_counter], dtype=torch.int32, device=self.device)
            }, batch_size=self.batch_size)
            
            self.client.simPause(False)
            return obs
            
        except Exception as e:
            print(f"Error getting observation for {self.drone_name}: {e}")
            # Return default observation
            return self.get_error_observation()

    def _compute_reward(self, obs_dict):
        """Compute reward based on distance to goal and other factors using observation data.
        
        Args:
            obs_dict: TensorDict containing the current observation with all sensor data
            
        Returns:
            float: Calculated reward
        """
        try:
            current_position = obs_dict["position"]
            target_distance = obs_dict["target_distance"]
            front_obs_distance = obs_dict["front_obs_distance"]
            
            # Calculate distance improvement using target_distance from observation
            if self.last_position is not None:
                prev_dist_to_goal = torch.norm(self.last_position - self.end_position)
                curr_dist_to_goal = target_distance.item()  # Use from observation
                
                # Calculate distance improvement
                dist_improvement = prev_dist_to_goal.item() - curr_dist_to_goal
            else:
                # First step, no previous position
                dist_improvement = 0.0
            
            # Base reward: progress toward goal
            reward = dist_improvement * 1.0
            
            # Penalty for collision
            if self.collision_detected:
                reward -= 2.0
                self.collision_detected = False  # Reset collision flag
            
            # Obstacle avoidance reward/penalty using front sensor data
            obstacle_distance = front_obs_distance.item()
            if obstacle_distance < self.obstacle_threshold:
                # Penalty for getting too close to obstacles
                obstacle_penalty = (self.obstacle_threshold - obstacle_distance) / self.obstacle_threshold
                reward -= obstacle_penalty * 0.5  # Scale the penalty
            
            # Small penalty for each step to encourage efficiency
            reward -= 0.01
            
            return reward

        except Exception as e:
            print(f"Error computing reward for {self.drone_name}: {e}")
            return -0.01  # Default small penalty

    def format_results(self,obs_dict: TensorDict, reward: float, error:bool=False) -> TensorDict:
        """Format results into a TensorDict with proper batch dimensions."""
        if error:
            print(f"Error in {self.drone_name}, returning minimal valid result")
        result = TensorDict({
                    'obs': obs_dict,
                    'depth_image': obs_dict['depth_image'],
                    'position': obs_dict['position'],  
                    'rotation': obs_dict['rotation'],
                    'velocity': obs_dict['velocity'],
                    'target_distance': obs_dict['target_distance'],
                    'front_obs_distance': obs_dict['front_obs_distance'],
                    'collisions': obs_dict['collisions'],
                    'done': torch.tensor([self.done], dtype=torch.bool, device=self.device),
                    'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
                }, batch_size=self.batch_size)
        return result

    def _reset(self, tensordict=None, reset_position: Optional[Tuple[float, float, float]] = None):
        """Reset the UAV to initial state."""
        try:
            # Reset state tracking
            self.done = False
            self.collision_detected = False
            self.collision_counter = 0
            self.last_position = None
            
            # Setup flight
            self._setup_flight(reset_position=reset_position)
            
            # Get initial observation
            obs_dict = self._get_observation()
            
            # Create result with proper batch dimensions
            result = self.format_results(obs_dict, 0.0)
            
            return result
            
        except Exception as e:
            print(f"Error during reset for {self.drone_name}: {e}")
            # Return minimal valid reset
            return self.format_results(self.get_error_observation(), 0.0, error=True)

    def _step(self, tensordict):
        """Execute one step for this UAV based on provided action."""
        try:
            # Skip if already done
            if self.done:
                obs_dict = self._get_observation()
                result = self.format_results(obs_dict, 0.0)
                return result
            
            # Extract action
            action = tensordict["action"]
            
            # Execute action
            self._execute_action(action)
            
            # Get observation
            
            obs_dict = self._get_observation()
            current_position = obs_dict["position"]
            
            # Calculate reward
            
            reward = self._compute_reward(obs_dict)
            
            # Check if goal reached
            dist_to_goal = torch.norm(current_position - self.end_position)
            
            if dist_to_goal < self.goal_threshold:
                self.done = True
                print(f"Goal reached by {self.drone_name}!")
            
            # Update last position
            self.last_position = current_position.clone()
            
            # Create return tensordict
            result = self.format_results(obs_dict, reward)

            return result
            
        except Exception as e:
            print(f"Error during step for {self.drone_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallback
            return self.format_results(self.get_error_observation(),0.0,error=True)