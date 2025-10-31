import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim
import time
import json
from ..swarm_config import SwarmConfig
from typing import List

class FastSwarmEnv(gym.Env):
    """
    A unified, high-performance swarm environment for AirSim.
    It centralizes all logic, removing the need for separate UAV envs,
    and uses simulation pausing and frame skipping to accelerate training.
    """

    def __init__(self, config: SwarmConfig, frame_skip_duration: float = 0.5, use_lidar: bool = False):
        super().__init__()
        
        self.config = config
        self.n_agents = config.n_agents
        self.drone_names = [f"uav{i}" for i in range(self.n_agents)]
        self.frame_skip_duration = frame_skip_duration
        self.act_duration=1.0 # 1 second action duration for AirSim commands
        self.use_lidar = use_lidar

        # Centralized AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # All state is managed here, not in sub-objects
        self.step_count = 0
        self.done_flags = [False] * self.n_agents
        self.last_positions = [np.zeros(3)] * self.n_agents
        self.start_positions = []
        self.end_positions = []
        self.initial_target_distances = np.zeros(self.n_agents)
        self.last_target_distances = np.zeros(self.n_agents)
        self.kinematic_states = {} # To store current state for action execution
        
        # Collision tracking
        self.collision_counters = [0] * self.n_agents
        self.episode_collision_log = []  # To store per-episode collision data
        
        self._define_spaces()
        
        # Pause simulation on init to prevent drift before first reset
        self.client.simPause(True)

    def _define_spaces(self):
        """Define observation and action spaces to match RLSwarmEnv."""
        # Define a base observation space for a single agent
        # Assuming image dimensions from a typical config
        img_height = self.config.img_height if hasattr(self.config, 'img_height') else 64
        img_width = self.config.img_width if hasattr(self.config, 'img_width') else 64
        
        agent_obs_space = spaces.Dict({
            'depth_image': spaces.Box(low=0, high=255, shape=(self.n_agents, img_height, img_width, 1), dtype=np.float32),
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents, 3), dtype=np.float32),
            'rotation': spaces.Box(low=-np.pi, high=np.pi, shape=(self.n_agents, 3), dtype=np.float32),
            'velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents, 3), dtype=np.float32),
            'target_distance': spaces.Box(low=0, high=np.inf, shape=(self.n_agents, 1), dtype=np.float32),
            'front_obs_distance': spaces.Box(low=0, high=np.inf, shape=(self.n_agents, 1), dtype=np.float32),
        })

        shared_obs_space = spaces.Dict({
            'inter_agent_distances': spaces.Box(low=0.0, high=np.inf, shape=(self.n_agents, self.n_agents), dtype=np.float32),
            'target_distances': spaces.Box(low=0.0, high=np.inf, shape=(self.n_agents,), dtype=np.float32),
            'velocities': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents, 3), dtype=np.float32),
            'obstacle_distances': spaces.Box(low=0.0, high=np.inf, shape=(self.n_agents,), dtype=np.float32)
        })

        self.observation_space = spaces.Dict({
            'agents': agent_obs_space,
            'shared_observation': shared_obs_space
        })
        
        # Define action space for multiple agents
        if self.config.action_type == "discrete":
            # Use a Tuple to define a multi-agent action space
            self.action_space = spaces.Tuple([spaces.Discrete(8)] * self.n_agents)
        else:  # continuous
            # Use a Tuple for continuous actions as well
            agent_action_space = spaces.Box(
                low=np.array([-5.0, -5.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([5.0, 5.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
            self.action_space = spaces.Tuple([agent_action_space] * self.n_agents)

    def reset(self, seed=None, options=None):
        """Resets the entire swarm and returns the initial observation dictionary."""
        super().reset(seed=seed)
        
        #self.client.simPause(True)
        
        self.step_count = 0
        self.done_flags = [False] * self.n_agents
        self.collision_counters = [0] * self.n_agents  # Reset collision counters
        
        self.start_positions, self.end_positions = self.config.generate_positions()
        
        # Calculate and store initial distances for truncation and reward logic
        self.initial_target_distances = np.linalg.norm(np.array(self.start_positions) - np.array(self.end_positions), axis=1)
        self.last_target_distances = self.initial_target_distances.copy()
        self.client.simPause(False)
        for i in range(self.n_agents):
            name = self.drone_names[i]
            start_pos = self.start_positions[i]
            pose = airsim.Pose(airsim.Vector3r(*start_pos), airsim.Quaternionr())
            self.client.simSetVehiclePose(pose, True, vehicle_name=name)
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
            self.last_positions[i] = np.array(start_pos)

        #self.client.simPause(False)
        #time.sleep(0.1)
        #self.client.simPause(True)
        #frames=int(self.act_duration / self.frame_skip_duration)
        #for _ in range(frames):
        #    self.client.simContinueForTime(self.frame_skip_duration)

        obs = self._get_swarm_observation()
        self.client.simPause(True)
        # Update last_target_distances with the first real observation
        self.last_target_distances = obs['shared_observation']['target_distances'].copy()
        
        infos = self._get_infos(obs)
        # Remove per-step logging: self.current_episode_infos.append(infos)  # Log initial state

        return obs, infos

    def step(self, actions):
        """Executes the pause-act-unpause-wait cycle and returns aggregated results."""
        # Convert actions to a standard Python list to handle inputs from wrappers
        if not isinstance(actions, list):
            actions = list(actions)

        if all(self.done_flags):
            obs = self._get_swarm_observation()
            return obs, 0.0, True, self.step_count >= self.config.max_steps, self._get_infos(obs)

        action_futures = []
        for i, name in enumerate(self.drone_names):
            if not self.done_flags[i]:
                future = self._execute_action(name, i, actions[i],self.act_duration)
                action_futures.append(future)

        self.step_count += 1
        #self.client.simPause(False)
        #time.sleep(self.frame_skip_duration)
        #self.client.simPause(True)
        frames=int(self.act_duration / self.frame_skip_duration)
        for _ in range(frames+1):
            self.client.simContinueForTime(self.frame_skip_duration)
        # Check for collisions after the step
        self.end_swarm_actions()  # Ensure all actions are completed
        self.client.simContinueForTime(0.1)
        self._check_collisions()

        swarm_obs = self._get_swarm_observation()
        
        # Compute rewards and done flags using the new centralized logic
        individual_rewards, terminateds, truncateds = self._compute_rewards_and_dones(swarm_obs)
        
        # Aggregate rewards and dones for the environment step return
        reward = sum(individual_rewards)
        terminated = any(terminateds)
        truncated = any(truncateds)
        
        # Update done flags for the next step
        self.done_flags = [t or d for t, d in zip(terminateds, truncateds)]
        
        infos = self._get_infos(swarm_obs)
        # Remove per-step logging: self.current_episode_infos.append(infos)  # Log info for this step
        
        return swarm_obs, reward, terminated, truncated, infos

    def _check_collisions(self) -> List[bool]:
        """Checks for collisions for all agents and updates counters."""
        collisions_this_step = [False] * self.n_agents
        for i, name in enumerate(self.drone_names):
            collision_info = self.client.simGetCollisionInfo(vehicle_name=name)
            if collision_info.has_collided:
                self.collision_counters[i] += 1
                collisions_this_step[i] = True
        return collisions_this_step

    def _get_front_obstacle_distances(self) -> List[float]:
        """
        Gets front obstacle distance for all agents, using either Lidar or Distance Sensors.
        This logic is adapted from the single-agent UAV environment.
        """
        all_min_distances = []
        max_obs_dist = self.config.max_obstacle_distance if hasattr(self.config, 'max_obstacle_distance') else 100.0

        if self.use_lidar:
            for name in self.drone_names:
                try:
                    lidar_data = self.client.getLidarData(lidar_name="LidarSensor1", vehicle_name=name)
                    point_cloud = np.array(lidar_data.point_cloud, dtype=np.float32)
                    
                    min_distance = max_obs_dist
                    if len(point_cloud) >= 3:
                        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0]/3), 3))
                        
                        # Filter for points in front of the drone (positive x in drone's local frame)
                        front_points = point_cloud[point_cloud[:, 0] > 0]
                        
                        if len(front_points) > 0:
                            distances = np.linalg.norm(front_points, axis=1)
                            min_distance = min(np.min(distances), max_obs_dist)
                    
                    all_min_distances.append(min_distance)
                except Exception:
                    all_min_distances.append(max_obs_dist)
        else: # Use Distance Sensors
            sensor_names = ["DistanceSensor1", "DistanceSensor2", "DistanceSensor3"]
            for name in self.drone_names:
                try:
                    sensor_distances = []
                    for sensor_name in sensor_names:
                        distance_data = self.client.getDistanceSensorData(sensor_name, vehicle_name=name)
                        
                        dist = max_obs_dist
                        if distance_data.distance > 0:
                            dist = min(distance_data.distance, max_obs_dist)
                        
                        sensor_distances.append(dist)
                    
                    min_distance = min(sensor_distances) if sensor_distances else max_obs_dist
                    all_min_distances.append(min_distance)
                except Exception:
                    all_min_distances.append(max_obs_dist)
                    
        return all_min_distances

    def _get_swarm_observation(self):
        """
        Gets and composes observations for all drones into the nested dictionary format.
        """
        # 1. Get individual agent data
        positions, velocities, rotations, depth_images = [], [], [], []
        
        # Get front obstacle distances for all agents at once
        front_obs_distances = self._get_front_obstacle_distances()
        
        # Prepare image requests for batch processing
        img_height = self.config.img_height if hasattr(self.config, 'img_height') else 84
        img_width = self.config.img_width if hasattr(self.config, 'img_width') else 84
        
        # Get state for each drone individually (simulation is paused, so it's synchronized)
        states = {name: self.client.getMultirotorState(vehicle_name=name) for name in self.drone_names}
        
        # Track image request failures
        image_failures = [False] * self.n_agents
    
        for i, name in enumerate(self.drone_names):
            state = states[name]
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            rot = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
            
            positions.append(np.array([pos.x_val, pos.y_val, pos.z_val]))
            velocities.append(np.array([vel.x_val, vel.y_val, vel.z_val]))
            rotations.append(np.array(rot))

            # Get depth image for this specific drone
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ], vehicle_name=name)
            
            # Process image response with error handling
            response = responses[0]  # Only one request per call
            if len(response.image_data_uint8) == 0:
                # If image data is empty, create a default black image
                img_gray = np.zeros((img_height, img_width, 1), dtype=np.float32)
                image_failures[i] = True
                print(f"Warning: Depth image request failed for agent {name} (index {i}). Using default image.")
            else:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgba = img1d.reshape(response.height, response.width, 4)
                img_gray = img_rgba[:, :, 0:1]  # Use one channel for depth
                img_gray = img_gray.astype(np.float32)
                image_failures[i] = False
            
            depth_images.append(img_gray)

        self.last_positions = positions
        
        # 2. Calculate shared observation data
        distance_matrix = np.linalg.norm(np.array(positions)[:, np.newaxis, :] - np.array(positions)[np.newaxis, :, :], axis=2)
        agent_target_distances = np.linalg.norm(np.array(positions) - np.array(self.end_positions), axis=1)

        # 3. Assemble the final observation dictionary
        obs = {
            'agents': {
                'depth_image': np.stack(depth_images),
                'position': np.array(positions, dtype=np.float32),
                'rotation': np.array(rotations, dtype=np.float32),
                'velocity': np.array(velocities, dtype=np.float32),
                'target_distance': agent_target_distances.reshape(-1, 1).astype(np.float32),
                'front_obs_distance': np.array(front_obs_distances).reshape(-1, 1).astype(np.float32),
            },
            'shared_observation': {
                'inter_agent_distances': distance_matrix.astype(np.float32),
                'target_distances': agent_target_distances.astype(np.float32),
                'velocities': np.array(velocities, dtype=np.float32),
                'obstacle_distances': np.array(front_obs_distances, dtype=np.float32),
            }
        }
        
        # Store image failures for info
        self.image_failures = image_failures
        
        return obs

    def _map_int_act(self, action: int):
        """Maps an integer action to a continuous velocity/yaw_rate vector."""
        mapping = {
            0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Forward
            1: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),  # Right
            2: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),  # Down
            3: np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32), # Backward
            4: np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32), # Left
            5: np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32), # Up
            6: np.array([0.0, 0.0, 0.0, 45.0], dtype=np.float32), # Yaw Right
            7: np.array([0.0, 0.0, 0.0, -45.0], dtype=np.float32),# Yaw Left
        }
        return mapping.get(action, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def _execute_action(self, drone_name: str, agent_index: int, action: int, duration_cmd: float = 0.5):
        """Maps integer action to AirSim call and returns the future, using stored state."""
        
        drone_name = self.drone_names[agent_index]
        # For yaw actions, we need the current orientation from the last synced state
        vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
        if self.config.action_type == "discrete":
            action_idx = int(action)
            act_vec = self._map_int_act(action_idx)
            vx, vy, vz, yaw = float(act_vec[0]), float(act_vec[1]), float(act_vec[2]), float(act_vec[3])
            yaw_rate = yaw / duration_cmd  # Convert to rate
            yaw_set = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            return self.client.moveByVelocityAsync(vx, vy, vz, duration_cmd, vehicle_name=drone_name, yaw_mode=yaw_set)

            # if action_idx in [6, 7]:
            #     if drone_name in self.kinematic_states:
            #         current_orientation = self.kinematic_states[drone_name].kinematics_estimated.orientation
            #         current_yaw = airsim.to_eularian_angles(current_orientation)[2]
            #         yaw_increment = self._map_int_act(action_idx)[3]
            #         target_yaw_rad = current_yaw + math.radians(yaw_increment)
            #         # Normalize yaw to be within [-pi, pi]
            #         target_yaw_rad = (target_yaw_rad + math.pi) % (2 * math.pi) - math.pi

            #         return self.client.rotateToYawAsync(math.degrees(target_yaw_rad), duration_cmd, vehicle_name=drone_name)
            #     else:
            #         # If state is not available (e.g., first step), hover
            #         return self.client.hoverAsync(vehicle_name=drone_name)
        # For movement actions
        else:  # Continuous
            vx, vy, vz, yaw_increment = action.astype(np.float64)
            yaw_rate = yaw_increment / duration_cmd  # Convert to rate
            yaw = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            return self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration_cmd, vehicle_name=drone_name, yaw_mode=yaw)

    def _compute_rewards_and_dones(self, current_obs: dict):
        """
        Computes rewards, termination, and truncation flags for the entire swarm.
        """
        # --- Reward weights (ideally move these to SwarmConfig) ---
        reward_weights = {
            'progress': 1.5,
            'obstacle': 1.0,
            'formation': 0.5,
            'step': 0.01
        }

        # --- Extract data from observations ---
        current_target_distances = current_obs['shared_observation']['target_distances']
        front_obs_distances = current_obs['shared_observation']['obstacle_distances']
        distance_matrix = current_obs['shared_observation']['inter_agent_distances']

        # --- Initialize return lists ---
        individual_rewards = [0.0] * self.n_agents
        terminateds = [False] * self.n_agents
        truncateds = [False] * self.n_agents

        # --- Global Done Conditions ---
        max_steps_reached = self.step_count >= self.config.max_steps
        swarm_dispersed = np.max(distance_matrix) > self.config.max_formation_distance
        goal_reached = np.any(current_target_distances < self.config.goal_threshold)

        # --- Per-Agent Reward and Truncation Calculation ---
        for i in range(self.n_agents):
            # 1. Progress Reward: Dense reward for getting closer to the goal.
            dist_improvement = self.last_target_distances[i] - current_target_distances[i]
            progress_reward = dist_improvement * reward_weights['progress']

            # 2. Obstacle Penalty: Quadratic penalty for being too close to obstacles.
            obstacle_penalty = 0.0
            if front_obs_distances[i] < self.config.obstacle_threshold:
                violation = (self.config.obstacle_threshold - front_obs_distances[i]) / self.config.obstacle_threshold
                obstacle_penalty = (violation ** 2) * reward_weights['obstacle']

            # 3. Formation Penalty: Quadratic penalty for being too close to other agents.
            formation_penalty = 0.0
            for j in range(self.n_agents):
                if i == j: continue
                if distance_matrix[i, j] < self.config.min_distance_threshold:
                    violation = (self.config.min_distance_threshold - distance_matrix[i, j]) / self.config.min_distance_threshold
                    formation_penalty += (violation ** 2) * reward_weights['formation']
            
            # 4. Step Penalty: Small cost for taking a step to encourage efficiency.
            step_penalty = reward_weights['step']

            # --- Combine and Clip Reward ---
            total_reward = progress_reward - obstacle_penalty - formation_penalty - step_penalty
            individual_rewards[i] = np.clip(total_reward, -1.0, 1.0)

            # --- Per-Agent Truncation ---
            agent_too_far = current_target_distances[i] > (self.initial_target_distances[i] * 2.0)
            
            # --- Set Final Done Flags ---
            terminateds[i] = goal_reached
            truncateds[i] = agent_too_far or max_steps_reached or swarm_dispersed

        # Update the last known target distances for the next step's calculation
        self.last_target_distances = current_target_distances.copy()

        return individual_rewards, terminateds, truncateds

    def _get_infos(self, obs: dict) -> List[dict]:
        """Creates the info dictionary for each agent for the current step."""
        infos = []
        for i in range(self.n_agents):
            info = {
                'episode_step': self.step_count,
                'collisions': self.collision_counters[i],
                'target_distance': float(obs['shared_observation']['target_distances'][i]),
                'image_failed': self.image_failures[i] if hasattr(self, 'image_failures') else False,
            }
            infos.append(info)
        return infos

    def end_swarm_actions(self):
        for i, name in enumerate(self.drone_names):
            self.client.cancelLastTask(vehicle_name=name)

    def close(self):
        """Clean up resources and save episode logs to a JSON file."""
        # Save only per-episode collision data
        episode_data = {
            'episode_collisions': self.collision_counters.copy(),
            'n_agents': self.n_agents
        }
        self.episode_collision_log.append(episode_data)
        
        try:
            with open("training_log.json", "w") as f:
                json.dump(self.episode_collision_log, f, indent=4)
            print("Successfully saved training log to training_log.json")
        except Exception as e:
            print(f"Error saving training log: {e}")

        # Resume simulation and clean up AirSim
        self.client.simPause(False)
        self.client.reset()
        for name in self.drone_names:
            self.client.armDisarm(False, vehicle_name=name)
            self.client.enableApiControl(False, vehicle_name=name)
