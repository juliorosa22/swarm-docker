import math
import torch
import airsim
import json
from typing import Optional, List
import time
import numpy as np
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite,
    Unbounded,
    Bounded,
    Categorical,
)

from .swarm_config import SwarmConfig

class SwarmTorchEnv(EnvBase):
    """
    A TorchRL-native, high-performance swarm environment for AirSim, inheriting from EnvBase.
    This environment uses torch tensors for state management and is designed for optimal
    compatibility with the TorchRL framework.
    """

    def __init__(
        self,
        config: SwarmConfig,
        frame_skip_duration: float = 0.5,
        use_lidar: bool = False,
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
        training_file: Optional[str] = "training_log_torch.json",
    ):
        super().__init__(device=device)
        print("using device:", self.device)
        self.config = config
        self.n_agents = config.n_agents
        self.drone_names = [f"uav{i}" for i in range(self.n_agents)]
        self.frame_skip_duration = frame_skip_duration
        self.act_duration = 1.0
        self.use_lidar = use_lidar

        # Centralized AirSim client
        self.client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.client.confirmConnection()

        # Define specs before initializing state
        self._define_specs()

        # State variables (as tensors)
        self.step_count = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.start_positions = torch.zeros((self.n_agents, 3), device=self.device)
        self.end_positions = torch.zeros((self.n_agents, 3), device=self.device)
        self.initial_target_distances = torch.full((self.n_agents,), 100.0, device=self.device)
        self.last_target_distances = torch.zeros(self.n_agents, device=self.device)
        self.collision_counters = torch.zeros(self.n_agents, dtype=torch.int32, device=self.device)
        self.episode_collision_log = []

        # Pause simulation on init
        #self.client.simPause(True)

    def _define_specs(self):
        """Define observation, action, and reward specs for TorchRL."""
        img_height = self.config.img_height if hasattr(self.config, 'img_height') else 64
        img_width = self.config.img_width if hasattr(self.config, 'img_width') else 64

        # Observation Specs
        agent_obs_spec = Composite({
            "depth_image": Bounded(
                low=0, high=255, shape=(self.n_agents, 1, img_height, img_width), device=self.device
            ),
            "position": Unbounded(shape=(self.n_agents, 3), device=self.device),
            "rotation": Unbounded(shape=(self.n_agents, 3), device=self.device),
            "velocity": Unbounded(shape=(self.n_agents, 3), device=self.device),
            "target_distance": Unbounded(shape=(self.n_agents, 1), device=self.device),
            "front_obs_distance": Unbounded(shape=(self.n_agents, 1), device=self.device),
        })

        shared_obs_spec = Composite({
            "inter_agent_distances": Unbounded(shape=(self.n_agents, self.n_agents), device=self.device),
            "target_distances": Unbounded(shape=(self.n_agents,), device=self.device),
            "velocities": Unbounded(shape=(self.n_agents, 3), device=self.device),
            "obstacle_distances": Unbounded(shape=(self.n_agents,), device=self.device),
        })

        self.observation_spec = Composite({
            "agents": agent_obs_spec,
            "shared_observation": shared_obs_spec
        }).to(self.device)

        # Action Spec
        if self.config.action_type == "discrete":
            # Use Categorical for discrete actions, which is more appropriate.
            # n=8 corresponds to the 8 possible actions in _map_int_act.
            self.action_spec = Categorical(
                n=8, shape=(self.n_agents,), device=self.device
            )
        else: # continuous
            self.action_spec = Bounded(
                low=torch.tensor([-1.0, -1.0, -1.0, -1.0], device=self.device).repeat(self.n_agents, 1),
                high=torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device).repeat(self.n_agents, 1),
                shape=(self.n_agents, 4),
                device=self.device,
            )

        # Reward and Done Specs
        self.reward_spec = Unbounded(shape=(1,), device=self.device)
        self.done_spec = Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device)

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Resets the environment and returns the initial observation."""
        # --- Log data from the previous episode before resetting ---
        # (Only if it's not the very first run of the environment)
        if self.step_count.item() > 0:
            episode_data = {
                'episode_number': len(self.episode_collision_log) + 1,
                'episode_collisions': self.collision_counters.cpu().numpy().tolist(),
                'steps_in_episode': self.step_count.item()
            }
            self.episode_collision_log.append(episode_data)

        # --- Reset state for the new episode ---
        self.step_count.zero_()
        self.collision_counters.zero_()

        start_pos_np, end_pos_np = self.config.generate_positions()
        self.start_positions = torch.tensor(start_pos_np, dtype=torch.float32, device=self.device)
        self.end_positions = torch.tensor(end_pos_np, dtype=torch.float32, device=self.device)

        self.initial_target_distances = torch.norm(self.start_positions - self.end_positions, dim=1)
        self.last_target_distances = self.initial_target_distances.clone()

        #self.client.simPause(False)
        for i, name in enumerate(self.drone_names):
            x,y,z = self.start_positions[i].cpu().numpy().tolist()
            position = airsim.Vector3r(x, y, z)
            orientation = airsim.Quaternionr(0, 0, -0.707, 0.707)
            pose = airsim.Pose(position, orientation)
            self.client.simSetVehiclePose(pose, True, vehicle_name=name)
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
        #self.client.simPause(True)

        obs_td = self._get_swarm_observation_torch()
        self.last_target_distances = obs_td["shared_observation", "target_distances"].clone()

        return obs_td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Performs one step in the environment.
        """
        actions = tensordict.get("action", None)

        # If no action is provided (e.g., during check_env_specs), sample a random one.
        if actions is None:
            actions = self.action_spec.rand()

        # The rest of your logic remains the same
        if actions.ndim == 0:
            actions = actions.unsqueeze(0)
        
        actions_np = actions.cpu().numpy()
        
        # Step the underlying AirSim environment
        for i, name in enumerate(self.drone_names):
            self._execute_action(name, i, actions[i])

        self.step_count += 1
        
        # Unpause, wait, and re-pause
        #self.client.simPause(False)
        self.client.simContinueForTime(self.act_duration)
        self.end_swarm_actions()
        #time.sleep(self.frame_skip_duration)

        #self.client.simPause(True)

        collisions_this_step = self._check_collisions_torch()
        obs_td = self._get_swarm_observation_torch()
        rewards, terminateds, truncateds = self._compute_rewards_and_dones_torch(obs_td, collisions_this_step)

        total_reward = rewards.sum()
        done = terminateds.any() or truncateds.any()

        # Populate the output tensordict
        # The observation keys from obs_td must be at the top level to match observation_spec
        tensordict_out = obs_td.clone()  # Start with the observation tensordict
        tensordict_out.set("reward", total_reward.unsqueeze(0))
        tensordict_out.set("done", done.unsqueeze(0))

        return tensordict_out

    def _get_swarm_observation_torch(self) -> TensorDict:
        """Gets and composes observations for all drones into a TensorDict."""
        img_height = self.config.img_height if hasattr(self.config, 'img_height') else 64
        img_width = self.config.img_width if hasattr(self.config, 'img_width') else 64

        # Get states for all drones
        states = {name: self.client.getMultirotorState(vehicle_name=name) for name in self.drone_names}
        
        
        # Process data
        positions_list, velocities_list, rotations_list, depth_images_list = [], [], [], []
        for i, name in enumerate(self.drone_names):
            state = states[name]
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            rot = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
            positions_list.append([pos.x_val, pos.y_val, pos.z_val])
            velocities_list.append([vel.x_val, vel.y_val, vel.z_val])
            rotations_list.append(rot)

            # --- Corrected Image Request & Processing Logic ---
            img_responses = self.client.simGetImages([
                airsim.ImageRequest("front-center", airsim.ImageType.DepthVis, pixels_as_float=True)
            ], vehicle_name=name)
            response = img_responses[0]

            if response and response.image_data_float:
                # Convert list of floats to a tensor
                depth_tensor = torch.tensor(response.image_data_float, dtype=torch.float32, device=self.device)
                
                # Reshape to (channels, height, width) which is standard for PyTorch
                depth_img = depth_tensor.reshape(1, response.height, response.width)
                
                # Clip the values to a max distance (e.g., 100m) as in the reference
                depth_img = torch.clamp(depth_img, 0.0, 100.0)
            else:
                print(f"Warning: Agent {name} received an EMPTY image response. Check camera name in settings.json. Using default image.")
                depth_img = torch.zeros((1, img_height, img_width), dtype=torch.float32, device=self.device)
            
            depth_images_list.append(depth_img)

        # Convert to tensors
        positions = torch.tensor(positions_list, dtype=torch.float32, device=self.device)
        velocities = torch.tensor(velocities_list, dtype=torch.float32, device=self.device)
        rotations = torch.tensor(rotations_list, dtype=torch.float32, device=self.device)
        depth_images = torch.stack(depth_images_list).to(self.device)
        front_obs_distances = torch.tensor(self._get_front_obstacle_distances(), dtype=torch.float32, device=self.device)

        # Calculate shared observations
        distance_matrix = torch.cdist(positions, positions)
        agent_target_distances = torch.norm(positions - self.end_positions, dim=1)

        # Assemble the final observation tensordict
        obs = TensorDict({
            "agents": {
                "depth_image": depth_images,
                "position": positions,
                "rotation": rotations,
                "velocity": velocities,
                "target_distance": agent_target_distances.unsqueeze(-1),
                "front_obs_distance": front_obs_distances.unsqueeze(-1),
            },
            "shared_observation": {
                "inter_agent_distances": distance_matrix,
                "target_distances": agent_target_distances,
                "velocities": velocities,
                "obstacle_distances": front_obs_distances,
            }
        }, batch_size=[], device=self.device)
        return obs

    def _compute_rewards_and_dones_torch(self, current_obs: TensorDict, collisions_this_step: torch.Tensor):
        """Computes rewards and dones for the swarm using torch tensors."""
        reward_weights = {
            'progress': 1.5, 'obstacle': 1.0, 'formation': 0.5, 'step': 0.01
        }

        current_target_distances = current_obs["shared_observation", "target_distances"]
        front_obs_distances = current_obs["shared_observation", "obstacle_distances"]
        distance_matrix = current_obs["shared_observation", "inter_agent_distances"]

        # 1. Progress Reward
        dist_improvement = self.last_target_distances - current_target_distances
        progress_reward = dist_improvement * reward_weights['progress']

        # 2. Obstacle Penalty
        obs_threshold = self.config.obstacle_threshold
        obs_violation = torch.clamp((obs_threshold - front_obs_distances) / obs_threshold, min=0)
        obstacle_penalty = (obs_violation ** 2) * reward_weights['obstacle']

        # 3. Formation Penalty
        form_threshold = self.config.min_distance_threshold
        form_violation = torch.clamp((form_threshold - distance_matrix) / form_threshold, min=0)
        torch.diagonal(form_violation).fill_(0) # Ignore self-distance
        formation_penalty = torch.sum(form_violation ** 2, dim=1) * reward_weights['formation']

        # 4. Collision Penalty
        #collision_penalty = collisions_this_step.float() * reward_weights['collision']

        # 5. Step Penalty
        step_penalty = reward_weights['step']

        # Combine and clip rewards
        total_rewards = progress_reward - obstacle_penalty - formation_penalty  - step_penalty
        individual_rewards = torch.clamp(total_rewards, -1.0, 1.0)

        # Done conditions
        max_steps_reached = self.step_count >= self.config.max_steps
        swarm_dispersed = torch.tensor(False, device=self.device)
        if self.n_agents > 1:
            # Get indices of the upper triangle, excluding the diagonal (k=1)
            upper_triangle_indices = torch.triu_indices(self.n_agents, self.n_agents, offset=1, device=self.device)
            # Select the unique distances from the distance matrix
            unique_distances = distance_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
            # Calculate the mean
            mean_swarm_distance = unique_distances.mean()
            swarm_dispersed = mean_swarm_distance > self.config.max_formation_distance
        
        goal_reached = torch.any(current_target_distances < self.config.goal_threshold)
        agent_too_far = current_target_distances > (self.initial_target_distances * 2.0)

        # Print terminated conditions
        if goal_reached.item():
            print("Terminated: Goal reached!")
        if max_steps_reached.item():
            print("Truncated: Max steps reached!")
        if swarm_dispersed.item():
            print("Truncated: Swarm dispersed!")
        if agent_too_far.any().item():
            too_far_agents = torch.where(agent_too_far)[0].cpu().numpy()
            print(f"Truncated: Agents too far from start: {too_far_agents}")

        terminateds = torch.full((self.n_agents,), goal_reached, device=self.device)
        truncateds = (agent_too_far | max_steps_reached | swarm_dispersed )

        self.last_target_distances = current_target_distances.clone()
        return individual_rewards, terminateds, truncateds

    def _check_collisions_torch(self) -> torch.Tensor:
        """Checks for collisions and returns a boolean tensor."""
        collisions_this_step = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        for i, name in enumerate(self.drone_names):
            if self.client.simGetCollisionInfo(vehicle_name=name).has_collided:
                self.collision_counters[i] += 1
                collisions_this_step[i] = True
        return collisions_this_step

    def _execute_action(self, drone_name: str, agent_index: int, action: torch.Tensor):
        """Maps a tensor action to an AirSim command."""
        if self.config.action_type == "discrete":
            action_idx = action.item()
            act_vec = self._map_int_act(action_idx)
            vx, vy, vz, yaw = float(act_vec[0]), float(act_vec[1]), float(act_vec[2]), float(act_vec[3])
            yaw_rate = yaw / self.act_duration
        else: # continuous
            act_vec = 5*action.cpu().numpy()
            vx, vy, vz, yaw_increment = float(act_vec[0]), float(act_vec[1]), float(act_vec[2]), float(act_vec[3])
            yaw_rate = yaw_increment / self.act_duration

        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        self.client.moveByVelocityAsync(vx, vy, vz, self.act_duration, vehicle_name=drone_name, yaw_mode=yaw_mode)

    def _map_int_act(self, action_idx: int) -> list:
        """Maps an integer action to a velocity/yaw vector."""
        mapping = [
            [3.0, 0.0, 0.0, 0.0],  # Forward
            [0.0, 3.0, 0.0, 0.0],  # Right
            [0.0, 0.0, 3.0, 0.0],  # Down
            [-3.0, 0.0, 0.0, 0.0], # Backward
            [0.0, -3.0, 0.0, 0.0],# Left
            [0.0, 0.0, -3.0, 0.0],# Up
            [0.0, 0.0, 0.0, 45.0],# Yaw Right
            [0.0, 0.0, 0.0, -45.0],# Yaw Left
        ]
        return mapping[action_idx] if 0 <= action_idx < len(mapping) else [0.0, 0.0, 0.0, 0.0]

    def _get_front_obstacle_distances(self) -> List[float]:
        """Gets front obstacle distance for all agents. (Kept as is for simplicity with AirSim API)."""
        # This function's logic remains the same as it's heavily tied to the AirSim API structure.
        # It returns a list of floats, which is then converted to a tensor.
        all_min_distances = []
        max_obs_dist = self.config.max_obstacle_distance if hasattr(self.config, 'max_obstacle_distance') else 50.0
        sensor_names = ["DistanceSensor1", "DistanceSensor2", "DistanceSensor3"]
        for name in self.drone_names:
            try:
                sensor_distances = [min(d.distance, max_obs_dist) for d in 
                                    [self.client.getDistanceSensorData(s, name) for s in sensor_names] if d.distance > 0]
                min_distance = min(sensor_distances) if sensor_distances else max_obs_dist
                all_min_distances.append(min_distance)
            except Exception:
                all_min_distances.append(max_obs_dist)
        return all_min_distances

    def _set_seed(self, seed: Optional[int]):
        """Sets the seed for the environment."""
        if seed is not None:
            torch.manual_seed(seed)

    def close(self, **kwargs):
        """Cleans up resources and saves episode logs."""
        # The **kwargs is added to match the base EnvBase signature and handle extra arguments.
        super().close(**kwargs) # It's good practice to call the parent's close method.
        
        # --- Log the final, possibly incomplete, episode ---
        if self.step_count.item() > 0:
            final_episode_data = {
                'episode_number': len(self.episode_collision_log) + 1,
                'episode_collisions': self.collision_counters.cpu().numpy().tolist(),
                'steps_in_episode': self.step_count.item()
            }
            self.episode_collision_log.append(final_episode_data)
        
        # --- Save the complete log for all episodes ---
        try:
            with open(self.training_file, "w") as f:
                json.dump(self.episode_collision_log, f, indent=4)
            print(f"Successfully saved training log for {len(self.episode_collision_log)} episodes to {self.training_file}")
        except Exception as e:
            print(f"Error saving training log: {e}")

        #self.client.simPause(False)
        #self.client.reset()
        for name in self.drone_names:
            self.client.armDisarm(False, vehicle_name=name)
            self.client.enableApiControl(False, vehicle_name=name)

    def end_swarm_actions(self):
            for i, name in enumerate(self.drone_names):
                self.client.cancelLastTask(vehicle_name=name)