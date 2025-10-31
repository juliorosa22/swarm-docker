import airsim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from typing import Dict, List, Tuple, Optional, Union
from torchrl.data import Composite, Unbounded, Bounded
from .uav_deprecated import UAVEnv
import numpy as np

class SwarmEnvComposer(EnvBase):
    """
    Composes multiple UAVEnv instances into a single swarm environment.
    Uses nested TensorDict structure for cleaner agent data access.
    """
    
    def __init__(
        self,
        n_agents: int = 5,
        action_type: str = "continuous",
        start_positions: Optional[List[Tuple[float, float, float]]] = None,
        end_positions: Optional[List[Tuple[float, float, float]]] = None,
        observation_img_size: Tuple[int, int] = (64, 64),
        close_distance_threshold: float = 2.0,
        far_distance_threshold: float = 5.0,
        max_swarm_distance: float = 10.0,
        max_steps: int = 500,
        device: Optional[torch.device] = None,
    ):
        # Set device to CUDA if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Swarm Using device: {self.device}")
        
        super().__init__()
        self.n_agents = n_agents
        
        # Generate default positions if not provided or set tested positions
        if start_positions is None:
            start_positions = [
                (-3.5, 3, -10),
                (-2, 1.5, -10),
                (0, 0, -10),
                (2, 1.5, -10),
                (3.5, 3, -10)
            ]#self._generate_default_positions(n_agents, is_start=True)
        if end_positions is None:
            end_positions = [
                (-3.5, 3-210, -10),
                (-2, 1.5-210, -10),
                (0, -210, -10),
                (2, 1.5-210, -10),
                (3.5, 3-210, -10)
            ]
            
        # Create individual UAV environments
        self.client = airsim.MultirotorClient(ip='127.0.0.1')
        self.client.confirmConnection()
        
        self.uav_envs = []
        for i in range(n_agents):
            drone_name = f"uav{i}"
            env = UAVEnv(
                drone_name=drone_name,
                client=self.client,
                start_position=start_positions[i],
                end_position=end_positions[i],
                action_type=action_type,
                observation_img_size=observation_img_size,
                device=self.device  # Pass the same device to all envs
            )
            self.uav_envs.append(env)
            
        # Environment parameters
        self.close_distance_threshold = close_distance_threshold
        self.far_distance_threshold = far_distance_threshold
        self.max_swarm_distance = max_swarm_distance
        self.max_steps = max_steps  # Store max steps for truncation
        
        # Define the composite spec for the swarm with nested structure
        self._define_swarm_spaces()
        
        # Connect to AirSim (will be shared across all UAV envs)
        #self.client = self._connect_to_airsim()
        
        # State tracking
        self.done_flags = None
        self.collision_counts = None
        self.step_count = 0  # Track current step count
    
    def _define_swarm_spaces(self):
        """Define simplified observation and action spaces for the swarm."""
        
        # Get the base specs from a single UAV environment
        base_obs_spec = self.uav_envs[0].observation_spec
        base_action_spec = self.uav_envs[0].action_spec
        
        # Create stacked specs for all agents
        stacked_obs_specs = {}
        
        # For each observation component, create a stacked version
        for key, spec in base_obs_spec.items():
            original_shape = spec.shape
            # Stack shape: (n_agents, *original_shape)
            stacked_shape = (self.n_agents, *original_shape)
            
            if hasattr(spec, 'low') and hasattr(spec, 'high'):
                # Bounded spec - need to expand low/high to match stacked shape
                if spec.low.numel() == 1:
                    # Scalar bounds - expand to stacked shape
                    low = spec.low.expand(stacked_shape)
                    high = spec.high.expand(stacked_shape)
                else:
                    # Tensor bounds - stack them for each agent
                    low = spec.low.unsqueeze(0).expand(self.n_agents, *spec.low.shape)
                    high = spec.high.unsqueeze(0).expand(self.n_agents, *spec.high.shape)
                
                stacked_obs_specs[key] = Bounded(
                    shape=stacked_shape,
                    dtype=spec.dtype,
                    low=low,
                    high=high,
                    device=self.device
                )
            else:
                # Unbounded spec
                stacked_obs_specs[key] = Unbounded(
                    shape=stacked_shape,
                    dtype=spec.dtype,
                    device=self.device
                )
        
        # Add neighbor information specs
        stacked_obs_specs["closest_neighbor1"] = Unbounded(
            shape=(self.n_agents, 3), 
            dtype=torch.float32,
            device=self.device
        )
        stacked_obs_specs["closest_neighbor2"] = Unbounded(
            shape=(self.n_agents, 3), 
            dtype=torch.float32,
            device=self.device
        )
        
        # For actions, create stacked version
        if hasattr(base_action_spec, 'low') and hasattr(base_action_spec, 'high'):
            # Expand action bounds for stacking
            if base_action_spec.low.numel() == 1:
                action_low = base_action_spec.low.expand(self.n_agents, *base_action_spec.shape)
                action_high = base_action_spec.high.expand(self.n_agents, *base_action_spec.shape)
            else:
                action_low = base_action_spec.low.unsqueeze(0).expand(self.n_agents, *base_action_spec.low.shape)
                action_high = base_action_spec.high.unsqueeze(0).expand(self.n_agents, *base_action_spec.high.shape)
            
            stacked_action_spec = Bounded(
                shape=(self.n_agents, *base_action_spec.shape),
                dtype=base_action_spec.dtype,
                low=action_low,
                high=action_high,
                device=self.device
            )
        else:
            stacked_action_spec = Unbounded(
                shape=(self.n_agents, *base_action_spec.shape),
                dtype=base_action_spec.dtype,
                device=self.device
            )
        
        # Set the specs
        self.observation_spec = Composite({
            "agents": Composite(stacked_obs_specs)
        })
        
        self.action_spec = Composite({
            "agents": stacked_action_spec
        })
        
        # Simplified reward and done specs
        self.reward_spec = Composite({
            "reward": Unbounded(shape=(1,), dtype=torch.float32, device=self.device)
        })
        
        self.done_spec = Composite({
            "done": Bounded(shape=(1,), dtype=torch.bool, low=0, high=1, device=self.device)
        })
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for the environment.
        
        Args:
            seed (Optional[int]): Random seed. If None, a random seed will be used.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # Propagate seed to individual UAV environments
            for env in self.uav_envs:
                env._set_seed(seed)



    
    def _get_observation(self):
        """
        Get observations for all agents by using each UAV's observation function
        and then adding neighbor information.
        
        Returns:
            TensorDict: Complete observations for all agents with neighbor information
        """
        # Create result TensorDict
        result = TensorDict({}, batch_size=[])
        
        # First, collect individual observations from each UAV
        individual_obs_td = []
        for i, env in enumerate(self.uav_envs):
            obs_td = env._get_observation()
            individual_obs_td.append(obs_td)
        
        # Extract positions from all agents for neighbor calculations
        all_positions = torch.stack([obs_td['obs']["position"] for obs_td in individual_obs_td])
        
        # Build nested structure for each agent with neighbor information
        for i, obs_td in enumerate(individual_obs_td):
            agent_dict = TensorDict({}, batch_size=[])
            agent_obs = TensorDict({}, batch_size=[])
            
            # Copy base observations from the 'obs' key in UAVEnv's tensordict
            for key, value in obs_td['obs'].items():
                agent_obs[key] = value
            
            # Calculate neighbor information using tensor operations
            agent_pos = all_positions[i].unsqueeze(0)  # [1, 3]
            
            # Create mask for other agents (excluding self)
            other_agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
            other_agents_mask[i] = False
            
            # Get positions of other agents
            other_positions = all_positions[other_agents_mask]  # [n_agents-1, 3]
            
            # Calculate distances to all other agents using efficient broadcasting
            dists = torch.norm(agent_pos - other_positions, dim=1)  # [n_agents-1]
            
            # Get closest neighbors
            closest_neighbor1 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
            closest_neighbor2 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
            
            if dists.shape[0] > 0:  # If there are other agents
                # Sort distances and get indices
                sorted_dists, sorted_indices = torch.sort(dists)
                
                if dists.shape[0] >= 1:
                    # Find which original agent index this corresponds to
                    other_agent_indices = torch.nonzero(other_agents_mask).squeeze()
                    original_idx = other_agent_indices[sorted_indices[0]]
                    closest_neighbor1 = all_positions[original_idx]
                
                if dists.shape[0] >= 2:
                    original_idx = other_agent_indices[sorted_indices[1]]
                    closest_neighbor2 = all_positions[original_idx]
            
            # Add neighbor information
            agent_obs["closest_neighbor1"] = closest_neighbor1
            agent_obs["closest_neighbor2"] = closest_neighbor2
            
            # Add observations to agent dict
            agent_dict["observation"] = agent_obs
            
            # Add done flag from the UAVEnv TensorDict
            agent_dict["agent_done"] = obs_td["done"]
            
            # Add reward from the UAVEnv TensorDict (will be updated during step)
            agent_dict["reward"] = obs_td["reward"]
            
            # Add collisions count from the UAVEnv TensorDict
            agent_dict["collisions"] = obs_td["collisions"]
            
            # Add to result at top level, matching the spec structure
            result[f"agent{i}"] = agent_dict
        
        # Add global done flag and step count
        global_done = all([obs_td["done"].item() for obs_td in individual_obs_td]) if individual_obs_td else False
        result["global_done"] = torch.tensor([global_done], dtype=torch.bool, device=self.device)
        result["step_count"] = torch.tensor([self.step_count], dtype=torch.int64, device=self.device)
        
        return result

    def _compute_swarm_cohesion(self, position_tensor):
        """
        Compute rewards/penalties based on swarm cohesion.
        
        Args:
            position_tensor: Tensor of positions for all agents [n_agents, 3]
            
        Returns:
            torch.Tensor: Swarm cohesion rewards/penalties for each agent
        """
        # Initialize swarm rewards as zeros
        swarm_factors = torch.zeros(self.n_agents, device=self.device)
        
        # Compute pairwise distances between all agents
        # Shape: [n_agents, 1, 3] - [1, n_agents, 3] = [n_agents, n_agents, 3]
        pos_diff = position_tensor.unsqueeze(1) - position_tensor.unsqueeze(0)
        
        # Compute Euclidean distance
        # Shape: [n_agents, n_agents]
        distances = torch.norm(pos_diff, dim=2)
        
        # Set diagonal to a large value (to ignore self-distance)
        eye_mask = torch.eye(distances.shape[0], device=self.device, dtype=torch.bool)
        distances = distances.masked_fill(eye_mask, float('inf'))
        
        # Compute penalties for being too close
        too_close_mask = distances < self.close_distance_threshold
        if torch.any(too_close_mask):
            # For each agent, calculate penalty for being too close to others
            for i in range(self.n_agents):
                # Skip if agent is done
                if self.done_flags[i]:
                    continue
                    
                # Get distances to other agents that are too close
                close_distances = distances[i][too_close_mask[i]]
                
                if close_distances.shape[0] > 0:
                    # Penalty proportional to how close the agents are
                    close_penalty = -0.5 * torch.sum(
                        (self.close_distance_threshold - close_distances) / self.close_distance_threshold
                    )
                    swarm_factors[i] += close_penalty
        
        # Compute penalties for being too far
        too_far_mask = (distances > self.far_distance_threshold) & (distances < float('inf'))
        if torch.any(too_far_mask):
            # For each agent, calculate penalty for being too far from others
            for i in range(self.n_agents):
                # Skip if agent is done
                if self.done_flags[i]:
                    continue
                    
                # Get distances to other agents that are too far
                far_distances = distances[i][too_far_mask[i]]
                
                if far_distances.shape[0] > 0:
                    # Penalty proportional to how far the agents are
                    far_penalty = -0.3 * torch.sum(
                        (far_distances - self.far_distance_threshold) / self.far_distance_threshold
                    )
                    swarm_factors[i] += far_penalty
        
        return swarm_factors

    def _compute_swarm_reward(self, individual_results, swarm_rewards, positions):
        """
        Compute total rewards for each agent considering individual and swarm rewards.
        
        Args:
            individual_results (List[TensorDict]): Results from individual UAV environments
            swarm_rewards (torch.Tensor): Rewards from swarm cohesion [n_agents]
            positions (torch.Tensor): Agent positions [n_agents, 3]
        
        Returns:
            torch.Tensor: Combined rewards for each agent [n_agents]
        """
        # Initialize rewards tensor
        total_rewards = torch.zeros(self.n_agents, device=self.device)
        
        # Calculate individual rewards
        for i, res in enumerate(individual_results):
            # Get base reward from UAV environment
            base_reward = res["reward"].item()
            
            # Add swarm cohesion reward
            swarm_reward = swarm_rewards[i].item()
            
            # Combine rewards
            total_rewards[i] = base_reward + swarm_reward
            
        return total_rewards

    def _add_neighbor_info(self, positions, agent_idx):
        """Helper function to add neighbor information to agent observations."""
        # Get base observation from UAV environment
        base_obs = self.uav_envs[agent_idx]._get_observation()
        
        # Calculate neighbor information
        agent_pos = positions[agent_idx].unsqueeze(0)
        other_agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        other_agents_mask[agent_idx] = False
        
        other_positions = positions[other_agents_mask]
        dists = torch.norm(agent_pos - other_positions, dim=1)
        
        # Sort distances and get indices
        sorted_dists, sorted_indices = torch.sort(dists)
        
        # Get closest neighbors
        closest_neighbor1 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
        closest_neighbor2 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
        
        if dists.shape[0] >= 1:
            closest_neighbor1 = other_positions[sorted_indices[0]]
        if dists.shape[0] >= 2:
            closest_neighbor2 = other_positions[sorted_indices[1]]
        
        # Create observation dictionary with consistent structure
        return TensorDict({
            "base_obs": base_obs,  # Contains position, rotation, velocity, rgb_image
            "swarm_obs": {
                "closest_neighbor1": closest_neighbor1,
                "closest_neighbor2": closest_neighbor2
            }
        }, batch_size=[])


    def _stack_agent_observations(self, individual_results, include_rewards=False):
        """Stack individual agent observations into tensors."""
        
        # Get all agent positions for neighbor calculations
        all_positions = torch.stack([
            result['obs']["position"] for result in individual_results
        ])
        
        # Initialize stacked observation dict
        stacked_obs = TensorDict({}, batch_size=torch.Size([]))
        
        # Stack each observation component
        obs_keys = ["depth_image", "position", "rotation", "velocity"]
        
        for key in obs_keys:
            # Stack observations from all agents
            stacked_tensor = torch.stack([
                result['obs'][key] for result in individual_results
            ])  # Shape: (n_agents, *original_shape)
            stacked_obs[key] = stacked_tensor
        
        # Calculate and add neighbor information
        neighbor1_stack = []
        neighbor2_stack = []
        
        for i in range(self.n_agents):
            agent_pos = all_positions[i]
            
            # Calculate distances to other agents
            other_agents_mask = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
            other_agents_mask[i] = False
            other_positions = all_positions[other_agents_mask]
            
            if other_positions.shape[0] > 0:
                dists = torch.norm(agent_pos.unsqueeze(0) - other_positions, dim=1)
                sorted_dists, sorted_indices = torch.sort(dists)
                
                # Get original indices - handle both 0-D and 1-D cases
                other_agent_indices = torch.nonzero(other_agents_mask, as_tuple=False).squeeze(-1)
                
                # Ensure other_agent_indices is at least 1-D
                if other_agent_indices.dim() == 0:
                    other_agent_indices = other_agent_indices.unsqueeze(0)
                
                # Get closest neighbors
                if sorted_indices.shape[0] >= 1 and other_agent_indices.shape[0] >= 1:
                    neighbor1_idx = other_agent_indices[sorted_indices[0]]
                    neighbor1 = all_positions[neighbor1_idx]
                else:
                    neighbor1 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
                    
                if sorted_indices.shape[0] >= 2 and other_agent_indices.shape[0] >= 2:
                    neighbor2_idx = other_agent_indices[sorted_indices[1]]
                    neighbor2 = all_positions[neighbor2_idx]
                else:
                    neighbor2 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
            else:
                # No other agents
                neighbor1 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
                neighbor2 = torch.full((3,), float('inf'), dtype=torch.float32, device=self.device)
            
            neighbor1_stack.append(neighbor1)
            neighbor2_stack.append(neighbor2)
        
        # Stack neighbor information
        stacked_obs["closest_neighbor1"] = torch.stack(neighbor1_stack)  # Shape: (n_agents, 3)
        stacked_obs["closest_neighbor2"] = torch.stack(neighbor2_stack)  # Shape: (n_agents, 3)
        
        return stacked_obs

    def _reset(self, tensordict=None):
        """Reset all UAV environments and return stacked observations."""
        # Reset state tracking variables
        self.done_flags = [False] * self.n_agents
        self.collision_counts = [0] * self.n_agents
        self.step_count = 0
        
        # Reset all individual UAVs
        individual_obs = []
        self.client.reset()
        for env in self.uav_envs:
            obs = env._reset(tensordict)
            individual_obs.append(obs)
        
        # Stack all agent observations
        stacked_obs = self._stack_agent_observations(individual_obs)
        
        # Create result TensorDict
        result = TensorDict({
            "agents": stacked_obs,
            "reward": torch.tensor([0.0], dtype=torch.float32, device=self.device),
            "done": torch.tensor([False], dtype=torch.bool, device=self.device)
        },batch_size=torch.Size([]))
        
        return result

    def _step(self, tensordict):
        """Execute one step for all agents with stacked observations."""
        self.step_count += 1
        
        # Extract stacked actions
        stacked_actions = tensordict["agents"]  # Shape: (n_agents, action_dim)
        
        # Forward actions to individual UAVs
        individual_results = []
        for i in range(self.n_agents):
            # Extract action for this agent
            agent_action = stacked_actions[i]  # Shape: (action_dim,)
            
            # Create agent-specific tensordict
            agent_td = TensorDict({"action": agent_action}, batch_size=torch.Size([]))
            
            # Step the individual UAV environment
            result = self.uav_envs[i]._step(agent_td)
            individual_results.append(result)
        
        # Stack observations from individual results
        stacked_obs = self._stack_agent_observations(individual_results, include_rewards=True)
        
        # Calculate total reward and done flags
        total_reward = sum(result["reward"].item() for result in individual_results)
        all_done = all(result["done"].item() for result in individual_results)
        
        # Check for truncation due to max steps
        truncated = self.step_count >= self.max_steps
        done = all_done or truncated
        
        # Create result TensorDict
        result = TensorDict({
            "agents": stacked_obs,
            "reward": torch.tensor([total_reward], dtype=torch.float32, device=self.device),
            "done": torch.tensor([done], dtype=torch.bool, device=self.device)
        }, batch_size=torch.Size([]))
        
        return result