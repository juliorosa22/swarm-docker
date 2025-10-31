import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
import airsim
from .async_uav import AsyncUAVGymEnv  # Import the async UAV env
import asyncio  # Added for async support

class AsyncSwarmGymEnv(gym.Env):  # Renamed class for clarity
    """
    A Gymnasium environment that composes multiple AsyncUAVGymEnv instances into a single swarm environment.
    Uses distance graph observation space and balanced swarm rewards.
    Suitable for Multi-Agent Reinforcement Learning (MARL) algorithms, with async I/O for performance.
    """

    def __init__(
        self,
        n_agents: int = 5,
        action_type: str = "continuous",
        start_positions: Optional[List[Tuple[float, float, float]]] = None,
        end_positions: Optional[List[Tuple[float, float, float]]] = None,
        observation_img_size: Tuple[int, int] = (64, 64),
        obstacle_threshold: float = 2.0,
        goal_threshold: float = 1.0,
        max_target_distance: Optional[float] = None,
        max_obstacle_distance: float = 50.0,
        max_steps: int = 500,
        # New swarm-specific parameters
        min_distance_threshold: float = 3.0,
        max_formation_distance: float = 10.0,
        safety_penalty_weight: float = 0.5,
        formation_bonus_weight: float = 0.2,
        coordination_bonus_weight: float = 0.1,
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.action_type = action_type
        self.max_steps = max_steps
        
        # Better balanced swarm parameters
        self.min_distance_threshold = min_distance_threshold
        self.max_formation_distance = max_formation_distance
        self.safety_penalty_weight = 0.7      # Reduced from 0.8
        self.formation_bonus_weight = 1.0     # Reduced from 1.2  
        self.coordination_bonus_weight = 0.6  # Increased from 0.5
        
        # Reduced max penalty cap
        self.max_safety_penalty = 1.2  # Reduced from 1.5

        # Generate default positions if not provided
        if start_positions is None:
            start_positions = [(i * 5.0, 0.0, -2.0) for i in range(n_agents)]
        if end_positions is None:
            end_positions = [(i * 5.0, 50.0, -2.0) for i in range(n_agents)]
        
        # Ensure we have positions for all agents
        assert len(start_positions) == n_agents, "start_positions must match n_agents"
        assert len(end_positions) == n_agents, "end_positions must match n_agents"
        
        # Create shared AirSim client
        self.client = airsim.MultirotorClient(ip='127.0.0.1')
        self.client.confirmConnection()
        
        # Create individual UAV environments (async)
        self.uav_envs = []
        for i in range(n_agents):
            drone_name = f"uav{i}"
            uav_env = AsyncUAVGymEnv(  # Use async UAV
                client=self.client,
                drone_name=drone_name,
                start_position=start_positions[i],
                end_position=end_positions[i],
                action_type=action_type,
                observation_img_size=observation_img_size,
                obstacle_threshold=obstacle_threshold,
                goal_threshold=goal_threshold,
                max_target_distance=max_target_distance,
                max_obstacle_distance=max_obstacle_distance,
            )
            self.uav_envs.append(uav_env)
        
        # Define swarm observation and action spaces
        self._define_spaces()
        
        # State tracking
        self.step_count = 0
        self.done_flags = [False] * n_agents
        self.collision_counts = [0] * n_agents
    
    def _define_spaces(self):
        """Define observation and action spaces using distance graph representation."""
        
        # Get base action space from first UAV env
        base_action_space = self.uav_envs[0].action_space
        
        # Define distance graph observation space
        self.observation_space = spaces.Dict({
            # Inter-agent distance matrix (n_agents x n_agents)
            'inter_agent_distances': spaces.Box(
                low=0.0,
                high=1000.0,  # Max reasonable distance
                shape=(self.n_agents, self.n_agents),
                dtype=np.float32
            ),
            # Distance to target for each agent
            'target_distances': spaces.Box(
                low=0.0,
                high=1000.0,
                shape=(self.n_agents,),
                dtype=np.float32
            ),
            # Velocity for each agent (for coordination)
            'velocities': spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.n_agents, 3),
                dtype=np.float32
            ),
            # Obstacle distances for each agent
            'obstacle_distances': spaces.Box(
                low=0.0,
                high=100.0,
                shape=(self.n_agents,),
                dtype=np.float32
            ),
            # Agent status (done/active)
            'agent_status': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_agents,),
                dtype=np.float32
            )
        })
        
        # Swarm action space: list of individual action spaces
        self.action_space = spaces.Tuple([base_action_space] * self.n_agents)
    
    def _create_distance_graph_observation(self, individual_observations):
        """Create distance graph representation from individual UAV observations."""
        
        # Extract positions and other data
        positions = np.array([obs['position'] for obs in individual_observations])
        target_distances = np.array([obs['target_distance'][0] for obs in individual_observations])
        velocities = np.array([obs['velocity'] for obs in individual_observations])
        obstacle_distances = np.array([obs['front_obs_distance'][0] for obs in individual_observations])
        
        # Create inter-agent distance matrix
        distance_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])
                else:
                    distance_matrix[i, j] = 0.0  # Self-distance is 0
        
        # Agent status (1.0 = active, 0.0 = done)
        agent_status = np.array([0.0 if done else 1.0 for done in self.done_flags], dtype=np.float32)
        
        return {
            'inter_agent_distances': distance_matrix,
            'target_distances': target_distances,
            'velocities': velocities,
            'obstacle_distances': obstacle_distances,
            'agent_status': agent_status
        }
    
    def _compute_safety_penalty(self, positions, agent_idx):
        """Compute safety penalty with smoother scaling."""
        penalty = 0.0
        agent_pos = positions[agent_idx]
        
        for j in range(self.n_agents):
            if agent_idx != j and not self.done_flags[j]:
                distance = np.linalg.norm(agent_pos - positions[j])
                if distance < self.min_distance_threshold:
                    # Smoother penalty function
                    violation_ratio = (self.min_distance_threshold - distance) / self.min_distance_threshold
                    # Use sqrt for gentler penalty
                    single_penalty = (violation_ratio ** 0.5) * self.safety_penalty_weight
                    penalty += single_penalty
        
        return min(penalty, self.max_safety_penalty)
    
    def _compute_formation_bonus(self, positions, agent_idx):
        """Compute formation bonus with smoother scaling."""
        if self.done_flags[agent_idx]:
            return 0.0
        
        agent_pos = positions[agent_idx]
        active_neighbors = []
        
        for j in range(self.n_agents):
            if agent_idx != j and not self.done_flags[j]:
                distance = np.linalg.norm(agent_pos - positions[j])
                if distance <= self.max_formation_distance:
                    active_neighbors.append(distance)
        
        if len(active_neighbors) > 0:
            avg_neighbor_distance = np.mean(active_neighbors)
            ideal_distance = (self.min_distance_threshold + self.max_formation_distance) / 2
            
            # Smoother bonus calculation
            distance_error = abs(avg_neighbor_distance - ideal_distance)
            normalized_error = distance_error / ideal_distance
            
            # Use quadratic falloff for smoother gradients
            bonus = max(0, 1 - normalized_error**2) * self.formation_bonus_weight
            
            # Simpler neighbor scaling
            optimal_neighbors = 2.0
            neighbor_factor = max(0.7, 1.0 - abs(len(active_neighbors) - optimal_neighbors) / optimal_neighbors)
            
            return bonus * neighbor_factor
        
        return 0.0
    
    def _compute_coordination_bonus(self, individual_observations, agent_idx):
        """Compute coordination bonus with better scaling."""
        if self.done_flags[agent_idx]:
            return 0.0
        
        # Bonus for collective progress (all agents moving towards goals)
        total_progress = 0.0
        active_agents = 0
        
        for i, obs in enumerate(individual_observations):
            if not self.done_flags[i]:
                # Normalized progress (closer to goal = higher score)
                max_distance = self.uav_envs[i].max_target_distance
                current_distance = obs['target_distance'][0]
                # Add small epsilon to prevent division issues
                progress = max(0, (max_distance - current_distance) / (max_distance + 1e-6))
                total_progress += progress
                active_agents += 1
        
        if active_agents > 0:
            avg_progress = total_progress / active_agents
            # Square root for more pronounced coordination benefits
            return (avg_progress ** 0.75) * self.coordination_bonus_weight
        
        return 0.0
    
    def _compute_swarm_rewards(self, individual_rewards, individual_observations):
        """Compute balanced swarm rewards combining individual rewards with swarm factors."""
        
        positions = np.array([obs['position'] for obs in individual_observations])
        swarm_rewards = []
        
        for i in range(self.n_agents):
            # Start with individual reward
            base_reward = individual_rewards[i]
            
            # Add swarm components
            safety_penalty = self._compute_safety_penalty(positions, i)
            formation_bonus = self._compute_formation_bonus(positions, i)
            coordination_bonus = self._compute_coordination_bonus(individual_observations, i)
            
            # Combine all components
            total_reward = (base_reward 
                          - safety_penalty 
                          + formation_bonus 
                          + coordination_bonus)
            
            swarm_rewards.append(total_reward)
        
        return swarm_rewards
    
    def reset(self, seed=None, options=None):
        """Reset all UAVs in the swarm. Returns (graph_observation, infos)."""
        # Wrap async reset in sync for Gym compatibility
        return asyncio.run(self._async_reset(seed, options))
    
    async def _async_reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.done_flags = [False] * self.n_agents
        self.collision_counts = [0] * self.n_agents
        
        # Parallel reset using asyncio.gather
        reset_tasks = [uav_env.reset(seed=seed, options=options) for uav_env in self.uav_envs]
        results = await asyncio.gather(*reset_tasks)
        
        individual_observations = [obs for obs, _ in results]
        infos = [info for _, info in results]
        
        for i, info in enumerate(infos):
            self.collision_counts[i] = info.get('collisions', 0)
        
        # Create distance graph observation
        graph_observation = self._create_distance_graph_observation(individual_observations)
        
        return graph_observation, infos
    
    def step(self, actions):
        """Execute one step for all UAVs. Returns (graph_observation, swarm_rewards, terminateds, truncateds, infos)."""
        # Wrap async step in sync for Gym compatibility
        return asyncio.run(self._async_step(actions))
    
    async def _async_step(self, actions):
        """Async version of step for parallel execution."""
        
        # Check if all agents are done
        if all(self.done_flags):
            # Parallel observation gathering
            obs_tasks = [uav_env._get_obs() for uav_env in self.uav_envs]
            individual_observations = await asyncio.gather(*obs_tasks)
            graph_observation = self._create_distance_graph_observation(individual_observations)
            rewards = [0.0] * self.n_agents
            terminateds = self.done_flags
            truncateds = [False] * self.n_agents
            infos = [{'collisions': count, 'episode_step': self.step_count} for count in self.collision_counts]
            return graph_observation, rewards, terminateds, truncateds, infos
        
        self.step_count += 1
        
        # Parallel actions and observations using asyncio.gather
        action_tasks = [uav_env.act(action) for uav_env, action in zip(self.uav_envs, actions)]
        await asyncio.gather(*action_tasks)
        
        obs_tasks = [uav_env._get_obs() for uav_env in self.uav_envs]
        individual_observations = await asyncio.gather(*obs_tasks)
        
        # Process results (sync parts)
        individual_rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, (uav_env, obs) in enumerate(zip(self.uav_envs, individual_observations)):
            reward = uav_env._compute_reward(obs)
            target_distance = obs['target_distance'][0]
            current_position = obs['position']
            
            terminated = False
            if target_distance < uav_env.goal_threshold:
                uav_env.done = True
                terminated = True
                print(f"Goal reached by {uav_env.drone_name}!")
            
            uav_env.truncated = uav_env.step_count > uav_env.max_steps
            
            uav_env.last_position = current_position.copy()
            
            individual_rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(uav_env.truncated)
            infos.append({
                'collisions': uav_env.collision_counter,
                'episode_step': uav_env.step_count,
                'terminated': terminated,
                'truncated': uav_env.truncated,
                'target_distance': float(target_distance)
            })
            
            self.done_flags[i] = terminated
            self.collision_counts[i] = uav_env.collision_counter
        
        # Compute swarm rewards
        swarm_rewards = self._compute_swarm_rewards(individual_rewards, individual_observations)
        
        # Create distance graph observation
        graph_observation = self._create_distance_graph_observation(individual_observations)
        
        # Global truncation if max steps reached
        if self.step_count >= self.max_steps:
            truncateds = [True] * self.n_agents
        
        # Add swarm metrics to info
        for i, info in enumerate(infos):
            info.update({
                'individual_reward': individual_rewards[i],
                'swarm_reward': swarm_rewards[i],
                'swarm_reward_components': {
                    'safety_penalty': self._compute_safety_penalty(
                        np.array([obs['position'] for obs in individual_observations]), i),
                    'formation_bonus': self._compute_formation_bonus(
                        np.array([obs['position'] for obs in individual_observations]), i),
                    'coordination_bonus': self._compute_coordination_bonus(individual_observations, i)
                }
            })
        
        return graph_observation, swarm_rewards, terminateds, truncateds, infos
    
    def close(self):
        """Close all UAV environments."""
        for uav_env in self.uav_envs:
            uav_env.close()
    
    def render(self, mode='human'):
        """Render the swarm (placeholder - implement if needed)."""
        pass
    
    # Additional methods for swarm-specific logic
    def get_swarm_positions(self):
        """Get current positions of all UAVs."""
        positions = []
        for uav_env in self.uav_envs:
            obs = asyncio.run(uav_env._get_obs())  # Sync wrapper for async
            positions.append(obs['position'])
        return np.array(positions)
    
    def compute_swarm_cohesion(self):
        """Compute cohesion metric for the swarm."""
        positions = self.get_swarm_positions()
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        return np.mean(distances)
    
    def get_swarm_statistics(self):
        """Get comprehensive swarm statistics."""
        positions = self.get_swarm_positions()
        
        # Inter-agent distances
        inter_distances = []
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                inter_distances.append(dist)
        
        return {
            'mean_position': np.mean(positions, axis=0),
            'position_std': np.std(positions, axis=0),
            'cohesion': self.compute_swarm_cohesion(),
            'min_inter_distance': np.min(inter_distances) if inter_distances else float('inf'),
            'max_inter_distance': np.max(inter_distances) if inter_distances else 0.0,
            'mean_inter_distance': np.mean(inter_distances) if inter_distances else 0.0,
            'active_agents': sum(1 for done in self.done_flags if not done),
            'total_collisions': sum(self.collision_counts),
            'step_count': self.step_count
        }