import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
import airsim
from .uav import UAVGymEnv
from .swarm_config import SwarmConfig  # Add this import
from .uav_config import UAVConfig  # Ensure this is imported
import time

class SwarmGymEnv(gym.Env):
    """
    A Gymnasium environment that composes multiple UAVGymEnv instances into a single swarm environment.
    Uses distance graph observation space and balanced swarm rewards.
    Suitable for Multi-Agent Reinforcement Learning (MARL) algorithms.
    """

    def __init__(self, config: SwarmConfig):
        super().__init__()
        
        # Store the config for direct access
        self.config = config
        self.n_agents = config.n_agents
        # Generate UAV configs
        uav_configs = config.create_uav_configs()
        
        # Create shared AirSim client
        self.client = airsim.MultirotorClient(ip='127.0.0.1')
        self.client.confirmConnection()
        
        # Create individual UAV environments using UAVConfig
        self.uav_envs = []
        for uav_config in uav_configs:
            uav_env = UAVGymEnv(client=self.client, config=uav_config)
            self.uav_envs.append(uav_env)
        
        # Define swarm observation and action spaces
        self._define_spaces()
        
        # State tracking
        self.step_count = 0
        self.done_flags = [False] * self.config.n_agents
        self.collision_counts = [0] * self.config.n_agents

        self.individual_observations = [self.uav_envs[i]._get_default_obs() for i in range(self.config.n_agents)]  # Initialize with default obs

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
                shape=(self.config.n_agents, self.config.n_agents),
                dtype=np.float32
            ),
            # Distance to target for each agent
            'target_distances': spaces.Box(
                low=0.0,
                high=1000.0,
                shape=(self.config.n_agents,),
                dtype=np.float32
            ),
            # Velocity for each agent (for coordination)
            'velocities': spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.config.n_agents, 3),
                dtype=np.float32
            ),
            # Obstacle distances for each agent
            'obstacle_distances': spaces.Box(
                low=0.0,
                high=100.0,
                shape=(self.config.n_agents,),
                dtype=np.float32
            ),
        })
        
        # Swarm action space: list of individual action spaces
        self.action_space = spaces.Tuple([base_action_space] * self.config.n_agents)
    
    def _create_distance_graph_observation(self):
        """Create distance graph representation from individual UAV observations."""
        # Extract positions and other data
        positions = np.array([obs['position'] for obs in self.individual_observations])
        # Create inter-agent distance matrix
        distance_matrix = np.zeros((self.config.n_agents, self.config.n_agents), dtype=np.float32)
        for i in range(self.config.n_agents):
            for j in range(self.config.n_agents):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])
                else:
                    distance_matrix[i, j] = 0.0  # Self-distance is 0
        
        # Agent status (1.0 = active, 0.0 = done)
        
        
        return distance_matrix
    
    def _compute_safety_penalty(self, positions, agent_idx):
        """Compute safety penalty with smoother scaling."""
        penalty = 0.0
        agent_pos = positions[agent_idx]
        
        for j in range(self.config.n_agents):
            if agent_idx != j and not self.done_flags[j]:
                distance = np.linalg.norm(agent_pos - positions[j])
                if distance < self.config.min_distance_threshold:
                    # Smoother penalty function
                    violation_ratio = (self.config.min_distance_threshold - distance) / self.config.min_distance_threshold
                    # Use sqrt for gentler penalty
                    single_penalty = (violation_ratio ** 0.5) * self.config.safety_penalty_weight
                    penalty += single_penalty
        
        return min(penalty, self.config.max_safety_penalty)
    
    def _compute_formation_bonus(self, positions, agent_idx):
        """Compute formation bonus with smoother scaling."""
        if self.done_flags[agent_idx]:
            return 0.0
        
        agent_pos = positions[agent_idx]
        active_neighbors = []
        
        for j in range(self.config.n_agents):
            if agent_idx != j and not self.done_flags[j]:
                distance = np.linalg.norm(agent_pos - positions[j])
                if distance <= self.config.max_formation_distance:
                    active_neighbors.append(distance)
        
        if len(active_neighbors) > 0:
            avg_neighbor_distance = np.mean(active_neighbors)
            ideal_distance = (self.config.min_distance_threshold + self.config.max_formation_distance) / 2
            
            # Smoother bonus calculation
            distance_error = abs(avg_neighbor_distance - ideal_distance)
            normalized_error = distance_error / ideal_distance
            
            # Use quadratic falloff for smoother gradients
            bonus = max(0, 1 - normalized_error**2) * self.config.formation_bonus_weight
            
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
            return (avg_progress ** 0.75) * self.config.coordination_bonus_weight
        
        return 0.0
    
    def _compute_swarm_rewards(self, individual_rewards):
        """Compute balanced swarm rewards combining individual rewards with swarm factors."""
        individual_observations = self.individual_observations  # Use the cached observations
        positions = np.array([obs['position'] for obs in individual_observations])
        swarm_rewards = []
        
        for i in range(self.config.n_agents):
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
        """Reset all UAVs in the swarm with positions from config. Returns (graph_observation, infos)."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.done_flags = [False] * self.config.n_agents
        self.collision_counts = [0] * self.config.n_agents
        self.individual_observations = []
        infos = []
        print("Resetting Swarm Environment...")
        # Generate or get positions from config (modular call)
        new_start_positions, new_end_positions = self.config.generate_positions()
        
        terminateds = [False] * self.config.n_agents
        truncateds = [False] * self.config.n_agents
        rewards = [0.0] * self.config.n_agents
        
        for i, uav_env in enumerate(self.uav_envs):
            # Pass both start and end positions as a dict
            reset_options = {'reset_positions': {'start': new_start_positions[i], 'end': new_end_positions[i]}}
            obs, info = uav_env.reset(seed=seed, options=reset_options)
            terminateds[i] = info.get('terminated', False)
            truncateds[i] = info.get('truncated', False)
            self.individual_observations.append(obs)
            infos.append(info)
            self.collision_counts[i] = info.get('collisions', 0)
        
        # Create graph observation
        distance_matrix = self._create_distance_graph_observation()
        swarm_obs = {
            'inter_agent_distances': distance_matrix,
            'target_distances': np.array([obs['target_distance'][0] for obs in self.individual_observations], dtype=np.float32),
            'velocities': np.array([obs['velocity'] for obs in self.individual_observations], dtype=np.float32),
            'obstacle_distances': np.array([obs['front_obs_distance'][0] for obs in self.individual_observations], dtype=np.float32)
        }
    
        return swarm_obs, infos



    #this function is responsible for calling the _get_obs function of each uav and then save the obs in a class variable so each computation that requires the individual obs can access it without calling the _get_obs function again
    def update_swarm_obs(self):
        """Get observations from all UAVs using parallel threads. Returns list of individual observations."""
        self.individual_observations = []
        for uav_env in self.uav_envs:
            try:
                obs = uav_env._get_obs()
                self.individual_observations.append(obs)
            except Exception as e:
                print(f"Error getting obs for {uav_env.drone_name}: {e}")
                # Append a default obs or skip
                self.individual_observations.append(self.uav_env._get_default_obs())

    # This function will return the swarm env observation by calling the _get_obs function of each uav and then creating the distance graph observation
    def _get_obs(self):
        """Return the swarm env observation by updating individual observations and creating the distance graph observation."""
        try:
            self.update_swarm_obs()
            distance_matrix = self._create_distance_graph_observation()
            return {
                'inter_agent_distances': distance_matrix,
                'target_distances': np.array([obs['target_distance'][0] for obs in self.individual_observations], dtype=np.float32),
                'velocities': np.array([obs['velocity'] for obs in self.individual_observations], dtype=np.float32),
                'obstacle_distances': np.array([obs['front_obs_distance'][0] for obs in self.individual_observations], dtype=np.float32)
            }
        except Exception as e:
            print(f"Error in _get_obs: {e}")
            # Return a default swarm observation dict
            return {
                'inter_agent_distances': np.zeros((self.config.n_agents, self.config.n_agents), dtype=np.float32),
                'target_distances': np.zeros(self.config.n_agents, dtype=np.float32),
                'velocities': np.zeros((self.config.n_agents, 3), dtype=np.float32),
                'obstacle_distances': np.zeros(self.config.n_agents, dtype=np.float32)
            }
        
    
    def step(self, actions):
        """Execute one step for all UAVs using act_no_join with centralized future polling and threaded obs. Returns (graph_observation, swarm_rewards, terminateds, truncateds, infos)."""
        
        # Check if all agents are done
        if all(self.done_flags):
            swarm_obs = self._get_obs()  # Wrap the individual observations updates and return the swarm observation
            rewards = [0.0] * self.config.n_agents  # Updated
            terminateds = self.done_flags
            truncateds = [False] * self.config.n_agents  # Updated
            infos = [{'collisions': count, 'episode_step': self.step_count} for count in self.collision_counts]
            return swarm_obs, rewards, terminateds, truncateds, infos
        
        self.step_count += 1
        
        # Collect futures from act_no_join (sequential to avoid IOLoop conflicts)
        action_futures = []
        for i in range(self.config.n_agents):  # Updated
            future = self.uav_envs[i].act_no_join(actions[i])
            if future is not None:
                action_futures.append(future)
        
        # waits for all actions to complete
        #time.sleep(1)
        #this prevents getting stuck when an agent keeps colliding and its action future never completes
        #for uav in self.uav_envs:
        #    uav.end_last_action()

        # This wrap the individual observations updates and return the swarm observation
        swarm_obs = self._get_obs()
        
        # Compute individual rewards, etc. (sequential, as before)
        individual_rewards = []
        terminateds = []
        truncateds = []
        infos = []
        #now uses tha self.individual_observations variable that was updated in the _get_obs function
        for i, obs in enumerate(self.individual_observations):
            reward = self.uav_envs[i]._compute_reward(obs)
            target_distance = obs['target_distance'][0]
            current_position = obs['position']
            
            terminated = False
            if target_distance < self.uav_envs[i].goal_threshold:
                self.uav_envs[i].done = True
                terminated = True
                print(f"Goal reached by {self.uav_envs[i].drone_name}!")
            
            self.uav_envs[i].truncated = self.uav_envs[i].step_count > self.uav_envs[i].max_steps
            
            self.uav_envs[i].last_position = current_position.copy()
            
            individual_rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(self.uav_envs[i].truncated)
            infos.append({
                'collisions': self.uav_envs[i].collision_counter,
                'episode_step': self.uav_envs[i].step_count,
                'terminated': terminated,
                'truncated': self.uav_envs[i].truncated,
                'target_distance': float(target_distance)
            })
            
            self.done_flags[i] = terminated
            self.collision_counts[i] = self.uav_envs[i].collision_counter
        
        # Compute swarm rewards
        #now can use the self.individual_observations internally without calling the _get_obs function again
        swarm_rewards = self._compute_swarm_rewards(individual_rewards)
        
        
        
        # Global truncation if max steps reached
        if self.step_count >= self.config.max_steps:  # Updated
            truncateds = [True] * self.config.n_agents  # Updated
        
        # Add swarm metrics to info
        positions = np.array([obs['position'] for obs in self.individual_observations])
        for i, info in enumerate(infos):
            info.update({
                'individual_reward': individual_rewards[i],
                'swarm_reward': swarm_rewards[i],
                'swarm_reward_components': {
                    'safety_penalty': self._compute_safety_penalty(positions, i),
                    'formation_bonus': self._compute_formation_bonus(positions, i),
                    'coordination_bonus': self._compute_coordination_bonus(self.individual_observations, i)
                }
            })
        

        return swarm_obs, swarm_rewards, terminateds, truncateds, infos
    
  
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
            obs = uav_env._get_obs()
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
        for i in range(self.config.n_agents):  # Updated
            for j in range(i+1, self.config.n_agents):  # Updated
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