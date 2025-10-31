import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
import airsim
from .swarm import SwarmGymEnv  # Import the base class
from ..swarm_config import SwarmConfig
from .uav_config import UAVConfig
import time

class FullSwarmGymEnv(SwarmGymEnv):
    """
    A Gymnasium environment that extends SwarmGymEnv but uses full stacked UAV observations instead of graph-based observations.
    Suitable for Multi-Agent Reinforcement Learning (MARL) algorithms with raw state data.
    """

    def __init__(self, config: SwarmConfig):
        super().__init__(config)
        print(f"FullSwarmGymEnv initialized with {self.config.n_agents} agents.")
        # Override the observation space to use full stacked obs
        self._define_full_spaces()

    def _define_full_spaces(self):
        """Define observation and action spaces using full stacked UAV observations."""
        
        # Get base action space from first UAV env
        base_action_space = self.uav_envs[0].action_space
        
        # Define full stacked observation space
        # Assume shapes from individual UAV obs (adjust based on UAVGymEnv)
        obs_space = self.uav_envs[0].observation_space
        
        # The observation space is the dictionary of stacked observations directly.
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(
                low=obs_space['depth_image'].low.min(),
                high=obs_space['depth_image'].high.max(),
                shape=(self.config.n_agents,) + obs_space['depth_image'].shape,
                dtype=np.float32
            ),
            'position': spaces.Box(
                low=obs_space['position'].low.min(),
                high=obs_space['position'].high.max(),
                shape=(self.config.n_agents, 3),
                dtype=np.float32
            ),
            'rotation': spaces.Box(
                low=obs_space['rotation'].low.min(),
                high=obs_space['rotation'].high.max(),
                shape=(self.config.n_agents, 3),
                dtype=np.float32
            ),
            'velocity': spaces.Box(
                low=obs_space['velocity'].low.min(),
                high=obs_space['velocity'].high.max(),
                shape=(self.config.n_agents, 3),
                dtype=np.float32
            ),
            'target_distance': spaces.Box(
                low=obs_space['target_distance'].low.min(),
                high=obs_space['target_distance'].high.max(),
                shape=(self.config.n_agents, 1),
                dtype=np.float32
            ),
            'front_obs_distance': spaces.Box(
                low=obs_space['front_obs_distance'].low.min(),
                high=obs_space['front_obs_distance'].high.max(),
                shape=(self.config.n_agents, 1),
                dtype=np.float32
            ),
        })
        
        # Swarm action space: list of individual action spaces (inherited)
        self.action_space = spaces.Tuple([base_action_space] * self.config.n_agents)

    def reset(self, seed=None, options=None):
        """Reset all UAVs in the swarm with inverted V formation. Returns (full_stacked_observation, info_dict)."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.done_flags = [False] * self.config.n_agents
        self.collision_counts = [0] * self.config.n_agents
        self.individual_observations = []
        agent_infos = []
        # Get y_leader and x_leader from options or config
        if options and 'y_leader' in options and 'x_leader' in options:
            y_leader = options['y_leader']
            x_leader = options['x_leader']
        else:
            # Use config's reset_leader to get random positions
            x_leader, y_leader = self.config.get_random_leader_positions()
        
        # Generate new start positions for inverted V formation
        new_start_positions = self._generate_inverted_v_positions(y_leader, x_leader, y_offset=0.0)

        # Generate new end positions, shifted forward (e.g., -80 in y)
        new_end_positions = self._generate_inverted_v_positions(y_leader, x_leader, y_offset=-80.0)

        for i, uav_env in enumerate(self.uav_envs):
            # Pass both start and end positions as a dict
            reset_options = {'reset_positions': {'start': new_start_positions[i], 'end': new_end_positions[i]}}
            obs, info = uav_env.reset(seed=seed, options=reset_options)
            self.individual_observations.append(obs)
            agent_infos.append(info)
            self.collision_counts[i] = info.get('collisions', 0)
   
        # Create full stacked observation
        swarm_obs = self._get_obs()  # This will return the full stacked obs
        
        # Return info as a dict to comply with Gymnasium's check_env
        info_dict = {'agent_infos': agent_infos}  # Wrap the list in a dict

        return swarm_obs, info_dict

    def _get_obs(self):
        """Return the swarm env observation by updating individual observations and creating the full stacked observation."""
        try:
            self.update_swarm_obs()
            # Create full stacked observation
            return {
                'depth_image': np.array([obs['depth_image'] for obs in self.individual_observations], dtype=np.float32),
                'position': np.array([obs['position'] for obs in self.individual_observations], dtype=np.float32),
                'rotation': np.array([obs['rotation'] for obs in self.individual_observations], dtype=np.float32),
                'velocity': np.array([obs['velocity'] for obs in self.individual_observations], dtype=np.float32),
                'target_distance': np.array([obs['target_distance'] for obs in self.individual_observations], dtype=np.float32),
                'front_obs_distance': np.array([obs['front_obs_distance'] for obs in self.individual_observations], dtype=np.float32)
            }
        except Exception as e:
            print(f"Error in _get_obs: {e}")
            # Return a default full stacked observation dict
            # Assume default shapes; adjust based on actual UAV obs
            obs_space = self.uav_envs[0].observation_space
            return {
                'depth_image': np.zeros((self.config.n_agents,) + obs_space['depth_image'].shape, dtype=np.float32),
                'position': np.zeros((self.config.n_agents, 3), dtype=np.float32),
                'rotation': np.zeros((self.config.n_agents, 3), dtype=np.float32),
                'velocity': np.zeros((self.config.n_agents, 3), dtype=np.float32),
                'target_distance': np.zeros((self.config.n_agents, 1), dtype=np.float32),
                'front_obs_distance': np.zeros((self.config.n_agents, 1), dtype=np.float32)
            }
        
    def step(self, actions):
        """Execute one step for all UAVs using act_no_join with centralized future polling and threaded obs. Returns (full_stacked_observation, reward, terminated, truncated, info)."""
        
        # Check if all agents are done
        if all(self.done_flags):
            swarm_obs = self._get_obs()  # Wrap the individual observations updates and return the swarm observation
            reward = 0.0
            terminated = True
            truncated = True # Or False, depending on desired logic for this edge case
            infos = [{'collisions': count, 'episode_step': self.step_count} for count in self.collision_counts]
            info_dict = {'agent_infos': infos}
            return swarm_obs, reward, terminated, truncated, info_dict

        self.step_count += 1
        
        # Collect futures from act_no_join (sequential to avoid IOLoop conflicts)
        action_futures = []
        for i in range(self.config.n_agents):
            future = self.uav_envs[i].act_no_join(actions[i])
            if future is not None:
                action_futures.append(future)
        
        # waits for all actions to complete
        time.sleep(1)
        #this prevents getting stuck when an agent keeps colliding and its action future never completes
        for uav in self.uav_envs:
            uav.end_last_action()

        # This wrap the individual observations updates and return the swarm observation
        swarm_obs = self._get_obs()
        
        # Compute individual rewards, etc. (sequential, as before)
        individual_rewards = []
        terminateds = []
        truncateds = []
        infos = []
        # Now uses the self.individual_observations variable that was updated in the _get_obs function
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
        # Now can use the self.individual_observations internally without calling the _get_obs function again
        swarm_rewards = self._compute_swarm_rewards(individual_rewards)
        
        # Aggregate for Gym compliance
        reward = sum(swarm_rewards)  # Sum rewards for single float value
        terminated = any(terminateds)  # Aggregate to a single boolean
        truncated = any(truncateds)  # Aggregate to a single boolean
        
        # Global truncation if max steps reached
        if self.step_count >= self.config.max_steps:
            truncated = True
        
        # Add swarm metrics to info
        positions = np.array([obs['position'] for obs in self.individual_observations])
        info = {
            'agent_infos': infos,
            'swarm_reward': reward,
            'swarm_reward_components': {
                'safety_penalty': self._compute_safety_penalty(positions, 0),  # Example for agent 0
                'formation_bonus': self._compute_formation_bonus(positions, 0),
                'coordination_bonus': self._compute_coordination_bonus(self.individual_observations, 0)
            }
        }
        
        # Return aggregated values
        return swarm_obs, reward, terminated, truncated, info

    # Other methods are inherited from SwarmGymEnv and can be used as-is, e.g., close(), render(), get_swarm_positions(), etc.
    # If needed, override specific ones, but for now, they work with the new obs format.