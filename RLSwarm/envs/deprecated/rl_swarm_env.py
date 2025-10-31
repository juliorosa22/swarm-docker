import torch
from torchrl.envs import GymWrapper
import numpy as np
from .swarm import SwarmGymEnv
from gymnasium import spaces
from ..swarm_config import SwarmConfig
from tensordict import TensorDict
import time

class RLSwarmEnv(SwarmGymEnv):
    """
    TorchRL wrapper for the SwarmGymEnv.
    Handles multi-agent observations and actions for MARL.
    """
    def __init__(self, config: SwarmConfig, device: torch.device = None):
        # Create the underlying Gym env
        super().__init__(config)
        print(f"RLSwarmGymEnv initialized with {self.config.n_agents} agents.")
        self.config = config
        self.n_agents = config.n_agents
        
        # Override spaces for stacked/multi-agent format
        self._override_spaces()
 
    
    def _override_spaces(self):
        base_action_space = self.uav_envs[0].action_space
        obs_space = self.uav_envs[0].observation_space

        # Define stacked observation space for agents (individual features)
        agent_obs_space = spaces.Dict({
            'depth_image': spaces.Box(
                low=obs_space['depth_image'].low.min(),
                high=obs_space['depth_image'].high.max(),
                shape=(self.n_agents,) + obs_space['depth_image'].shape,
                dtype=np.float32
            ),
            'position': spaces.Box(
                low=obs_space['position'].low.min(),
                high=obs_space['position'].high.max(),
                shape=(self.n_agents, 3),
                dtype=np.float32
            ),
            'rotation': spaces.Box(
                low=obs_space['rotation'].low.min(),
                high=obs_space['rotation'].high.max(),
                shape=(self.n_agents, 3),
                dtype=np.float32
            ),
            'velocity': spaces.Box(
                low=obs_space['velocity'].low.min(),
                high=obs_space['velocity'].high.max(),
                shape=(self.n_agents, 3),
                dtype=np.float32
            ),
            'target_distance': spaces.Box(
                low=obs_space['target_distance'].low.min(),
                high=obs_space['target_distance'].high.max(),
                shape=(self.n_agents, 1),
                dtype=np.float32
            ),
            'front_obs_distance': spaces.Box(
                low=obs_space['front_obs_distance'].low.min(),
                high=obs_space['front_obs_distance'].high.max(),
                shape=(self.n_agents, 1),
                dtype=np.float32
            ),
        })
        
        # Define shared observation space (global features for critic)
        shared_obs_space = spaces.Dict({
            'inter_agent_distances': spaces.Box(
                low=0.0, high=self.config.max_formation_distance,
                shape=(self.n_agents, self.n_agents), dtype=np.float32
            ),
            'target_distances': spaces.Box(
                low=0.0, high=np.inf, shape=(self.n_agents,), dtype=np.float32
            ),
            'velocities': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_agents, 3), dtype=np.float32
            ),
            'obstacle_distances': spaces.Box(
                low=0.0, high=np.inf, shape=(self.n_agents,), dtype=np.float32
            )
        })
        
        # Swarm action space: tuple of individual actions
        self.action_space = spaces.Tuple([base_action_space] * self.n_agents)
        
        # Set observation space as Dict for GymWrapper
        self.observation_space = spaces.Dict({
            'agents': agent_obs_space,
            'shared_observation': shared_obs_space
        })
    

    def _get_obs_td(self):
        """Return the swarm observation as a dict matching the overridden space."""
        try:
            shared_obs = self._get_obs()  # Assumes SwarmGymEnv has this method
            
            # Get individual stacked obs
            individual_obs = {
                'depth_image': np.array([obs['depth_image'] for obs in self.individual_observations], dtype=np.float32),
                'position': np.array([obs['position'] for obs in self.individual_observations], dtype=np.float32),
                'rotation': np.array([obs['rotation'] for obs in self.individual_observations], dtype=np.float32),
                'velocity': np.array([obs['velocity'] for obs in self.individual_observations], dtype=np.float32),
                'target_distance': np.array([obs['target_distance'] for obs in self.individual_observations], dtype=np.float32).reshape((self.n_agents, 1)),
                'front_obs_distance': np.array([obs['front_obs_distance'] for obs in self.individual_observations], dtype=np.float32).reshape((self.n_agents, 1))
            }
            # Get global obs from swarm
            return {'agents': individual_obs, 'shared_observation': shared_obs}

        except Exception as e:
            print(f"Error in _get_obs: {e}")
            # Return defaults
            return {
                'agents': {
                    'depth_image': np.zeros((self.n_agents,) + self.uav_envs[0].observation_space['depth_image'].shape, dtype=np.float32),
                    'position': np.zeros((self.n_agents, 3), dtype=np.float32),
                    'rotation': np.zeros((self.n_agents, 3), dtype=np.float32),
                    'velocity': np.zeros((self.n_agents, 3), dtype=np.float32),
                    'target_distance': np.zeros((self.n_agents, 1), dtype=np.float32),
                    'front_obs_distance': np.zeros((self.n_agents, 1), dtype=np.float32)
                },
                'shared_observation': {
                    'inter_agent_distances': np.zeros((self.n_agents, self.n_agents), dtype=np.float32),
                    'target_distances': np.zeros(self.n_agents, dtype=np.float32),
                    'velocities': np.zeros((self.n_agents, 3), dtype=np.float32),
                    'obstacle_distances': np.zeros(self.n_agents, dtype=np.float32)
                }
            }
    
    def reset(self, seed=None, options=None):
        """Reset and return TensorDict for TorchRL."""
        shared_obs, info = super().reset(seed=seed, options=options)
        individual_obs = {
            'depth_image': np.array([obs['depth_image'] for obs in self.individual_observations], dtype=np.float32),
            'position': np.array([obs['position'] for obs in self.individual_observations], dtype=np.float32),
            'rotation': np.array([obs['rotation'] for obs in self.individual_observations], dtype=np.float32),
            'velocity': np.array([obs['velocity'] for obs in self.individual_observations], dtype=np.float32),
            'target_distance': np.array([obs['target_distance'] for obs in self.individual_observations], dtype=np.float32).reshape((self.n_agents, 1)),
            'front_obs_distance': np.array([obs['front_obs_distance'] for obs in self.individual_observations], dtype=np.float32).reshape((self.n_agents, 1))
        }
        out_td = dict({
            "agents": individual_obs,
            "shared_observation": shared_obs
        })

        return out_td, info

    def step(self, actions):
        """Step and return TensorDict for TorchRL."""
        
        # Collect futures from act_no_join (sequential to avoid IOLoop conflicts)
        shared_obs, swarm_rewards, terminateds, truncateds, info = super().step(actions)
        # Aggregate for Gym compliance
        reward = sum(swarm_rewards)  # Sum rewards for single float value
        terminated = any(terminateds)  # Aggregate to a single boolean
        truncated = any(truncateds)  # Aggregate to a single boolean
        
        # Global truncation if max steps reached
        if self.step_count >= self.config.max_steps:
            truncated = True
        
        individual_obs = {
            'depth_image': np.array([obs['depth_image'] for obs in self.individual_observations], dtype=np.float32),
            'position': np.array([obs['position'] for obs in self.individual_observations], dtype=np.float32),
            'rotation': np.array([obs['rotation'] for obs in self.individual_observations], dtype=np.float32),
            'velocity': np.array([obs['velocity'] for obs in self.individual_observations], dtype=np.float32),
            'target_distance': np.array([obs['target_distance'] for obs in self.individual_observations], dtype=np.float32).reshape((self.n_agents, 1)),
            'front_obs_distance': np.array([obs['front_obs_distance'] for obs in self.individual_observations], dtype=np.float32).reshape((self.n_agents, 1))
        }
        
        # Convert to TensorDict
        obs = dict({
            "agents": individual_obs,
            "shared_observation": shared_obs,
        })
        return obs, reward, terminated, truncated, info
