import os
import sys
import torch
from tensordict import TensorDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.swarm_config import SwarmConfig
from envs.swarm_torch import SwarmTorchEnv
from policies.decentralized_policy import DecentralizedPolicy
from policies.centralized_critic import CentralizedCritic
from torchrl.collectors import SyncDataCollector

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = "continuous_mappo_v1"
    json_path = os.path.join(os.path.dirname(__file__), 'reset_positions.json')
    swarm_config = SwarmConfig.from_json(json_path)
    swarm_config.action_type = "continuous"
    n_agents = swarm_config.n_agents
    swarm_config.max_steps = 1000 # Longer episodes for meaningful collection
    
    env = SwarmTorchEnv(config=swarm_config, device=device,training_file=f"{base_name}_env_log.json")
    
    single_agent_obs_spec = env.observation_spec["agents"]
    
    shared_observation_spec = env.observation_spec["shared_observation"]
    agent_group_action_spec = env.action_spec
    
    policy = DecentralizedPolicy(
        observation_spec=single_agent_obs_spec,
        action_spec=agent_group_action_spec,
        hidden_dim=64,
        device=device
    )
    
    frames_per_batch = 8
    total_frames = 64
    collected_frames = 0
    # Create data collector with minimal settings for testing
    collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=device,
            reset_at_each_iter=True,
        )
    for i, tensordict_data in enumerate(collector):
            # Stop the loop once the total frame count is reached
            if collected_frames >= total_frames:
                break

if __name__ == "__main__":
    main()