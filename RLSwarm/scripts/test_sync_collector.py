import torch
import sys
import os
import numpy as np
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.rl_swarm_env import RLSwarmEnv
from envs.swarm_config import SwarmConfig

class DecentralizedPolicy(nn.Module):
    def __init__(self, n_agent_inputs, n_agent_outputs, n_agents, device, is_discrete=False):
        super().__init__()
        self.n_agents = n_agents
        self.device = device
        self.is_discrete = is_discrete
        self.policy_net = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralised=False,
            share_params=True,
            device=device
        )
        if self.is_discrete:
            # No additional params for discrete
            pass
        else:
            self.action_logstd = nn.Parameter(torch.zeros(n_agent_outputs)).to(device)
    
    def forward(self, agents):
        # agents is the sub-TensorDict
        depth_image = agents["depth_image"]
        position = agents["position"]
        rotation = agents["rotation"]
        velocity = agents["velocity"]
        target_distance = agents["target_distance"]
        front_obs_distance = agents["front_obs_distance"]
        # Concatenate inputs
        x = torch.cat([
            depth_image.flatten().to(self.device),
            position.flatten().to(self.device),
            rotation.flatten().to(self.device),
            velocity.flatten().to(self.device),
            target_distance.flatten().to(self.device),
            front_obs_distance.flatten().to(self.device)
        ], dim=-1)
        print(f"Policy x device: {x.device}")
        # Reshape x to (n_agents, per_agent_obs_dim)
        per_agent_obs_dim = x.shape[0] // self.n_agents
        x_reshaped = x.view(self.n_agents, per_agent_obs_dim)
        if self.is_discrete:
            logits = self.policy_net(x_reshaped)  # (n_agents, n_actions)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()  # (n_agents,) long
            log_prob = dist.log_prob(action)  # (n_agents,)
        else:
            action_mean = self.policy_net(x_reshaped)  # (n_agents, act_dim)
            action_logstd = self.action_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1)
        # Update the input tensordict in place
        agents = agents.set("agents", action)
        agents = agents.set("sample_log_prob", log_prob)
        return agents

def test_sync_collector():
    """Test RLSwarmEnv compatibility with SyncDataCollector."""
    print("Starting SyncDataCollector test for RLSwarmEnv...")
    
    # Load SwarmConfig from JSON
    json_path = os.path.join(os.path.dirname(__file__), 'reset_positions.json')
    config = SwarmConfig.from_json(json_path)
    config.max_steps = 10  # Short episodes for testing
    is_discrete = config.action_type == 'discrete'
    # Create RLSwarmEnv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RLSwarmEnv(config=config, device=device)
    
    print(f"Environment created with {config.n_agents} agents on {device}.")
    
    # # Check environment specs
    print("Checking environment specs...")
    try:
        check_env_specs(env)
        print("Environment specs are valid!")
    except Exception as e:
        print(f"Spec check failed: {e}")
        return
    
    # Obs dim for stacked agents
    obs_spec = env.observation_spec["agents"]
    obs_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())
    act_spec = env.action_spec["agents"]
    is_discrete = hasattr(act_spec, 'n')
    if is_discrete:
        print("Using discrete action space.")
        act_dim = act_spec.n
    else:
        act_dim = act_spec.shape[-1]
    
    # Per-agent obs dim
    per_agent_obs_dim = obs_dim // config.n_agents
    
    # Use DecentralizedPolicy as policy, wrapped in TensorDictModule
    policy_module = DecentralizedPolicy(
        n_agent_inputs=per_agent_obs_dim,
        n_agent_outputs=act_dim,
        n_agents=config.n_agents,
        device=device,
        is_discrete=is_discrete
    )
    policy = TensorDictModule(policy_module, in_keys=["agents"], out_keys=["sample_log_prob"])
    
    # For critic, use a custom forward to reshape input for MultiAgentMLP
    critic_net = MultiAgentMLP(
        n_agent_inputs=per_agent_obs_dim,
        n_agent_outputs=1,
        n_agents=config.n_agents,
        centralised=True,  # For MAPPO
        share_params=True,
        device=device
    )
    
    def critic_forward(agents):
        # agents is the sub-TensorDict
        depth_image = agents["depth_image"]
        position = agents["position"]
        rotation = agents["rotation"]
        velocity = agents["velocity"]
        target_distance = agents["target_distance"]
        front_obs_distance = agents["front_obs_distance"]
        # Concatenate inputs
        x = torch.cat([
            depth_image.flatten().to(device),
            position.flatten().to(device),
            rotation.flatten().to(device),
            velocity.flatten().to(device),
            target_distance.flatten().to(device),
            front_obs_distance.flatten().to(device)
        ], dim=-1)
        print(f"Critic x device: {x.device}")
        # Reshape x to (n_agents, per_agent_obs_dim)
        x_reshaped = x.view(config.n_agents, per_agent_obs_dim)
        value = critic_net(x_reshaped)  # (1,) for centralized
        # Update the input tensordict in place
        agents = agents.set("value", value)
        return agents

    critic = TensorDictModule(critic_forward, in_keys=["agents"], out_keys=["value"])
    
    # Create SyncDataCollector
    collector = SyncDataCollector(
        create_env_fn=lambda: env,  # Provide a callable that returns the env
        policy=policy,
        frames_per_batch=50,
        total_frames=100,  # Collect 100 frames for testing
        device=device,
        trust_policy=True  # Trust the policy to handle env outputs
    )
    
    print("SyncDataCollector created. Starting data collection...")
    
    # Collect data
    for i, data in enumerate(collector):
        print(f"Batch {i}: Collected {data.shape[0]} frames")
        print(f"Sample keys: {list(data.keys())}")
        print(f"Sample agents obs shape: {data['agents']['depth_image'].shape}")
        print(f"Sample shared obs shape: {data['shared_observation']['inter_agent_distances'].shape}")
        if i >= 1:  # Collect 2 batches for testing
            break
    
    print("Data collection completed successfully!")
    print("RLSwarmEnv is compatible with SyncDataCollector.")

if __name__ == "__main__":
    test_sync_collector()