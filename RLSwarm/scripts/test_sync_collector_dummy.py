import torch
import sys
import os
import numpy as np
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import check_env_specs
from torchrl.envs import GymWrapper
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.full_swarm import FullSwarmGymEnv  # Use the new full swarm env
from envs.swarm_config import SwarmConfig

class DummyPolicy(nn.Module):
    """A simple dummy policy that samples random actions from the action spec, handling discrete or continuous."""
    def __init__(self, action_spec, device):
        super().__init__()
        self.action_spec = action_spec
        self.device = device
        # Check if discrete (assume Tuple of Discrete or single Discrete)
        self.is_discrete = isinstance(action_spec, tuple) and hasattr(action_spec[0], 'n') or hasattr(action_spec, 'n')
        if self.is_discrete:
            self.n_actions = action_spec[0].n if isinstance(action_spec, tuple) else action_spec.n
    
    def forward(self, depthimg, position, rotation, velocity, target_distance, front_obs_distance, tensordict=None):
        # The policy receives the full tensordict. Observations are in tensordict['obs'].
        # Since this is a dummy policy, we don't need to read the observations.

        # Sample random action from the spec (Tuple of actions)
        action = self.action_spec.sample()
        # Convert to tensor and move to CPU to avoid numpy conversion issues with CUDA (TensorDict will move to CUDA automatically)
        action_tensor = tuple(torch.tensor(a, device='cpu').detach() for a in action)
        
        if self.is_discrete:
            # For discrete, log_prob is log(1/n_actions) for uniform sampling
            log_prob = torch.log(torch.tensor(1.0 / self.n_actions, device='cpu')).expand(len(action))
        else:
            # For continuous, dummy log_prob (zeros)
            log_prob = torch.zeros(len(action), device='cpu')
        
        # Create the action inside the dict passed (in-place assignment)
        tensordict = TensorDict({
            "depth_image": depthimg,
            "position": position,
            "rotation": rotation,
            "velocity": velocity,
            "target_distance": target_distance,
            "front_obs_distance": front_obs_distance
        }, batch_size=depthimg.shape[:-3], device=self.device)
        tensordict["action"] = action_tensor
        tensordict["sample_log_prob"] = log_prob
        return tensordict

def dummy_critic_forward(tensordict):
    """A simple dummy critic that returns a fixed value."""
    value = torch.tensor(0.0, device=tensordict.device)  # Fixed value for simplicity
    tensordict.set_("value", value)
    return tensordict

def test_sync_collector_dummy():
    """Test FullSwarmGymEnv compatibility with SyncDataCollector using a dummy policy."""
    print("Starting SyncDataCollector test for FullSwarmGymEnv with dummy policy...")
    
    # Load SwarmConfig from JSON
    json_path = os.path.join(os.path.dirname(__file__), 'reset_positions.json')
    config = SwarmConfig.from_json(json_path)
    config.max_steps = 10  # Short episodes for testing
    
    # Create FullSwarmGymEnv and wrap with GymWrapper for TorchRL
    full_swarm_env = FullSwarmGymEnv(config=config)
    env = GymWrapper(full_swarm_env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Environment created with {config.n_agents} agents on {env.device}.")
    print(f"Action space type: {type(env.action_spec)}")
    if hasattr(env.action_spec, 'n'):
        print(f"Discrete actions with {env.action_spec.n} options.")
    elif isinstance(env.action_spec, tuple) and len(env.action_spec) > 0 and hasattr(env.action_spec[0], 'n'):
        print(f"Tuple of discrete actions with {env.action_spec[0].n} options per agent.")
    else:
        print("Continuous actions.")
    
    # # Check environment specs
    # print("Checking environment specs...")
    # try:
    #     check_env_specs(env)
    #     print("Environment specs are valid!")
    # except Exception as e:
    #     print(f"Spec check failed: {e}")
    #     return
    
    # Use DummyPolicy as policy, wrapped in TensorDictModule.
    # Pass the full tensordict by setting in_keys to None.
    policy = TensorDictModule(
        DummyPolicy(env.action_spec, env.device), 
        in_keys=["depth_image", "position","rotation","velocity","target_distance","front_obs_distance"],  # Specify in_keys to match observation keys
        out_keys=["action", "sample_log_prob"]
    )

    # Use dummy critic
    # critic = TensorDictModule(dummy_critic_forward, in_keys=None, out_keys=["value"])
    
    # Create SyncDataCollector
    collector = SyncDataCollector(
        create_env_fn=lambda: env,  # Provide a callable that returns the wrapped env
        policy=policy,
        frames_per_batch=50,
        total_frames=100,  # Collect 100 frames for testing
        device=env.device,
        trust_policy=True  # Trust the policy to handle env outputs
    )
    
    print("SyncDataCollector created. Starting data collection...")
    
    # Collect data
    for i, data in enumerate(collector):
        print(f"Batch {i}: Collected {data.shape[0]} frames")
        print(f"Sample keys: {list(data.keys())}")
        # Access observations inside the 'obs' nested tensordict
        print(f"Sample depth_image shape: {data['depth_image'].shape}")
        print(f"Sample position shape: {data['position'].shape}")
        if i >= 1:  # Collect 2 batches for testing
            break
    
    print("Data collection completed successfully!")
    print("FullSwarmGymEnv is compatible with SyncDataCollector using dummy policy.")

if __name__ == "__main__":
    test_sync_collector_dummy()