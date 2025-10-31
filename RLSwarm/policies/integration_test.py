# test_collector_integration.py
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.modules import Actor
from tensordict.nn import TensorDictModule

import sys
import os

# Adjust path to import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.swarm_torch import SwarmTorchEnv
from envs.swarm_config import SwarmConfig


def create_simple_policy(env):
    """
    Create a simple random policy compatible with TorchRL collectors.
    The policy must accept a TensorDict and return a TensorDict with an "action" key.
    """
    action_spec = env.action_spec

    class RandomPolicyModule(torch.nn.Module):
        def __init__(self, action_spec):
            super().__init__()
            self.action_spec = action_spec

        def forward(self, tensordict):
            # The collector expects the policy to populate the "action" key
            batch_size = tensordict.batch_size
            action = self.action_spec.rand(batch_size)
            return tensordict.set("action", action)

    # Wrap the module in a TensorDictModule for compatibility
    policy_module = RandomPolicyModule(action_spec)
    return TensorDictModule(
        module=policy_module,
        in_keys=["agents"],  # This key is present in the observation spec
        out_keys=["action"],
    )

def test_collector_integration():
    """Test SyncDataCollector integration with SwarmTorchEnv."""
    print("=== Testing SyncDataCollector Integration with SwarmTorchEnv ===")
    
    try:
        # --- Setup Environment ---
        # Correct the path to go up one directory and then into 'scripts'
        json_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reset_positions.json')
        config = SwarmConfig.from_json(json_path)
        config.max_steps = 20
        config.n_agents = 5
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create the native TorchRL environment
        env = SwarmTorchEnv(config=config, device=device)
        
        # --- Create Policy ---
        policy = create_simple_policy(env)
        
        # --- Create Collector ---
        collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=10,
            max_frames_per_traj=50,
            total_frames=10,
            device=device
        )
        
        # --- Collect Data ---
        print("Starting data collection...")
        collected_frames = 0
        for i, data in enumerate(collector):
            collected_frames += data.numel()
            print(f"Batch {i}: Collected {data.numel()} frames. Total frames: {collected_frames}")
            # The data contains keys like 'observation', 'action', 'next', 'reward', 'done'
            assert "action" in data.keys()
            assert ("next", "reward") in data.keys(True)
            if i >= 1:  # Just test the first couple of batches
                break
        
        collector.shutdown()
        print("\n✓ SyncDataCollector integration with SwarmTorchEnv works!")
        return True
        
    except Exception as e:
        print(f"\n✗ SyncDataCollector integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_collector_integration()