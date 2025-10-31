import torch
import sys
import os
import gc
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv, Transform

# Add parent directory to path to import SwarmEnvComposer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.swarm_deprecated import SwarmEnvComposer

class RandomPolicy:
    """Simple random policy for testing with SwarmEnv"""
    def __init__(self, action_spec, device):
        self.action_spec = action_spec
        self.device = device
        
        # Pre-process and cache action specs on correct device
        self.processed_specs = {}
        for key, spec in self.action_spec.items():
            action_spec = spec["action"]
            if hasattr(action_spec, "low") and hasattr(action_spec, "high"):
                # Cache processed bounds
                self.processed_specs[key] = {
                    "low": action_spec.low.clone().to(device),
                    "high": action_spec.high.clone().to(device),
                    "shape": action_spec.shape,
                    "dtype": action_spec.dtype
                }

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        batch_size = tensordict.batch_size
        actions = TensorDict({}, batch_size=batch_size)
        
        for key in self.action_spec.keys():
            spec = self.processed_specs[key]
            
            if spec["dtype"] == torch.int64:  # Discrete actions
                action = torch.randint(
                    low=0,
                    high=10,
                    size=(*batch_size, 1),
                    device=self.device
                )
            else:  # Continuous actions
                # Generate random actions between bounds
                action = torch.rand(
                    (*batch_size, *spec["shape"]),
                    device=self.device
                )
                action = action * (spec["high"] - spec["low"]) + spec["low"]
            
            actions[key] = {"action": action}
        
        return actions

class SwarmRewardSum(Transform):
    def __init__(self, n_agents: int = 5):
        super().__init__()
        self.n_agents = n_agents

    def _call(self, tensordict):
        # Sum agent rewards
        agent_rewards = torch.stack([
            tensordict[f"agent{i}"]["agent_reward"].squeeze(-1)
            for i in range(self.n_agents)
        ])
        total_reward = agent_rewards.sum(dim=0)
        
        # Ensure reward has correct shape [batch_size]
        if len(total_reward.shape) == 0:
            total_reward = total_reward.unsqueeze(0)
        
        tensordict.set("reward", total_reward)
        return tensordict

    def _reset(self, tensordict, next_tensordict=None, **kwargs):
        # Initialize reward in tensordict
        if tensordict is not None:
            tensordict.set("reward", torch.zeros(1, device=tensordict.device))
        
        # Initialize reward in next_tensordict if it exists
        if next_tensordict is not None:
            next_tensordict.set("reward", torch.zeros(1, device=next_tensordict.device))
            return next_tensordict
        
        return tensordict

def main():
    # Set device and memory options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Using device: {device}")
    
    # Initialize environment with TransformedEnv wrapper
    base_env = SwarmEnvComposer(
        n_agents=5,
        action_type="continuous",
        max_steps=100,
        device=device,
        close_distance_threshold=2.0,
        far_distance_threshold=8.0,
        max_swarm_distance=15.0,
        observation_img_size=(64, 64)
    )
    
    # Wrap environment with custom reward transform
    env = TransformedEnv(
        base_env,
        SwarmRewardSum(n_agents=base_env.n_agents)  # Pass n_agents explicitly
    )
    
    # Create random policy
    policy = RandomPolicy(env.action_spec, device)
    
    # Create data collector with minimal settings for testing
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=10,  # Very small batch for testing
        total_frames=30,      # Just 3 batches
        split_trajs=True,
        device=device,
        trust_policy=True
    )
    
    try:
        print("\nStarting collector test...")
        for i, batch in enumerate(collector):
            print(f"\nBatch {i+1}:")
            print(f"Batch shape: {batch.batch_size}")
            print(f"Reward shape: {batch['reward'].shape}")
            
            # Print individual agent rewards
            for j in range(env.n_agents):
                agent_reward = batch[f"agent{j}"]["agent_reward"]
                print(f"Agent {j} reward shape: {agent_reward.shape}")
            
            # Calculate statistics for this batch
            total_reward = batch["reward"].mean().item()  # Now using transformed reward
            
            # Print per-agent rewards
            for agent_idx in range(env.n_agents):
                agent_rewards = batch[f"agent{agent_idx}"]["agent_reward"]
                mean_reward = agent_rewards.mean().item()
                print(f"  Agent {agent_idx} mean reward: {mean_reward:.2f}")
            
            print(f"  Batch total reward: {total_reward:.2f}")
            print(f"  Batch size: {batch.batch_size}")
            
            # Print some position information
            for agent_idx in range(env.n_agents):
                positions = batch[f"agent{agent_idx}"]["observation"]["base_obs"]["position"]
                mean_pos = positions.mean(dim=0)
                print(f"  Agent {agent_idx} mean position: ({mean_pos[0]:.2f}, {mean_pos[1]:.2f}, {mean_pos[2]:.2f})")
            
            # Check done flags
            done_counts = torch.stack([
                batch[f"agent{i}"]["agent_done"].sum() 
                for i in range(env.n_agents)
            ])
            print(f"  Episodes completed: {done_counts.max().item()}")
            
            # Clear some memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        collector.shutdown()
        env.close()
        print("\nTest completed! here")

if __name__ == "__main__":
    main()