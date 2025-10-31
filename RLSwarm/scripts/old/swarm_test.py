import torch
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.envs import TransformedEnv, StepCounter
from torchrl.envs.transforms import StepCounter
from typing import Dict
import sys
import os
import gc

# Add memory management imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Add parent directory to path to import SwarmEnvComposer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.swarm_deprecated import SwarmEnvComposer

class RandomPolicy:
    """Simple random policy for testing"""
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
                # Generate random actions directly on device
                action = torch.rand(
                    (*batch_size, *spec["shape"]),
                    device=self.device
                )
                # Use cached bounds that are already on correct device
                action = action * (spec["high"] - spec["low"]) + spec["low"]
            
            actions[key] = {"action": action}
        
        return actions


def main():
    # Set device and memory options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # Print GPU memory info
        print(f"GPU Memory before environment creation:")
        print(torch.cuda.memory_summary(device=device))
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Using device: {device}")
    
    # Reduce batch size and buffer size
    BATCH_SIZE = 32  # Reduced from 64
    FRAMES_PER_BATCH = 500  # Reduced from 1000
    BUFFER_SIZE = 5000  # Reduced from 10000
    
    # Define start and end positions for 5 agents
    start_positions = [(-3.5,3,-10),(-2,1.5,-10),(0,0,-10),(2,1.5,-10),(3.5,3,-10)]
    
    end_positions = [(-3.5,3-210,-10),(-2,1.5-210,-10),(0,-210,-10),(2,1.5-210,-10),(3.5,3-210,-10)]
    
    # Initialize environment with specific positions
    env = SwarmEnvComposer(
        n_agents=5,
        action_type="discrete",
        max_steps=100,
        start_positions=start_positions,
        end_positions=end_positions,
        device=device,
        close_distance_threshold=2.0,
        far_distance_threshold=8.0,
        max_swarm_distance=15.0,
        observation_img_size=(64, 64)  # Reduced image size
    )
    
    # Add step counter to environment
    env = TransformedEnv(
        env,
        StepCounter(max_steps=100)
    )
    
    # Create random policy
    policy = RandomPolicy(env.action_spec, device)
    
    # Initialize data collector with reduced batch size
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=-1,
        split_trajs=True,
        device=device,
        trust_policy=True  # Removed reset_at_start parameter
    )
    
    # Create replay buffer with reduced size
    replay_buffer = TensorDictReplayBuffer(
        storage=None,
        batch_size=BATCH_SIZE,
        max_size=BUFFER_SIZE,
        device=device
    )
    
    print("\nStarting environment test with random policy...")
    print(f"Initial formation radius: 5.0m")
    print(f"Target formation center: (20, 20, -2)")
    print(f"Number of agents: 5")
    
    # Collect trajectories with memory management
    try:
        for i, batch in enumerate(collector):
            if i >= 5:  # Collect 5 batches
                break
            
            # Move batch to CPU before adding to buffer to save GPU memory
            batch = batch.cpu()
            replay_buffer.extend(batch)
            
            # Print batch statistics
            rewards = batch["agent0"]["reward"]
            mean_reward = rewards.mean().item()
            done_masks = batch["done"]
            num_episodes = done_masks.sum().item()
            
            print(f"\nBatch {i}:")
            print(f"  Mean reward: {mean_reward:.2f}")
            print(f"  Episodes completed: {num_episodes}")
            print(f"  Buffer size: {len(replay_buffer)}")
            
            # Check swarm cohesion (on CPU to save memory)
            positions = torch.stack([
                batch[f"agent{i}"]["observation"]["position"] 
                for i in range(env.n_agents)
            ], dim=1)
            
            with torch.no_grad():  # Reduce memory usage during computation
                pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
                distances = torch.norm(pos_diff, dim=-1)
                avg_distances = distances[distances != 0].mean(dim=-1)
            
            print(f"  Average swarm distance: {avg_distances.mean().item():.2f}m")
            print("  Formation status: ", end="")
            if avg_distances.mean().item() > env.max_swarm_distance:
                print("WARNING: Swarm dispersed!")
            else:
                print("OK")
            
            # Clear some memory after each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            print("---")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Clean up
        env.close()
        collector.shutdown()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        print("\nTest completed!")

if __name__ == "__main__":
    main()