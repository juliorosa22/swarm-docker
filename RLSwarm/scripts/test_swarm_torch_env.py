import sys
import os
import random
import torch
from tensordict import TensorDict

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Assuming you have renamed 'swarm-torch.py' to 'swarm_torch.py' for valid import
from envs.swarm_torch import SwarmTorchEnv
from envs.swarm_config import SwarmConfig

def main():
    # Load SwarmConfig from JSON
    json_path = os.path.join(os.path.dirname(__file__), 'reset_positions.json')
    config = SwarmConfig.from_json(json_path)
    
    # Override config for testing
    config.max_steps = 30  # 30 steps per episode
    config.n_agents = 5    # Test with 5 agents

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the native TorchRL environment
    env = SwarmTorchEnv(
        config=config, 
        frame_skip_duration=0.5, 
        use_lidar=False, 
        device=device
    )
    
    # Number of episodes to test
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset the environment - returns a TensorDict
        td = env.reset()
        print(f"Initial observation keys: {list(td.get(('agents')).keys())}")
        
        # Print depth_image after reset
        depth_img = td["agents", "depth_image"]
        print(f"Initial depth_image shape: {depth_img.shape}, mean: {depth_img.mean().item():.3f}")
        
        total_reward = 0.0
        step_count = 0
        
        # The 'done' key is at the root of the tensordict
        done = td.get("done", torch.tensor([False], device=device))

        while not done.any() and step_count < 30:
            # Sample random actions and place them in the input TensorDict
            # The env.action_spec provides the shape and device
            actions = env.action_spec.rand()
            td.set("action", actions)
            
            # Step the environment using the full TensorDict
            td = env.step(td)
            
            # In EnvBase, step results are nested under "next"
            reward = td.get(("next", "reward"))
            done = td.get(("next", "done"))
            
            # Print depth_image after step
            depth_img = td["next", "observation", "agents", "depth_image"]
           # print(depth_img)
            print(f"Step {step_count+1} depth_image shape: {depth_img.shape}, mean: {depth_img.mean().item():.3f}")
            
            total_reward += reward.item()
            step_count += 1
            
            print(f"Step {step_count}: Reward={reward.item():.3f}, Done={done.item()}")
            
            # Check if episode is done
            if done.any():
                print(f"Episode {episode + 1} ended at step {step_count} with total reward {total_reward:.3f}")
                break
        
        if not done.any():
            print(f"Episode {episode + 1} reached max steps with total reward {total_reward:.3f}")
    
    # Close the environment (this will save the collision log)
    env.close()
    print("\nTesting completed. Check 'training_log_torch.json' for collision data.")

if __name__ == "__main__":
    main()
