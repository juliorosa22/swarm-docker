import torch
import sys
import os
import gc
from tensordict import TensorDict

# Add parent directory to path to import SwarmEnvComposer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.swarm_deprecated import SwarmEnvComposer

def generate_random_discrete_actions(n_agents: int, batch_size: int, device: torch.device) -> TensorDict:
    """Generate random discrete actions for each agent."""
    actions = TensorDict({}, batch_size=[])
    
    for i in range(n_agents):
        # Generate random discrete action (0-9) for each agent
        agent_action = torch.randint(
            low=0,
            high=10,  # 10 discrete actions
            size=(1,),
            device=device
        )
        
        actions[f"agent{i}"] = {
            "action": agent_action
        }
    
    return actions

def main():
    # Set device and memory options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Using device: {device}")
    
    # Define start and end positions for 5 agents
    start_positions = [
        (-3.5, 3, -10),
        (-2, 1.5, -10),
        (0, 0, -10),
        (2, 1.5, -10),
        (3.5, 3, -10)
    ]
    
    end_positions = [
        (-3.5, 3-210, -10),
        (-2, 1.5-210, -10),
        (0, -210, -10),
        (2, 1.5-210, -10),
        (3.5, 3-210, -10)
    ]
    
    # Initialize environment
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
        observation_img_size=(64, 64)
    )
    
    print("\nStarting direct environment test with random actions...")
    print(f"Number of agents: {env.n_agents}")
    print(f"Max steps: {env.max_steps}")
    
    # Run episodes
    n_episodes = 3
    max_steps_per_episode = 100
    
    try:
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            
            # Reset environment
            obs = env._reset()
            episode_reward = 0
            
            for step in range(max_steps_per_episode):
                # Generate random actions
                actions = generate_random_discrete_actions(env.n_agents, [], device)
                
                # Take step in environment
                next_obs = env._step(actions)
                
                # Get rewards directly from swarm environment
                rewards = []
                for i in range(env.n_agents):
                    agent_reward = next_obs[f"agent{i}"]["agent_reward"]
                    rewards.append(agent_reward)
                
                # Calculate total reward for this step
                step_reward = torch.stack(rewards).sum().item()
                episode_reward += step_reward
                
                # Print step information
                if step % 10 == 0:  # Print every 10 steps
                    print(f"\nStep {step + 1}:")
                    print(f"  Total swarm reward: {step_reward:.2f}")
                    
                    # Print individual agent rewards and positions
                    for i in range(env.n_agents):
                        pos = next_obs[f"agent{i}"]["observation"]["base_obs"]["position"]
                        agent_reward = next_obs[f"agent{i}"]["agent_reward"].item()
                        print(f"  Agent {i}:")
                        print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                        print(f"    Reward: {agent_reward:.2f}")
                
                # Check if episode is done
                if next_obs["global_done"].item():
                    print(f"\nEpisode finished after {step + 1} steps")
                    break
            
            print(f"Episode {episode + 1} total reward: {episode_reward:.2f}")
            
            # Clear some memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        env.close()
        print("\nTest completed!")

if __name__ == "__main__":
    main()