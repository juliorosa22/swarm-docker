import torch
import sys
import os
import numpy as np
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs
from torchrl.modules import TensorDictModule

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from envs.rl_swarm_env import RLSwarmEnv
from envs.swarm_config import SwarmConfig
from algorithms.mappo import MAPPO  # Import MAPPO for integration test
from policies.centralized_critic import CentralizedCritic
from policies.decentralized_policy import DecentralizedPolicy

def verify_rl_swarm_env():
    """Verify RLSwarmEnv is ready for training with MAPPO."""
    print("Starting verification of RLSwarmEnv...")
    
    # Load SwarmConfig from JSON
    json_path = os.path.join(os.path.dirname(__file__), 'reset_positions.json')
    config = SwarmConfig.from_json(json_path)
    config.max_steps = 10  # Short episodes for testing
    
    # Create RLSwarmEnv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RLSwarmEnv(config=config, device=device)
    
    # Check environment specs (disabled for testing)
    # print("Checking environment specs...")
    # try:
    #     check_env_specs(env)
    #     print("Environment specs are valid!")
    # except Exception as e:
    #     print(f"Spec check failed: {e}")
    #     return
    
    print(f"Environment created with {config.n_agents} agents on {device}.")
    
    # Test reset
    print("\nTesting reset...")
    reset_tensordict = env.reset()
    print(f"Reset successful. Agent 0 observation keys: {list(reset_tensordict['agent0']['observation'].keys())}")
    print(f"Shared observation shape: {reset_tensordict['shared_observation']['inter_agent_distances'].shape}")
    
    # Test a few steps with random actions
    print("\nTesting steps with random actions...")
    for step in range(5):
        # Generate random actions for each agent
        actions = {}
        for i in range(config.n_agents):
            action_space = env.env.action_space[i]  # Access underlying Gym space
            if hasattr(action_space, 'n'):  # Discrete
                actions[f"agent{i}"] = {"action": torch.randint(0, action_space.n, ())}
            else:  # Continuous (assume Box)
                actions[f"agent{i}"] = {"action": torch.randn(action_space.shape)}
        
        tensordict_in = TensorDict(actions, batch_size=())
        
        # Step the environment
        step_tensordict = env.step(tensordict_in)
        
        # Check outputs
        print(f"Step {step}: Agent 0 reward: {step_tensordict['agent0']['reward'].item():.3f}")
        print(f"Step {step}: Agent 0 done: {step_tensordict['agent0']['done'].item()}")
        
        if step_tensordict['agent0']['done'].item():
            print("Episode ended early.")
            break
    
    print("\nBasic environment test passed!")
    
    # Test integration with MAPPO (mini-training)
    print("\nTesting integration with MAPPO...")
    
    # Use DecentralizedPolicy as policy
    policy = DecentralizedPolicy(
        observation_spec=env.observation_spec["agent0"],
        action_spec=env.action_spec["agent0"],
        device=device
    )
    
    # Use CentralizedCritic wrapped in TensorDictModule for critic
    critic_module = CentralizedCritic(
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        n_agents=config.n_agents,
        device=device
    )
    critic = TensorDictModule(critic_module, in_keys=[], out_keys=["value"])  # Passes whole tensordict
    
    # Initialize MAPPO
    mappo = MAPPO(
        env=env,
        policy=policy,
        critic=critic,
        device=device,
        n_agents=config.n_agents,
        frames_per_batch=50,  # Small for testing
        batch_size=16
    )
    
    # Run a mini-training loop
    print("Running mini-training (50 frames)...")
    mappo.train(total_frames=50)
    print("Mini-training completed successfully!")
    
    print("\nVerification complete: RLSwarmEnv is ready for training with MAPPO!")

if __name__ == "__main__":
    verify_rl_swarm_env()