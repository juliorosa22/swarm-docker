import os
import sys
import torch
from tensordict import TensorDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.swarm_config import SwarmConfig
from torchrl.data import Bounded, Composite, Unbounded, Categorical
from envs.swarm_torch import SwarmTorchEnv
from policies.decentralized_policy import DecentralizedPolicy
from policies.centralized_critic import CentralizedCritic
from algorithms.mappo import MAPPO
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule
def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = "continuous_mappo_v1"
    json_path = os.path.join(os.path.dirname(__file__), 'reset_positions.json')
    swarm_config = SwarmConfig.from_json(json_path)
    n_agents = swarm_config.n_agents
    swarm_config.max_steps = 1000 # Longer episodes for meaningful collection
    # Initialize environment directly (no GymWrapper needed)
    base_env = SwarmTorchEnv(config=swarm_config, device=device,training_file=f"{base_name}_env_log.json")
    env = TransformedEnv(base_env, transform=RewardSum() )
    #check_env_specs(base_env)
    
    # --- Correctly derive single-agent specs from the environment ---
    single_agent_obs_spec = env.observation_spec["agents"]
    # The critic observes the full shared state
    shared_observation_spec = env.observation_spec["shared_observation"]
    single_agent_action_spec = env.action_spec[0]
    
    # Initialize policy (shared weights across agents)
    # The policy's internal networks use the single-agent specs,
    # but the final ProbabilisticActor needs the full multi-agent spec to sample correctly.
    #create the policy and adjust the action spec accordingly
    policy = DecentralizedPolicy(
        observation_spec=single_agent_obs_spec,
        action_spec=env.action_spec,
        hidden_dim=64,
        device=device
    )
    # Mounts the critic network that uses attention over shared observations
    critic_net = CentralizedCritic(
        observation_spec=shared_observation_spec,
        action_spec=single_agent_action_spec,
        hidden_dim=64,
        n_attention_heads=4,
        device=device
    )


    # print("="*60)
    # print("--- Policy Network Architecture ---")
    # print(policy)
    # policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    # print(f"Total trainable policy parameters: {policy_params:,}")
    # print("="*60)

    # print("\n" + "="*60)
    # print("--- Critic Network Architecture ---")
    # print(critic_net)
    # critic_params = sum(p.numel() for p in critic_net.parameters() if p.requires_grad)
    # print(f"Total trainable critic parameters: {critic_params:,}")
    # print("="*60)

    # sys.exit("\nInspection complete. Models seem reasonable. You can now start training.")
    #wraps the critic net in a tensordictmodule
    critic = TensorDictModule(
            module=critic_net,
            in_keys=["shared_observation"],
            out_keys=[("agents", "state_value")] # The critic's output is the value for each agent
        ).to(device)

    #print("Running policy:", policy(env.reset()))
    #print("Running value:", critic_module(env.reset()))

    # # Initialize MAPPO
    mappo = MAPPO(
        env=env, # Pass the transformed environment
        policy=policy,
        critic=critic,
        device=device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=1.0,
        c2=0.01,
        n_epochs=10,
        batch_size=256,#64,
        n_agents=n_agents,
        frames_per_batch=4096, # 4096Increased for better learning
        model_name=base_name,
        checkpoint_interval=100
    )

    # --- Training Loop ---
    total_frames = 2_000_000 # Set to a small number for a quick test
    print(f"Starting MAPPO training with {n_agents} agents using SwarmTorchEnv...")
    try:
        mappo.train(total_frames)
        print("Training complete!")
    finally:
        print("\n--- Closing Environment ---")
        env.close()

    # --- Save Models ---
    # Create a directory for trained models if it doesn't exist
    dir="/home/torchrl/training/models"
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_dir = dir
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the state_dict of the actual networks into the 'models' directory
    torch.save(policy.state_dict(), os.path.join(model_dir, f'{base_name}_policy.pth'))
    torch.save(critic_net.state_dict(), os.path.join(model_dir, f'{base_name}_critic.pth'))
    print(f"Models saved in '{model_dir}' directory!")

if __name__ == "__main__":
    main()