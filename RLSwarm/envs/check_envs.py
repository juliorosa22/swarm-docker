import sys
import os
from gymnasium.utils.env_checker import check_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('C:\\Users\\julio\\OneDrive\\Documents\\Programming\\Drones\\Airsim\\PythonClient')
from envs.uav import UAVGymEnv
from envs.swarm import SwarmGymEnv
from envs.swarm_config import SwarmConfig
from envs.uav_config import UAVConfig
import airsim
import numpy as np

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("AirSim connected!")

# Test UAV Environment
print("\n=== Testing UAV Environment ===")
uav_config = UAVConfig(
    drone_name="uav2",
    start_position=(0.0, 0.0, -5.0),
    end_position=(10.0, 10.0, -5.0),
    action_type="discrete"  # Changed to discrete to match swarm expectations
)
uav_env = UAVGymEnv(client=client, config=uav_config)

try:
    check_env(uav_env)
    print("UAV Environment passes all checks!")
except Exception as e:
    print(f"UAV Environment has issues: {e}")

# Test Swarm Environment
print("\n=== Testing Swarm Environment ===")
swarm_config = SwarmConfig(
    n_agents=3,  # Small number for testing
    min_distance_threshold=2.0,
    max_formation_distance=10.0,
    safety_penalty_weight=1.0,
    formation_bonus_weight=0.5,
    coordination_bonus_weight=0.2,
    max_safety_penalty=5.0,
    max_steps=100
)
swarm_env = SwarmGymEnv(config=swarm_config)

try:
    check_env(swarm_env)
    print("Swarm Environment passes all checks!")
except Exception as e:
    print(f"Swarm Environment has issues: {e}")

# Additional basic tests
print("\n=== Additional Basic Tests ===")

# Test UAV reset and step
print("Testing UAV reset and step...")
obs, info = uav_env.reset()  # Now returns 2 values
print(f"UAV Reset successful. Obs keys: {list(obs.keys())}")

action = uav_env.action_space.sample()
obs, reward, terminated, truncated, info = uav_env.step(action)
print(f"UAV Step successful. Reward: {reward}, Terminated: {terminated}")

# Test Swarm reset and step
print("Testing Swarm reset and step...")
swarm_obs, infos = swarm_env.reset()  # Now returns 2 values
print(f"Swarm Reset successful. Obs keys: {list(swarm_obs.keys())}")

# Fixed: Sample actions correctly as a tuple
actions = swarm_env.action_space.sample()  # Returns a tuple of n_agents actions
swarm_obs, swarm_rewards, terminateds, truncateds, infos = swarm_env.step(actions)
print(f"Swarm Step successful. Rewards: {swarm_rewards}")

# Close environments
uav_env.close()
swarm_env.close()
print("Environments closed successfully.")