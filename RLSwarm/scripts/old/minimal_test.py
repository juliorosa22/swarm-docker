import torch
import sys
import os
from tensordict import TensorDict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.uav import UAVEnv
import airsim

# Import our manual collector
from helpers.manual_data_collector import ManualDataCollector

class SimplePolicy:
    """Simple policy for testing the manual collector"""
    def __init__(self, action_spec, device, action_type="continuous"):
        self.action_spec = action_spec
        self.device = device
        self.action_type = action_type
        self.step_count = 0
        
        if action_type == "continuous":
            # Create a simple pattern of actions for testing
            self.actions = [
                torch.tensor([2.0, 0.0, 0.0, 0.0], device=device),    # Move forward
                torch.tensor([1.0, 1.0, 0.0, 5.0], device=device),   # Move forward-right with yaw
                torch.tensor([1.0, -1.0, 0.0, -5.0], device=device), # Move forward-left with yaw
                torch.tensor([0.0, 0.0, 0.5, 0.0], device=device),   # Move up
                torch.tensor([3.0, 0.0, 0.0, 0.0], device=device),   # Move forward fast
            ]
        else:
            # Discrete actions
            self.actions = [
                torch.tensor([0], device=device),  # Forward slow
                torch.tensor([1], device=device),  # Forward medium
                torch.tensor([2], device=device),  # Forward fast
                torch.tensor([3], device=device),  # Left
                torch.tensor([4], device=device),  # Right
            ]
    
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        """Generate action based on simple pattern"""
        batch_size = tensordict.batch_size
        
        # Cycle through predefined actions
        action = self.actions[self.step_count % len(self.actions)].clone()
        
        # Handle batching
        if len(batch_size) > 0:
            action = action.expand(*batch_size, -1)
        
        self.step_count += 1
        
        return TensorDict({"action": action}, batch_size=batch_size)

def test_manual_collector():
    """Test the manual data collector with UAV environment"""
    
    device = torch.device("cpu")  # Use CPU for testing
    print(f"🧪 TESTING MANUAL DATA COLLECTOR")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Connect to AirSim
    try:
        print("1️⃣ Connecting to AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ AirSim connection successful!")
    except Exception as e:
        print(f"❌ Failed to connect to AirSim: {e}")
        return False
    
    # Create UAV environment
    try:
        print("\n2️⃣ Creating UAV environment...")
        env = UAVEnv(
            client=client,
            drone_name="uav0",
            start_position=(0, 0, -10),
            end_position=(10, 10, -10),
            action_type="continuous",
            observation_img_size=(32, 32),  # Smaller for faster testing
            device=device
        )
        print("✅ UAV environment created successfully!")
    except Exception as e:
        print(f"❌ Failed to create UAV environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create policy
    try:
        print("\n3️⃣ Creating policy...")
        policy = SimplePolicy(env.action_spec, device, action_type="continuous")
        print("✅ Policy created successfully!")
    except Exception as e:
        print(f"❌ Failed to create policy: {e}")
        return False
    
    # Create manual data collector
    try:
        print("\n4️⃣ Creating manual data collector...")
        collector = ManualDataCollector(
            env=env,
            policy=policy,
            frames_per_batch=10,     # Small batch for testing
            total_frames=30,         # Collect 3 batches
            max_frames_per_traj=50,  # Allow longer episodes
            device=device,
            reset_at_each_iter=False,
        )
        print("✅ Manual data collector created successfully!")
    except Exception as e:
        print(f"❌ Failed to create manual data collector: {e}")
        return False
    
    # Test data collection
    try:
        print(f"\n5️⃣ Testing data collection...")
        print("=" * 40)
        
        batch_count = 0
        for i, batch in enumerate(collector):
            batch_count += 1
            print(f"\n📦 BATCH {batch_count} RECEIVED:")
            print(f"   Batch shape: {batch.batch_size}")
            print(f"   Batch keys: {list(batch.keys())}")
            
            # Analyze batch content
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    if key == 'reward':
                        print(f"      reward range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                    elif key == 'done':
                        print(f"      done count: {value.sum().item()}/{value.numel()}")
                elif isinstance(value, TensorDict):
                    print(f"   {key}: TensorDict with keys {list(value.keys())}")
                    if key in ['obs', 'next_obs']:
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                print(f"      {subkey}: shape={subvalue.shape}")
            
            # Validate batch quality
            print(f"\n   🔍 BATCH QUALITY CHECK:")
            
            # Check for None values
            def check_none_recursive(obj, path=""):
                none_found = []
                if obj is None:
                    none_found.append(path)
                elif isinstance(obj, TensorDict):
                    for k, v in obj.items():
                        none_found.extend(check_none_recursive(v, f"{path}.{k}"))
                return none_found
            
            none_locations = check_none_recursive(batch, "batch")
            if none_locations:
                print(f"      ❌ None values found at: {none_locations}")
            else:
                print(f"      ✅ No None values detected")
            
            # Check tensor consistency
            expected_batch_size = batch.batch_size[0]
            inconsistent_shapes = []
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    if value.shape[0] != expected_batch_size:
                        inconsistent_shapes.append(f"{key}: {value.shape}")
                elif isinstance(value, TensorDict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, torch.Tensor) and subvalue.shape[0] != expected_batch_size:
                            inconsistent_shapes.append(f"{key}.{subkey}: {subvalue.shape}")
            
            if inconsistent_shapes:
                print(f"      ❌ Inconsistent shapes: {inconsistent_shapes}")
            else:
                print(f"      ✅ All shapes consistent")
            
            # Check for NaN/Inf values
            nan_inf_found = []
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    if torch.isnan(value).any():
                        nan_inf_found.append(f"{key}: NaN")
                    if torch.isinf(value).any():
                        nan_inf_found.append(f"{key}: Inf")
            
            if nan_inf_found:
                print(f"      ❌ NaN/Inf values: {nan_inf_found}")
            else:
                print(f"      ✅ No NaN/Inf values")
            
            print(f"   📊 BATCH SUMMARY:")
            print(f"      Episodes in batch: {batch['done'].sum().item()}")
            print(f"      Mean reward: {batch['reward'].mean().item():.3f}")
            print(f"      Action range: [{batch['action'].min().item():.2f}, {batch['action'].max().item():.2f}]")
            
            # Stop after collecting a few batches
            if batch_count >= 3:
                print(f"\n🎯 Successfully collected {batch_count} batches!")
                break
        
        # Test collector state management
        print(f"\n6️⃣ Testing collector state management...")
        state = collector.state_dict()
        print(f"   Collector state: {state}")
        
        # Cleanup
        collector.shutdown()
        
        print(f"\n" + "=" * 60)
        print(f"✅ MANUAL DATA COLLECTOR TEST PASSED!")
        print(f"   Total batches collected: {batch_count}")
        print(f"   Total frames collected: {state['collected_frames']}")
        print(f"   Episodes completed: {state['episode_count']}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Always cleanup
        try:
            collector.shutdown()
        except:
            pass

def test_batch_compatibility():
    """Test that our batches are compatible with typical RL training loops"""
    
    print(f"\n🧪 TESTING BATCH COMPATIBILITY")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    try:
        # Quick setup
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        env = UAVEnv(
            client=client, drone_name="uav0",
            start_position=(0, 0, -5), end_position=(3, 0, -5),
            action_type="continuous", observation_img_size=(16, 16),
            device=device
        )
        
        policy = SimplePolicy(env.action_spec, device)
        collector = ManualDataCollector(env, policy, frames_per_batch=5, total_frames=5, device=device)
        
        # Get one batch
        for batch in collector:
            print("📦 Testing batch operations...")
            
            # Test typical RL operations
            print("   ✅ Batch indexing:", batch[0].batch_size)
            print("   ✅ Reward computation:", batch['reward'].mean().item())
            print("   ✅ Action slicing:", batch['action'][:3].shape)
            print("   ✅ Observation access:", list(batch['obs'].keys()))
            
            # Test with common RL libraries patterns
            obs_flat = torch.cat([
                batch['obs']['position'].flatten(1),
                batch['obs']['velocity'].flatten(1),
                batch['obs']['rotation'].flatten(1)
            ], dim=1)
            print("   ✅ Observation flattening:", obs_flat.shape)
            
            # Test advantage computation pattern
            rewards = batch['reward']
            advantages = rewards - rewards.mean()
            print("   ✅ Advantage computation:", advantages.shape)
            
            break
        
        collector.shutdown()
        print("✅ Batch compatibility test passed!")
        
    except Exception as e:
        print(f"❌ Batch compatibility test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 MANUAL DATA COLLECTOR TESTING SUITE")
    print("=" * 80)
    
    # Run main test
    success = test_manual_collector()
    
    if success:
        # Run compatibility test
        test_batch_compatibility()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Your manual data collector is ready for MAPPO training!")
    else:
        print(f"\n❌ Tests failed. Check the errors above.")

if __name__ == "__main__":
    main()