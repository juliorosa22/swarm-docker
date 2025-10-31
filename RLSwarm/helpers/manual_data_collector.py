import torch
from tensordict import TensorDict
from typing import Iterator, Optional, Callable, Union
import time

class ManualDataCollector:
    """
    Manual data collector that mimics SyncDataCollector behavior without the stacking bugs.
    Collects batches of transitions by manually stepping through the environment.
    """
    
    def __init__(
        self,
        env,
        policy: Callable,
        frames_per_batch: int = 50,
        total_frames: int = 1000,
        max_frames_per_traj: int = 200,
        device: Optional[torch.device] = None,
        reset_at_each_iter: bool = False,
        return_contiguous: bool = True,
    ):
        """
        Args:
            env: Environment to collect data from
            policy: Policy function that takes TensorDict and returns actions
            frames_per_batch: Number of frames to collect per batch
            total_frames: Total frames to collect (if -1, collect indefinitely)
            max_frames_per_traj: Maximum steps per episode before forced reset
            device: Device for tensor operations
            reset_at_each_iter: Whether to reset env at start of each batch
            return_contiguous: Whether to return contiguous tensors
        """
        self.env = env
        self.policy = policy
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.max_frames_per_traj = max_frames_per_traj
        self.device = device or torch.device("cpu")
        self.reset_at_each_iter = reset_at_each_iter
        self.return_contiguous = return_contiguous
        
        # Tracking variables
        self.collected_frames = 0
        self.episode_count = 0
        self.current_episode_steps = 0
        self.current_obs = None
        
        print(f"ManualDataCollector initialized:")
        print(f"  frames_per_batch: {frames_per_batch}")
        print(f"  total_frames: {total_frames}")
        print(f"  max_frames_per_traj: {max_frames_per_traj}")
        print(f"  device: {device}")
    
    def __iter__(self) -> Iterator[TensorDict]:
        """Iterator interface to match SyncDataCollector behavior"""
        while self.total_frames == -1 or self.collected_frames < self.total_frames:
            batch = self._collect_batch()
            if batch is not None:
                yield batch
            else:
                break
    
    def _collect_batch(self) -> Optional[TensorDict]:
        """Collect a single batch of transitions"""
        batch_data = []
        frames_in_batch = 0
        
        print(f"\n🔄 Collecting batch {len(batch_data)//self.frames_per_batch + 1}")
        print(f"  Target frames: {self.frames_per_batch}")
        print(f"  Total collected so far: {self.collected_frames}")
        
        # Reset environment if needed or if starting fresh
        if self.current_obs is None or self.reset_at_each_iter:
            self.current_obs = self._safe_reset()
            self.current_episode_steps = 0
        
        while frames_in_batch < self.frames_per_batch:
            # Check if we've reached total frames limit
            if self.total_frames != -1 and self.collected_frames >= self.total_frames:
                break
            
            # Collect one transition
            transition = self._collect_single_transition()
            
            if transition is not None:
                batch_data.append(transition)
                frames_in_batch += 1
                self.collected_frames += 1
                self.current_episode_steps += 1
                
                if frames_in_batch % 10 == 0 or frames_in_batch == self.frames_per_batch:
                    print(f"    Collected {frames_in_batch}/{self.frames_per_batch} frames")
            
            # Check if episode should end
            if self._should_reset_episode():
                self.current_obs = self._safe_reset()
                self.current_episode_steps = 0
                self.episode_count += 1
                print(f"    Episode {self.episode_count} completed, resetting...")
        
        if not batch_data:
            print("⚠️  No valid transitions collected")
            return None
        
        # Convert list of transitions to batched TensorDict
        try:
            batched_data = self._create_batch_from_transitions(batch_data)
            print(f"✅ Batch ready: {len(batch_data)} transitions")
            return batched_data
        except Exception as e:
            print(f"❌ Error creating batch: {e}")
            return None
    
    def _collect_single_transition(self) -> Optional[TensorDict]:
        """Collect a single environment transition"""
        try:
            # Generate action from current observation
            with torch.no_grad():
                action_td = self.policy(self.current_obs)
            
            # Validate action
            if not self._validate_action(action_td):
                print(f"⚠️  Invalid action generated, skipping step")
                return None
            
            # Take environment step
            next_obs = self.env.step(action_td)
            
            # Validate step result
            if not self._validate_step_result(next_obs):
                print(f"⚠️  Invalid step result, skipping transition")
                return None
            
            # Create transition with current obs, action, reward, done, next_obs
            transition = TensorDict({
                'obs': self._extract_obs(self.current_obs),
                'action': action_td['action'].clone(),
                'reward': next_obs['reward'].clone(),
                'done': next_obs['done'].clone(),
                'next_obs': self._extract_obs(next_obs),
            }, batch_size=torch.Size([]))
            
            # Update current observation for next step
            self.current_obs = next_obs
            
            return transition
            
        except Exception as e:
            print(f"❌ Error collecting transition: {e}")
            return None
    
    def _safe_reset(self) -> TensorDict:
        """Safely reset the environment with error handling"""
        try:
            reset_result = self.env.reset()
            if not self._validate_reset_result(reset_result):
                raise ValueError("Invalid reset result")
            return reset_result
        except Exception as e:
            print(f"❌ Error resetting environment: {e}")
            # Return a minimal valid observation
            return self._create_fallback_obs()
    
    def _should_reset_episode(self) -> bool:
        """Determine if the current episode should be reset"""
        if self.current_obs is None:
            return True
        
        # Check if done flag is set
        done = self.current_obs.get('done', torch.tensor([False]))
        if done.any():
            return True
        
        # Check if max trajectory length reached
        if self.current_episode_steps >= self.max_frames_per_traj:
            return True
        
        return False
    
    def _extract_obs(self, tensordict: TensorDict) -> TensorDict:
        """Extract observation from tensordict"""
        if 'obs' in tensordict:
            return tensordict['obs'].clone()
        else:
            # If no 'obs' key, assume the whole tensordict is the observation
            obs_dict = {}
            for key, value in tensordict.items():
                if key not in ['reward', 'done', 'action']:
                    obs_dict[key] = value.clone()
            return TensorDict(obs_dict, batch_size=torch.Size([]))
    
    def _create_batch_from_transitions(self, transitions: list) -> TensorDict:
        """Convert list of transitions to a batched TensorDict"""
        if not transitions:
            raise ValueError("Cannot create batch from empty transition list")
        
        # Get all keys from first transition
        keys = transitions[0].keys()
        
        # Stack each key across all transitions
        batched_dict = {}
        for key in keys:
            try:
                # Collect all values for this key
                values = [t[key] for t in transitions]
                
                # Check if all values are valid tensors
                for i, val in enumerate(values):
                    if val is None:
                        raise ValueError(f"None value found in transition {i} for key '{key}'")
                    if not isinstance(val, torch.Tensor):
                        raise ValueError(f"Non-tensor value in transition {i} for key '{key}': {type(val)}")
                
                # Stack tensors
                if isinstance(values[0], TensorDict):
                    # For nested TensorDicts (like observations)
                    batched_dict[key] = torch.stack(values, dim=0)
                else:
                    # For regular tensors
                    batched_dict[key] = torch.stack(values, dim=0)
                    
            except Exception as e:
                print(f"❌ Error stacking key '{key}': {e}")
                raise
        
        batch = TensorDict(batched_dict, batch_size=torch.Size([len(transitions)]))
        
        # Make contiguous if requested
        if self.return_contiguous:
            batch = batch.contiguous()
        
        return batch
    
    def _validate_action(self, action_td: TensorDict) -> bool:
        """Validate action tensordict"""
        if action_td is None:
            return False
        if 'action' not in action_td:
            return False
        if action_td['action'] is None:
            return False
        return True
    
    def _validate_step_result(self, step_result: TensorDict) -> bool:
        """Validate step result from environment"""
        if step_result is None:
            return False
        required_keys = ['reward', 'done']
        for key in required_keys:
            if key not in step_result or step_result[key] is None:
                return False
        return True
    
    def _validate_reset_result(self, reset_result: TensorDict) -> bool:
        """Validate reset result from environment"""
        if reset_result is None:
            return False
        # Should have observation data
        return True
    
    def _create_fallback_obs(self) -> TensorDict:
        """Create a minimal fallback observation"""
        return TensorDict({
            'obs': TensorDict({
                'depth_image': torch.ones((1, 16, 16), device=self.device) * 50.0,
                'position': torch.zeros(3, device=self.device),
                'rotation': torch.zeros(3, device=self.device),
                'velocity': torch.zeros(3, device=self.device),
            }),
            'reward': torch.tensor([0.0], device=self.device),
            'done': torch.tensor([False], device=self.device),
        }, batch_size=torch.Size([]))
    
    def shutdown(self):
        """Cleanup method to match SyncDataCollector interface"""
        try:
            if hasattr(self.env, 'close'):
                self.env.close()
        except Exception as e:
            print(f"Warning during shutdown: {e}")
        print("ManualDataCollector shut down successfully")
    
    def set_seed(self, seed: int) -> int:
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if hasattr(self.env, 'set_seed'):
            return self.env.set_seed(seed)
        return seed
    
    def state_dict(self) -> dict:
        """Get collector state for checkpointing"""
        return {
            'collected_frames': self.collected_frames,
            'episode_count': self.episode_count,
            'current_episode_steps': self.current_episode_steps,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load collector state from checkpoint"""
        self.collected_frames = state_dict.get('collected_frames', 0)
        self.episode_count = state_dict.get('episode_count', 0)
        self.current_episode_steps = state_dict.get('current_episode_steps', 0)