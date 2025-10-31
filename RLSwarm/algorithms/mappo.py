import torch
import time
import os
from typing import Dict, List
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization implementation using TorchRL.
    """
    def __init__(
        self,
        env,
        policy,
        critic,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,  # Value loss coefficient
        c2: float = 0.01,  # Entropy coefficient
        n_epochs: int = 10,
        batch_size: int = 64,
        n_agents: int = 5,
        frames_per_batch: int = 1000,
        model_name: str = "mappo_run", # 1. Add model_name parameter
        checkpoint_interval: int = 100
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.env = env
        self.policy = policy
        self.n_agents = n_agents
        self.critic = critic
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name

        # --- 2 & 3. Create structured directories for logs and checkpoints ---
        time_str = time.strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join("runs", f"{self.model_name}_{time_str}")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Data collector
        self.collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=device,
            reset_at_each_iter=True,
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            batch_size=batch_size
        )
        
        # PPO loss
        self.loss_module = ClipPPOLoss(
            actor_network=policy.policy, # Pass the inner ProbabilisticActor directly
            critic_network=self.critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=True,
            loss_critic_type="l2",
            entropy_coef=c2,
            critic_coef=c1,
            normalize_advantage=False,
            safe=True  # This tells the loss to ignore "next" keys automatically
        )
        ### Setting GAE like the tutorial way
        self.loss_module.set_keys(
            reward=env.reward_key,  # Ensure reward key is set correctly
            action=env.action_key,  # Ensure action key is set correctly
            value=("agents", "state_value"),  # Critic value key
            done=("agents", "done"),  # Done key
        )
        self.loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=gae_lambda)
        self.gae = self.loss_module.value_estimator

        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=lr)
        
        # TensorBoard Summary Writer pointed to the new log directory
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.last_avg_reward = 0.0 # Initialize last average reward
    
    def train_deprecated(self, total_frames: int):#not working well
        """Train the MAPPO agent."""
        # Wrap the training loop with tqdm for a progress bar
        progress_bar = tqdm(range(0, total_frames, self.collector.frames_per_batch))
        
        for frame_count in progress_bar:
            # --- Data Collection ---
            data = self.collector.next()
            
            # --- GAE Computation ---
            with torch.no_grad():
                self.gae(data)
            
            # Extract episode rewards for logging
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            avg_reward = episode_rewards.mean().item() if len(episode_rewards) > 0 else 0.0
            
            # --- Learning ---
            # Add collected data to the replay buffer
            self.buffer.extend(data)
            
            # Initialize loss tracking for this iteration
            total_policy_loss, total_value_loss, total_entropy_loss = 0, 0, 0
            
            for _ in range(self.n_epochs):
                for batch in self.buffer: # Iterate over mini-batches
                    # Perform a single update step
                    print("Batch keys:", batch.keys())
                    print("Batch data:", batch)
                    loss_dict = self.update(batch)
                    
                    # Accumulate losses
                    total_policy_loss += loss_dict["loss_objective"]
                    total_value_loss += loss_dict["loss_critic"]
                    total_entropy_loss += loss_dict["loss_entropy"]
            
            # Calculate average losses for this iteration
            num_updates = self.n_epochs * (len(self.buffer) // self.buffer.batch_size)
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_entropy_loss = total_entropy_loss / num_updates
            avg_total_loss = avg_policy_loss + avg_value_loss + avg_entropy_loss

            # --- Logging ---
            self.writer.add_scalar("Loss/Total", avg_total_loss, frame_count)
            self.writer.add_scalar("Loss/Policy", avg_policy_loss, frame_count)
            self.writer.add_scalar("Loss/Value", avg_value_loss, frame_count)
            self.writer.add_scalar("Loss/Entropy", avg_entropy_loss, frame_count)
            self.writer.add_scalar("Reward/Average", avg_reward, frame_count)
            
            # Update the progress bar with the latest metrics
            progress_bar.set_postfix({
                "Avg Reward": f"{avg_reward:.2f}",
                "Total Loss": f"{avg_total_loss:.4f}"
            })

    def train(self, total_frames: int):
        """
        An alternative training loop for MAPPO based on the torchrl tutorial,
        using the collector as an iterator.
        """
        # Initialize the progress bar to track total frames
        progress_bar = tqdm(total=total_frames, unit="frames")
        
        collected_frames = 0

        for i, tensordict_data in enumerate(self.collector):
            # Stop the loop once the total frame count is reached
            if collected_frames >= total_frames:
                break

            # --- GAE Computation ---
            # We need to expand the done and terminated signals to match the multi-agent shape
            # This is expected by the GAE value estimator.
            # The target shape is the same as the per-agent value, e.g., [B, N_AGENTS, 1]
            #print("tensordict data inspect",tensordict_data)
            
            
            tensordict_data.set(
                ("next", "agents", "done"),
                tensordict_data.get(("next", "done")).expand(tensordict_data.get_item_shape(("next",self.env.reward_key))),
            )
            tensordict_data.set(
                ("next", "agents", "terminated"),
                tensordict_data.get(("next", "terminated")).expand(tensordict_data.get_item_shape(("next",self.env.reward_key))),
            )

            with torch.no_grad():
                self.gae(tensordict_data)

            # --- Learning ---
            # Flatten the batch size to shuffle data across frames and agents
            data_view = tensordict_data.reshape(-1)
            self.buffer.extend(data_view)

            # Initialize loss tracking for this iteration
            total_loss_objective = 0.0
            total_loss_critic = 0.0
            total_loss_entropy = 0.0

            for _ in range(self.n_epochs):
                for _ in range(self.collector.frames_per_batch // self.batch_size):
                    subdata = self.buffer.sample()
                    
                    # Ensure the sub-batch is on the correct device
                    subdata = subdata.to(self.device)

                    loss_vals = self.loss_module(subdata)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    # Clip gradients for all parameters managed by the loss module
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Accumulate losses
                    total_loss_objective += loss_vals["loss_objective"].item()
                    total_loss_critic += loss_vals["loss_critic"].item()
                    total_loss_entropy += loss_vals["loss_entropy"].item()


            # Update the policy weights in the collector for the next rollout
            self.collector.update_policy_weights_()

            # --- Checkpointing ---
            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(i + 1)

            # --- Logging ---
            # Increment the collected frames and update the progress bar by the batch size
            frames_in_batch = tensordict_data.numel()
            progress_bar.update(frames_in_batch)
            collected_frames += frames_in_batch
            
            # Prevent division by zero if no updates were made
            num_updates = self.n_epochs * (self.collector.frames_per_batch // self.batch_size)
            if num_updates > 0:
                avg_loss_objective = total_loss_objective / num_updates
                avg_loss_critic = total_loss_critic / num_updates
                avg_loss_entropy = total_loss_entropy / num_updates
            else:
                avg_loss_objective = 0.0
                avg_loss_critic = 0.0
                avg_loss_entropy = 0.0

            avg_total_loss = avg_loss_objective + avg_loss_critic + avg_loss_entropy
            
            done = tensordict_data.get(("next", "done"))
            episode_rewards = tensordict_data["next", "episode_reward"][done]
            
            # Only update the average reward if new episodes have finished
            if len(episode_rewards) > 0:
                self.last_avg_reward = episode_rewards.mean().item()

            self.writer.add_scalar("Loss/Total", avg_total_loss, collected_frames)
            self.writer.add_scalar("Loss/Policy", avg_loss_objective, collected_frames)
            self.writer.add_scalar("Loss/Value", avg_loss_critic, collected_frames)
            self.writer.add_scalar("Loss/Entropy", avg_loss_entropy, collected_frames)
            self.writer.add_scalar("Reward/Average", self.last_avg_reward, collected_frames)

            # Update the progress bar with the latest metrics using set_postfix
            progress_bar.set_postfix({
                "Avg Reward": f"{self.last_avg_reward:.2f}",
                "Total Loss": f"{avg_total_loss:.4f}",
                "Policy Loss": f"{avg_loss_objective:.4f}",
                "Value Loss": f"{avg_loss_critic:.4f}",
            })
        
        # Close the progress bar at the end
        progress_bar.close()

    def save_checkpoint(self, iteration: int):
        """Saves a checkpoint of the policy and critic models."""
        # 4. Use new naming convention and save path
        policy_filename = f'policy_{self.model_name}_iter_{iteration}.pth'
        critic_filename = f'critic_{self.model_name}_iter_{iteration}.pth'
        
        policy_path = os.path.join(self.checkpoint_dir, policy_filename)
        critic_path = os.path.join(self.checkpoint_dir, critic_filename)
        
        # Save the state_dict of the actual networks
        torch.save(self.policy.state_dict(), policy_path)
        # The critic is wrapped, so we save the underlying module's state_dict
        torch.save(self.critic.module.state_dict(), critic_path)
        
        print(f"\n[Checkpoint] Saved models at iteration {iteration} to '{self.checkpoint_dir}'")

    def update(self, batch: TensorDict) -> Dict[str, float]:
        """Perform a single MAPPO update step on a mini-batch of data."""
        # Ensure the batch is on the correct device
        batch = batch.to(self.device)
        
        # Compute PPO loss
        loss_vals = self.loss_module(batch)
        
        # The total loss is a sum of policy, value, and entropy losses
        loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.critic_module.parameters()), max_norm=1.0)
        self.optimizer.step()
        
        # Return a dictionary of detached loss values for logging
        return {key: val.item() for key, val in loss_vals.items()}