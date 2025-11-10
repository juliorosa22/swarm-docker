import torch
from tensordict import TensorDict
from typing import Optional
from .swarm_config import SwarmConfig

class SwarmRewards:
    """
    Modular reward computation class for swarm navigation tasks.
    Provides multiple reward function implementations that can be easily swapped.
    """
    
    def __init__(
        self,
        config: SwarmConfig,
        device: torch.device
    ):
        """
        Initialize the reward computer.
        
        Args:
            config: SwarmConfig object containing environment parameters
            n_agents: Number of agents in the swarm
            device: Torch device (cuda/cpu)
        """
        self.config = config
        self.n_agents = config.n_agents
        self.device = device
        
        # Tracking variables for reward computation
        self.last_target_distances: Optional[torch.Tensor] = None
        self.initial_target_distances: Optional[torch.Tensor] = None
        self.end_positions: Optional[torch.Tensor] = None

    def update_tracking_vars(self, tracked_vars: TensorDict):
        """
        Update tracking variables needed for reward computation.
        This updating is done at step method each time.
        
        Args:
            tracked_vars: TensorDict containing:
                - last_target_distances: Previous step distances to targets
                - initial_target_distances: Initial distances at episode start
                - end_positions: Target positions for each agent
        """
        self.last_target_distances = tracked_vars["last_target_distances"]
        self.initial_target_distances = tracked_vars["initial_target_distances"]
        self.end_positions = tracked_vars["end_positions"]

    def compute_rewards(
        self,
        current_obs: TensorDict,
        collisions_this_step: torch.Tensor
    ) -> torch.Tensor:
        """
        Basic reward function with progress, obstacle avoidance, and formation penalties.
        
        Args:
            current_obs: Current observation TensorDict
            collisions_this_step: Boolean tensor of collisions
            
        Returns:
            Reward tensor of shape (n_agents,)
        """
        reward_weights = {
            'progress': 1.5,
            'obstacle': 1.0,
            'formation': 0.5,
            'step': 0.01
        }

        current_target_distances = current_obs["shared_observation", "target_distances"]
        front_obs_distances = current_obs["shared_observation", "obstacle_distances"]
        distance_matrix = current_obs["shared_observation", "inter_agent_distances"]

        # 1. Progress Reward
        dist_improvement = self.last_target_distances - current_target_distances
        progress_reward = dist_improvement * reward_weights['progress']

        # 2. Obstacle Penalty
        obs_threshold = self.config.obstacle_threshold
        obs_violation = torch.clamp((obs_threshold - front_obs_distances) / obs_threshold, min=0)
        obstacle_penalty = (obs_violation ** 2) * reward_weights['obstacle']

        # 3. Formation Penalty
        form_threshold = self.config.min_distance_threshold
        form_violation = torch.clamp((form_threshold - distance_matrix) / form_threshold, min=0)
        torch.diagonal(form_violation).fill_(0)
        formation_penalty = torch.sum(form_violation ** 2, dim=1) * reward_weights['formation']

        # 4. Step Penalty
        step_penalty = reward_weights['step']

        # Combine and clip rewards
        total_rewards = progress_reward - obstacle_penalty - formation_penalty - step_penalty
        individual_rewards = torch.clamp(total_rewards, -1.0, 1.0)

        # Update tracking
        self.last_target_distances = current_target_distances.clone()
        
        return individual_rewards

    def _laplace_formation_reward(self, distance_matrix, alpha=1.0, beta=0.5, lambda1=2.0, lambda2=0.5, as_penalty=False):
        """
        Laplace-based dense reward for cooperative inter-agent spacing.
        
        Args:
            distance_matrix: Inter-agent distance matrix (N, N)
            alpha: Repulsion strength
            beta: Attraction strength
            lambda1: Repulsion decay rate
            lambda2: Attraction decay rate
            as_penalty: If True, return positive values for bad formation (penalty-style)
        
        Returns:
            Reward/penalty tensor (N,)
        """
        n_agents = distance_matrix.shape[0]
        distance_matrix = distance_matrix.clone()
        torch.diagonal(distance_matrix).fill_(float("inf"))

        # Compute pairwise Laplace potentials
        U_rep = alpha * torch.exp(-lambda1 * distance_matrix)
        U_att = beta * torch.exp(-lambda2 * distance_matrix)
        U_total = U_rep - U_att  # potential field

        # Per-agent mean potential (lower = better)
        mean_potential = torch.sum(U_total, dim=1) / (n_agents - 1)

        if as_penalty:
            # Return penalty (higher potential = higher penalty)
            penalty = mean_potential
            return torch.clamp(penalty, min=-1.0, max=5.0)
        else:
            # Return reward (lower potential = higher reward)
            reward = -mean_potential
            return torch.clamp(reward, min=-5.0, max=1.0)
    
    def simple_reward(self, current_obs: TensorDict, collisions_this_step: torch.Tensor):
        """
        A MAPPO-stable, negative-based reward function with balanced and pre-clipped components.
        """
        # -------------------- CONFIG (Re-balanced for Stability) --------------------
        w_progress = 1.0
        w_obstacle = 1.5
        w_neighbor = 1.5
        w_laplacian = 0.8
        w_collision = 5.0   # Strong penalty for collisions
        w_time = 0.01
        w_goal_bonus = 5.0  # Reduced to be less greedy

        proximity_threshold = 3.0
        safe_obs_dist = getattr(self.config, "safe_obs_distance", 5.0)
        safe_neighbor_dist = self.config.min_distance_threshold

        # -------------------- EXTRACT OBS --------------------
        positions = current_obs["agents", "position"]
        velocities = current_obs["agents", "velocity"]
        targets = self.end_positions
        curr_dists = current_obs["shared_observation", "target_distances"]
        front_obs_distances = current_obs["shared_observation", "obstacle_distances"]
        distance_matrix = current_obs["shared_observation", "inter_agent_distances"]

        # =====================================================
        # 1. PROGRESS TERM (Range: [-12, -3])
        # =====================================================
        dist_delta = self.last_target_distances - curr_dists
        goal_vec = targets - positions
        goal_dir = goal_vec / (goal_vec.norm(dim=-1, keepdim=True) + 1e-6)
        vel_dir = velocities / (velocities.norm(dim=-1, keepdim=True) + 1e-6)
        alignment = (goal_dir * vel_dir).sum(dim=-1)
        normalized_dist = curr_dists / (self.initial_target_distances + 1e-6)
        base_progress = -5.0 * normalized_dist - 5.0
        modulation = 0.2 * torch.clamp(dist_delta, -5.0, 5.0) + 0.5 * alignment
        progress_term = torch.clamp(w_progress * (base_progress + modulation), -12.0, -3.0)

        # =====================================================
        # 2. OBSTACLE & NEIGHBOR PENALTIES (Range: [-3.0, 0])
        # =====================================================
        obs_penalty = -w_obstacle * torch.exp(-torch.clamp(front_obs_distances, 0.2, 50.0) / safe_obs_dist)
        obs_penalty = torch.clamp(obs_penalty, -3.0, 0.0)

        if self.n_agents > 1:
            dist_no_self = distance_matrix.clone()
            torch.diagonal(dist_no_self).fill_(float("inf"))
            nearest_neighbor_dist, _ = dist_no_self.min(dim=1)
            neighbor_penalty = -w_neighbor * torch.exp(-nearest_neighbor_dist / safe_neighbor_dist)
            neighbor_penalty = torch.clamp(neighbor_penalty, -3.0, 0.0)
        else:
            neighbor_penalty = torch.zeros_like(obs_penalty)

        # =====================================================
        # 3. FORMATION PENALTY (Range: [-4.0, 0])
        # =====================================================
        if self.n_agents > 1:
            formation_potential = self._laplace_formation_reward(distance_matrix, as_penalty=True)
            laplacian_penalty = -w_laplacian * formation_potential # as_penalty returns positive for bad, so we negate
            laplacian_penalty = torch.clamp(laplacian_penalty, -4.0, 0.0)
        else:
            laplacian_penalty = torch.zeros_like(progress_term)

        # =====================================================
        # 4. COLLISION PENALTY (Sparse, high-impact)
        # =====================================================
        collision_penalty = torch.zeros(self.n_agents, device=self.device)
        collision_penalty[collisions_this_step] = -w_collision

        # =====================================================
        # 5. GOAL PROXIMITY BONUS (Sparse, balanced)
        # =====================================================
        goal_bonus = torch.zeros(self.n_agents, device=self.device)
        goal_bonus[curr_dists < proximity_threshold] = w_goal_bonus

        # =====================================================
        # 6. TIME PENALTY (Constant)
        # =====================================================
        time_penalty = -w_time

        # =====================================================
        # 7. TOTAL REWARD (NO FINAL CLAMP)
        # =====================================================
        total_reward = (
            progress_term +
            obs_penalty +
            neighbor_penalty +
            laplacian_penalty +
            collision_penalty +
            goal_bonus +
            time_penalty
        )

        # Update tracking
        self.last_target_distances = curr_dists.clone()
        
        return total_reward
