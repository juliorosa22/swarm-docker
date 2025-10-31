import torch
import torch.nn as nn
from torchrl.data import Composite, Bounded, Categorical
from typing import List, Dict
from tensordict import TensorDict

class CentralizedCritic(nn.Module):
    """
    A centralized critic that uses a Transformer-style block to process shared observations.
    Implements a state-value function V(s) for MAPPO.
    """
    def __init__(self, observation_spec: Composite, action_spec, hidden_dim=128, n_attention_heads=4, device="cuda"):
        super().__init__()
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.hidden_dim = hidden_dim
        self.n_attention_heads = n_attention_heads
        self.device = device
        
        self._define_layers()

    def _define_layers(self):
        n_agents = self.observation_spec["inter_agent_distances"].shape[-1]

        # Feature dimension for each agent's individual data
        agent_feature_dim = (
            1  # target_distances
            + self.observation_spec["velocities"].shape[-1]  # velocities
            + 1  # obstacle_distances
        )
        
        # The input to the Transformer will be the agent's features + global context
        # We will add the global context (distance matrix) after an initial embedding
        self.agent_embedding = nn.Sequential(
            nn.Linear(agent_feature_dim, self.hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        # A separate embedding for the global distance information
        self.distance_embedding = nn.Sequential(
            nn.Linear(n_agents * n_agents, self.hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        # --- 2. Transformer Encoder Block ---
        # Attention module
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_attention_heads,
            batch_first=True # Important: This simplifies tensor manipulation
        ).to(self.device)
        
        # Layer Normalization
        self.layernorm1 = nn.LayerNorm(self.hidden_dim).to(self.device)
        self.layernorm2 = nn.LayerNorm(self.hidden_dim).to(self.device)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        ).to(self.device)
        
        # --- 3. Value Head ---
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict = tensordict.to(self.device)
        
        inter_agent_distances = tensordict.get("inter_agent_distances")
        target_distances = tensordict.get("target_distances")
        velocities = tensordict.get("velocities")
        obstacle_distances = tensordict.get("obstacle_distances")

        # --- Handle batch dimension ---
        is_batched = velocities.ndim == 3
        if not is_batched:
            # Unsqueeze all relevant tensors
            inter_agent_distances = inter_agent_distances.unsqueeze(0)
            target_distances = target_distances.unsqueeze(0)
            velocities = velocities.unsqueeze(0)
            obstacle_distances = obstacle_distances.unsqueeze(0)
        
        batch_size = velocities.shape[0]
        n_agents = velocities.shape[1]

        # 1. Create individual agent feature vectors
        per_agent_features = torch.cat([
            target_distances.unsqueeze(-1),
            velocities,
            obstacle_distances.unsqueeze(-1),
        ], dim=-1)

        # 2. Embed individual agent features
        agent_embeddings = self.agent_embedding(per_agent_features)

        # 3. Embed global distance matrix and add it to each agent's embedding
        # Flatten the distance matrix for each batch item
        flat_distances = inter_agent_distances.view(batch_size, -1)
        distance_embedding = self.distance_embedding(flat_distances)
        
        # Add the global context to each agent's representation
        # Unsqueeze distance_embedding to allow broadcasting: [B, 1, D]
        combined_embeddings = agent_embeddings + distance_embedding.unsqueeze(1)

        # 4. Pass through Transformer Encoder Block
        # Attention + Add & Norm
        attn_output, _ = self.attention(combined_embeddings, combined_embeddings, combined_embeddings)
        x = self.layernorm1(combined_embeddings + attn_output)
        
        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        processed_features = self.layernorm2(x + ffn_output)
        
        # 5. Compute state-value V(s)
        # Get a global representation by taking the mean across the agent dimension
        global_state_representation = processed_features.mean(dim=1)
        value = self.value_net(global_state_representation)
        
        # Remove the batch dimension if we added it
        if not is_batched:
            value = value.squeeze(0)
        
        return value