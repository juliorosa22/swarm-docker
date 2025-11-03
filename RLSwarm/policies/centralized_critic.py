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
    def __init__(self, observation_spec: Composite, action_spec, hidden_dim=64, n_attention_heads=4, device: torch.device = None):
        super().__init__()
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.hidden_dim = hidden_dim
        self.n_attention_heads = n_attention_heads
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_agents = observation_spec["inter_agent_distances"].shape[-1]
        self._define_layers()

    def _define_layers(self):
        n_agents = self.n_agents

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
        #print("Inside CentralizedCritic forward:", tensordict)
        is_from_loss_batch = tensordict["inter_agent_distances"].ndim < 3## means it was batched from the loss module
        if is_from_loss_batch:
            tensordict = tensordict.reshape(-1, self.n_agents) #unflatten the batch to [B, N, ...]
        inter_agent_distances = tensordict.get("inter_agent_distances")
        target_distances = tensordict.get("target_distances")
        velocities = tensordict.get("velocities")
        obstacle_distances = tensordict.get("obstacle_distances")
        batch_dims = velocities.shape[:-2]
        n_agents = velocities.shape[-2]
        # --- Handle batch dimension ---
        
        

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
        #print(f"inter_agent_distances shape: {inter_agent_distances.shape}")
        #print("inter_agent ndim:", inter_agent_distances.ndim)
        if inter_agent_distances.ndim < 3:
            #should undo the minibatch flattening
            #print("reshaping inter_agent_distances ")
            batch_size = inter_agent_distances.shape[0] // n_agents
            #print(f"Calculated batch_size: {batch_size}")
            inter_agent_distances = inter_agent_distances.reshape(batch_size, n_agents, n_agents)
            #print(f"Flattened inter_agent_distances after reshape: {inter_agent_distances.shape}")
            
        
        flat_distances = inter_agent_distances.flatten(start_dim=-2)
        #print(f"flat_distances shape: {flat_distances.shape}")
        distance_embedding = self.distance_embedding(flat_distances)
        
        # Add the global context to each agent's representation
        # Unsqueeze distance_embedding to allow broadcasting: [B, 1, D]
        combined_embeddings = agent_embeddings + distance_embedding.unsqueeze(-2)

        # 4. Pass through Transformer Encoder Block
        # Attention + Add & Norm
        attn_output, _ = self.attention(combined_embeddings, combined_embeddings, combined_embeddings)
        x = self.layernorm1(combined_embeddings + attn_output)
        
        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        processed_features = self.layernorm2(x + ffn_output)
        
        # 5. Compute a single, global state-value V(s)
        # Get a global representation by taking the mean across the agent dimension
        global_state_representation = processed_features.mean(dim=-2)
        global_value = self.value_net(global_state_representation) # Shape: [B, 1]
        
        # 6. Expand the global value to a per-agent shape [B, N, 1]
        # This makes the critic's output directly compatible with per-agent rewards.
        
        per_agent_value = global_value.unsqueeze(-2).expand(*batch_dims, n_agents, 1)
        if is_from_loss_batch:
            per_agent_value = per_agent_value.reshape(-1, 1) #flatten back to minibatch shape
            #print(f"Flattened per_agent_value shape: {per_agent_value.shape}")
        #print(f"Critic output shape (per-agent value): {per_agent_value.shape}")
        
        return per_agent_value