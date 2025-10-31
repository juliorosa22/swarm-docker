import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
from tensordict import TensorDict
import os


class CNNFeatureExtractor(nn.Module):
    """
    CNN model to extract features from depth images.
    Optimized for depth information processing with smaller network.
    """
    def __init__(self, image_shape: Tuple[int, int, int] = (1, 64, 64), output_dim: int = 32):
        super(CNNFeatureExtractor, self).__init__()
        
        # Define convolutional layers first
        self.conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Robustly calculate the flattened feature size with a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            conv_output = self.conv_net(dummy_input)
            flattened_size = conv_output.view(1, -1).shape[1]
        
        self.output_dim = output_dim
        self.fc = nn.Linear(flattened_size, self.output_dim)
        

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_net(x)
        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)
        # Pass through the fully connected layer
        x = F.relu(self.fc(x))
        return x


class VectorFeatureExtractor(nn.Module):
    """
    MLP to process vector inputs (position, velocity, etc.)
    Based on common architectures used in robotic navigation.
    """
    def __init__(self, input_dim: int):
        super(VectorFeatureExtractor, self).__init__()
        
        # Reduced hidden dimensions based on input size
        h1_dim = max(32, 2 * input_dim)  # First hidden layer
        h2_dim = max(32, input_dim)      # Second hidden layer
        
        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(input_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        
        # Output dimension
        self.output_dim = h2_dim
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Normalize inputs
        x = self.ln1(x)
        
        # First layer with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second layer with ReLU
        x = F.relu(self.fc2(x))
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for handling variable agent counts.
    Updated for full self-attention with pairwise masking (e.g., for graphs).
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Compute multi-head attention (supports self-attention for graphs).
        
        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim] (e.g., n_agents for self-attn)
            key: Key tensor [batch_size, seq_len_k, embed_dim] (e.g., n_agents)
            value: Value tensor [batch_size, seq_len_v, embed_dim] (e.g., n_agents)
            mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k] (for pairwise masking)
            
        Returns:
            attn_output: Attention output [batch_size, seq_len_q, embed_dim]
            attn_weights: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Linear projections
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q, D/H]
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, K, D/H]
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, K, D/H]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, Q, K]
        
        # Apply mask if provided
        if mask is not None:
            # Support both boolean and additive masks
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)  # [B, H, Q, K]
            else:
                attn_scores = attn_scores + mask.unsqueeze(1)  # Additive mask [B, H, Q, K]
        
        # Softmax for attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights
        attn_output = torch.matmul(attn_weights, v)  # [B, H, Q, D/H]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len_q, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

