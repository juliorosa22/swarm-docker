import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor, NormalParamExtractor
from helpers.features_extractors import CNNFeatureExtractor, VectorFeatureExtractor
from torch.distributions import OneHotCategorical,Categorical, Normal 


class FullPolicyModule(nn.Module):
    def __init__(self, cnn, vector_net, policy_net, action_head):
        super().__init__()
        self.cnn = cnn
        self.vector_net = vector_net
        self.policy_net = policy_net
        self.action_head = action_head
        
    
    def forward(self, agents_td):
        # agents_td is a tensordict where each entry has shape (n_agents, ...)
        original_shape = agents_td.batch_size
        
        # --- Feature Extraction ---
        depth_image = agents_td.get("depth_image")
        cnn_input = depth_image.reshape(-1, *depth_image.shape[-3:])
        img_features = self.cnn(cnn_input)
        
        vector_input_list = [
            agents_td.get("position"),
            agents_td.get("rotation"),
            agents_td.get("velocity"),
            agents_td.get("target_distance"),
            agents_td.get("front_obs_distance")
        ]
        concatenated_vectors = torch.cat(vector_input_list, dim=-1)
        vector_input = concatenated_vectors.reshape(-1, concatenated_vectors.shape[-1])
        vector_features = self.vector_net(vector_input)
        
        # Combine features
        combined_features = torch.cat([img_features, vector_features], dim=-1)
        
        # Policy and action head
        policy_features = self.policy_net(combined_features)
        action_output = self.action_head(policy_features)
        return action_output
        # --- Reshape and return the RAW tensor ---
        # The TensorDictModule wrapper will handle placing this in the main tensordict.
        # if self.out_keys == ["logits"]:
        #     action_output = action_output.view(*original_shape, -1)
        #     return action_output # Return the raw tensor
        # else:
        #     # For continuous actions, NormalParamExtractor outputs a tuple
        #     loc, scale = action_output
        #     loc = loc.view(*original_shape, -1)
        #     scale = scale.view(*original_shape, -1)
        #     # We need to stack them to return a single tensor that the wrapper can handle
        #     return torch.stack([loc, scale], dim=-1)


class DecentralizedPolicy(nn.Module):
    """
    Decentralized policy network with shared weights across agents.
    Takes stacked observations as input and outputs actions via an internal ProbabilisticActor.
    Supports variable number of agents and discrete/continuous actions.
    """
    def __init__(
        self,
        observation_spec: Dict,
        action_spec: Dict,
        hidden_dim: int = 64,
        device: torch.device = None
    ):
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_discrete = hasattr(action_spec, 'n')
        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.hidden_dim = hidden_dim
        
        # Make out_keys a local variable, not an instance attribute (self.out_keys)
        out_keys = ["logits"] if self.is_discrete else ["loc", "scale"]
        
        self._define_layers()

        module = FullPolicyModule(self.cnn,self.vector_net,self.policy_net,self.action_head)

        # This module will take the "agents" sub-TD and its output will also be placed within "agents".
        self.policy_module = TensorDictModule(
            module=module,
            in_keys=["agents"],
            out_keys=[("agents", key) for key in out_keys] # Nested output keys
        ).to(self.device)
        
        # The distribution parameters are now nested, e.g., ("agents", "logits")
        dist_param_keys = [("agents", key) for key in out_keys]
        
        # The final action should also be nested, e.g., ("agents", "action")
        env_action_key = [("agents", "action")]
        
        # Use torch.distributions.Categorical for discrete actions to get integer indices
        dist_class = Categorical if self.is_discrete else Normal
        
        self.policy = ProbabilisticActor(
            module=self.policy_module,
            in_keys=dist_param_keys,    # Use the nested keys for distribution params
            out_keys=env_action_key,    # Use the nested key for the final action
            spec=self.action_spec,      # The spec for the multi-agent action
            distribution_class=dist_class,
            return_log_prob=True,
            safe=True
        ).to(self.device)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Delegate to ProbabilisticActor
        #rint(f"inside forward of policy:{tensordict}")
        return self.policy(tensordict)

    def _define_layers(self):
        # Feature extractors
        
        h, w = self.observation_spec["depth_image"].shape[-2:]
        #print(f"image spec :{self.observation_spec['depth_image'].shape}")
        self.cnn = CNNFeatureExtractor(image_shape=(1, h, w)).to(self.device)
        vector_dim = 0
        for key in ["position", "rotation", "velocity", "target_distance", "front_obs_distance"]:
            vector_dim += self.observation_spec[key].shape[-1]
        self.vector_net = VectorFeatureExtractor(vector_dim).to(self.device)
        combined_dim = self.cnn.output_dim + self.vector_net.output_dim
        
        # Policy head (linear layers only)
        # This module operates on a raw tensor inside FullPolicyModule, so it MUST be nn.Sequential.
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        ).to(self.device)
        
        # Action head
        if self.is_discrete:
            action_dim = self.action_spec.n
            self.action_head = nn.Linear(self.hidden_dim, action_dim).to(self.device)
        else:
            action_dim = self.action_spec.shape[-1]
            self.action_head = NormalParamExtractor(self.hidden_dim, action_dim).to(self.device)
        #print(f"#### Action head output dim: {self.action_head} ####")