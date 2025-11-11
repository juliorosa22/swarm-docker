from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import random
import json
import numpy as np  # Ensure this is imported for _generate_inverted_v_positions
from .deprecated.uav_config import UAVConfig

@dataclass
class SwarmConfig:
    """Configuration class for Swarm parameters, using UAVConfig for UAV-specific settings."""
    
    # General swarm settings
    n_agents: int = 5
    max_steps: int = 500
    
    # Shared UAV parameters (used to create UAVConfig for each agent)
    action_type: str = "discrete"
    observation_img_size: Tuple[int, int] = (64, 64)
    obstacle_threshold: float = 2.0 # Distance to consider for obstacle avoidance
    goal_threshold: float = 5.0 # Distance to consider goal reached
    max_target_distance: Optional[float] = None
    max_obstacle_distance: float = 50.0 # Max distance to consider for obstacle observations, this is the limit of Distance sensor range
    
    # Position-related (can be auto-generated or provided)
    start_positions: Optional[List[Tuple[float, float, float]]] = None
    end_positions: Optional[List[Tuple[float, float, float]]] = None
    
    # Swarm-specific parameters
    min_distance_threshold: float = 2.0 # Minimum distance to consider for inter-agent distance calculations and front collision avoidance
    max_swarm_spread_distance: float = 50.0 # Max Swarm Spread distance is defined as a multiplier from the initial mean distance between agents
    safety_penalty_weight: float = 0.7
    formation_bonus_weight: float = 1.0
    coordination_bonus_weight: float = 0.6
    max_safety_penalty: float = 1.2
    max_delta_mean_swarm_distance: float = 5.0
    img_width: int = 64
    img_height: int = 64

    # Return full swarm obs like a list of all agents' obs
    return_full_swarm_obs: bool = True
    
    # Reset leader configuration from reset_positions.json
    reset_leader: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get_random_leader_positions(cls, reset_leader: Dict[str, Any]) -> Tuple[float, float]:
        """Get random x_leader and y_leader from reset_leader configuration."""
        
        # Select a random area
        areas = list(reset_leader.keys())
        selected_area = random.choice(areas)
        #print(f"Selected area: {selected_area}")
        area_data = reset_leader[selected_area]

        # Select random x_leader and y_leader from the lists
        x_leader = random.choice(area_data['x_leader'])
        y_leader = random.choice(area_data['y_leader'])
        
        return x_leader, y_leader
    
    @classmethod
    def _generate_inverted_v_positions(cls, n_agents: int, y_leader: int, x_leader: int, y_offset: float = 0.0, max_swarm_spread_distance: float = 20.0):
        """Generate positions for inverted V formation based on y_leader and y_offset, with improved distance control."""
        positions = []
        z = -20.0  # Fixed z-coordinate
        leader_idx = n_agents // 2
        y_base = float(y_leader) + y_offset
        
        # Scale step size inversely with n_agents to prevent excessive spread (base step for n_agents=5)
        base_step = 4.0
        step = base_step * (5 / max(n_agents, 1))  # Scale down for larger n_agents
        
        # Scale random offset inversely with n_agents
        random_scale = 0.5 * (5 / max(n_agents, 1))
        
        # Leader position
        positions.append((x_leader, y_base, z))
        
        # Left drones (indices < leader_idx)
        for i in range(leader_idx):
            x_offset = -(leader_idx - i) * step + np.random.uniform(-random_scale, random_scale)
            y_offset_val = (leader_idx - i) * step + np.random.uniform(-random_scale, random_scale)
            x = x_leader + x_offset
            y = y_base + y_offset_val
            # Ensure constraints: x < x_leader, y > y_base
            x = min(x, x_leader - 0.1)
            y = max(y, y_base + 0.1)
            positions.insert(0, (x, y, z))  # Insert at beginning for correct order
        
        # Right drones (indices > leader_idx)
        for i in range(leader_idx + 1, n_agents):
            x_offset = (i - leader_idx) * step + np.random.uniform(-random_scale, random_scale)
            y_offset_val = (i - leader_idx) * step + np.random.uniform(-random_scale, random_scale)
            x = x_leader + x_offset
            y = y_base + y_offset_val
            # Ensure constraints: x > x_leader, y > y_base
            x = max(x, x_leader + 0.1)
            y = max(y, y_base + 0.1)
            positions.append((x, y, z))
        
        # Post-generation check: Ensure no two agents are too far apart
        cls._enforce_max_distance(positions, max_swarm_spread_distance)
        
        return positions
    
    @classmethod
    def _enforce_max_distance(cls, positions: List[Tuple[float, float, float]], max_distance: float):
        """Adjust positions to ensure no two agents exceed max_distance."""
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1 = np.array(positions[i][:2])  # Ignore z for 2D distance
                pos2 = np.array(positions[j][:2])
                distance = np.linalg.norm(pos1 - pos2)
                if distance > max_distance:
                    # Move closer: average positions with a small adjustment
                    midpoint = (pos1 + pos2) / 2
                    positions[i] = (midpoint[0], midpoint[1], positions[i][2])
                    positions[j] = (midpoint[0], midpoint[1], positions[j][2])
                    # Add small random perturbation to avoid exact overlap
                    positions[i] = (positions[i][0] + np.random.uniform(-0.5, 0.5), positions[i][1] + np.random.uniform(-0.5, 0.5), positions[i][2])
                    positions[j] = (positions[j][0] + np.random.uniform(-0.5, 0.5), positions[j][1] + np.random.uniform(-0.5, 0.5), positions[j][2])
    
    @classmethod
    def from_json(cls, json_path: str) -> 'SwarmConfig':
        """Create a SwarmConfig instance from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get n_agents from JSON (required)
        n_agents = data.get('n_agents')
        if n_agents is None:
            raise ValueError("n_agents must be specified in reset_positions.json")
        
        # Read start_positions and end_positions from JSON (required for modularity)
        start_positions = data.get('start_positions')
        end_positions = data.get('end_positions')
        reset_leader = data.get('reset_leader')
        
        if start_positions is None or end_positions is None:
            # Fallback: Generate positions if not in JSON
            if reset_leader is None:
                raise ValueError("reset_leader must be specified in reset_positions.json if positions are not provided")
            x_leader, y_leader = cls.get_random_leader_positions(reset_leader)
            start_positions = cls._generate_inverted_v_positions(n_agents, y_leader, x_leader, y_offset=0.0)
            end_positions = cls._generate_inverted_v_positions(n_agents, y_leader, x_leader, y_offset=-80.0)
        else:
            # Validate lengths match n_agents
            if len(start_positions) != n_agents or len(end_positions) != n_agents:
                raise ValueError(f"start_positions and end_positions must have length {n_agents} to match n_agents")
            # Convert to tuples
            start_positions = [tuple(pos) for pos in start_positions]
            end_positions = [tuple(pos) for pos in end_positions]
        
        return cls(
            n_agents=n_agents,
            start_positions=start_positions,
            end_positions=end_positions,
            reset_leader=reset_leader
        )
    
    ##Method deprecated in favor of from_json
    def create_uav_configs(self) -> List[UAVConfig]:
        """Generate a list of UAVConfig instances for each agent."""
        if self.start_positions is None:
            self.start_positions = [(i * 5.0, 0.0, -2.0) for i in range(self.n_agents)]
        if self.end_positions is None:
            self.end_positions = [(i * 5.0, 50.0, -2.0) for i in range(self.n_agents)]
        
        assert len(self.start_positions) == self.n_agents, "start_positions must match n_agents"
        assert len(self.end_positions) == self.n_agents, "end_positions must match n_agents"
        
        uav_configs = []
        for i in range(self.n_agents):
            uav_config = UAVConfig(
                drone_name=f"uav{i}",
                start_position=self.start_positions[i],
                end_position=self.end_positions[i],
                action_type=self.action_type,
                observation_img_size=self.observation_img_size,
                obstacle_threshold=self.obstacle_threshold,
                goal_threshold=self.goal_threshold,
                max_target_distance=self.max_target_distance,
                max_obstacle_distance=self.max_obstacle_distance,
                max_steps=self.max_steps  # Shared max_steps
            )
            uav_configs.append(uav_config)
        return uav_configs
    
    def generate_positions(self) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """
        Generate or return positions and dynamically update max_swarm_spread_distance.
        Random if reset_leader is set, otherwise fixed.
        """
        if self.reset_leader:
            # Generate random positions using reset_leader
            x_leader, y_leader = self.get_random_leader_positions(self.reset_leader)
            start_positions = self._generate_inverted_v_positions(self.n_agents, y_leader, x_leader, y_offset=0.0)
            
            # Generate end positions with -80 offset on y-axis
            end_positions = [(x, y - 80.0, z) for x, y, z in start_positions]
        else:
            # Return fixed positions
            if self.start_positions is None or self.end_positions is None:
                raise ValueError("start_positions and end_positions must be set if reset_leader is not provided")
            start_positions, end_positions = self.start_positions, self.end_positions

        # --- Dynamically update max_swarm_spread_distance ---
        # 1. Create a distance matrix from the initial positions
        pos_array = np.array(start_positions)
        # Use broadcasting to compute pairwise squared distances: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = pos_array[:, np.newaxis, :] - pos_array[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        mean_distance = 0
        # Get the upper triangle of the matrix (excluding the diagonal) to get unique pairwise distances
        if self.n_agents > 1:
            upper_triangle_indices = np.triu_indices(self.n_agents, k=1)
            pairwise_distances = dist_matrix[upper_triangle_indices]
            
            # 2. Set max_swarm_spread_distance to 2 * mean of pairwise distances
            mean_distance = pairwise_distances.mean()
            self.max_swarm_spread_distance = self.max_delta_mean_swarm_distance * mean_distance 
        else:
            # If there's only one agent, set a default value
            self.max_swarm_spread_distance = 1.0
        #print(f"Start positions: {start_positions}")
        #print(f"End positions: {end_positions}")
        #print(f"Mean distance: {mean_distance}")
        #print(f"Max formation distance: {self.max_swarm_spread_distance}")
        return start_positions, end_positions