from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class UAVConfig:
    """Configuration class for UAV parameters, organized by category."""
    
    # AirSim-related
    drone_name: str
    
    # Position-related
    start_position: Tuple[float, float, float]
    end_position: Tuple[float, float, float]
    
    # Action-related
    action_type: str = "discrete"
    
    # Observation-related
    observation_img_size: Tuple[int, int] = (64, 64)
    max_target_distance: Optional[float] = None
    max_obstacle_distance: float = 50.0
    
    # Reward/Threshold-related
    obstacle_threshold: float = 2.0
    goal_threshold: float = 1.0
    
    # Episode-related
    max_steps: int = 300