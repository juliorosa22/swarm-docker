import numpy as np
import sys
sys.path.append('C:\\Users\\julio\\OneDrive\\Documents\\Programming\\Drones\\Airsim\\PythonClient')

import airsim
from airsim import MultirotorClient
import time
import pprint
import math

def ensure_native_types(value):
    """
    Convert numpy types to native Python types to avoid msgpack serialization errors
    """
    if isinstance(value, (np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(value)
    elif isinstance(value, (np.float16, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return [ensure_native_types(x) for x in value]
    elif isinstance(value, (list, tuple)):
        return [ensure_native_types(x) for x in value]
    elif isinstance(value, dict):
        return {k: ensure_native_types(v) for k, v in value.items()}
    else:
        return value

def compute_direction_vector(current_pos, target_pos):
    """
    Compute the normalized direction vector from current position to target position
    """
    direction = {
        'x': target_pos['x'] - current_pos['x'],
        'y': target_pos['y'] - current_pos['y'],
        'z': target_pos['z'] - current_pos['z']
    }
    
    # Calculate magnitude of the vector
    magnitude = math.sqrt(direction['x']**2 + direction['y']**2 + direction['z']**2)
    
    # Normalize the vector (avoid division by zero)
    if magnitude > 0.001:
        direction['x'] /= magnitude
        direction['y'] /= magnitude
        direction['z'] /= magnitude
    
    return direction

def compute_distance(pos1, pos2):
    """
    Compute the Euclidean distance between two positions
    """
    dx = pos1['x'] - pos2['x']
    dy = pos1['y'] - pos2['y']
    dz = pos1['z'] - pos2['z']
    
    return math.sqrt(dx**2 + dy**2 + dz**2)

def get_lidar_data(swarm, uav):
    """
    Get LiDAR readings and convert to numpy array of points
    """
    lidar_data = swarm.getLidarData(vehicle_name=uav)
    
    # Check if we have valid points
    if len(lidar_data.point_cloud) < 3:
        return np.array([])
    
    # Convert point cloud to a list of points
    points = []
    point_cloud = lidar_data.point_cloud
    for i in range(0, len(point_cloud), 3):
        if i + 2 < len(point_cloud):
            points.append([point_cloud[i], point_cloud[i+1], point_cloud[i+2]])
    
    if not points:
        return np.array([])
    
    # Convert to numpy array
    return np.array(points, dtype=np.float32)

def compute_potential_field_direction(current_pos, goal_pos, lidar_points, 
                                      attractive_weight=1.0, repulsive_weight=2.0, 
                                      influence_radius=5.0, min_distance=1.0):
    """
    Compute direction vector based on potential field approach
    
    Args:
        current_pos: Current position of the UAV
        goal_pos: Goal position of the UAV
        lidar_points: Numpy array of lidar points
        attractive_weight: Weight for the attractive force (goal)
        repulsive_weight: Weight for the repulsive force (obstacles)
        influence_radius: Maximum distance at which obstacles have influence
        min_distance: Minimum distance to obstacles (high repulsion inside this radius)
        
    Returns:
        direction: Normalized direction vector to follow
    """
    # Compute attractive force (points toward the goal)
    attractive_force = compute_direction_vector(current_pos, goal_pos)
    
    # Scale attractive force by weight
    attractive_force = {
        'x': attractive_force['x'] * attractive_weight,
        'y': attractive_force['y'] * attractive_weight,
        'z': attractive_force['z'] * attractive_weight
    }
    
    # Initialize repulsive force
    repulsive_force = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    # If no lidar points, return only the attractive force
    if len(lidar_points) == 0:
        # Normalize attractive force
        magnitude = math.sqrt(attractive_force['x']**2 + attractive_force['y']**2 + attractive_force['z']**2)
        if magnitude > 0.001:
            attractive_force['x'] /= magnitude
            attractive_force['y'] /= magnitude
            attractive_force['z'] /= magnitude
        return attractive_force
    
    # Compute repulsive force from each obstacle point
    total_weight = 0.0
    
    for point in lidar_points:
        # Calculate distance to the point
        dist = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        
        # Skip points outside influence radius
        if dist > influence_radius:
            continue
            
        # Direction from obstacle to UAV (unit vector)
        if dist < 0.001:
            # If point is very close, push away in an arbitrary direction
            direction = {'x': 0.0, 'y': 0.0, 'z': 1.0}
        else:
            direction = {
                'x': -point[0] / dist,  # Negative because we want to move away from the obstacle
                'y': -point[1] / dist,
                'z': -point[2] / dist
            }
        
        # Calculate repulsive weight based on distance
        # Closer obstacles have greater influence
        if dist < min_distance:
            # Very high weight for very close obstacles
            weight = repulsive_weight * (2.0 - dist/min_distance)
        else:
            # Weight decreases with square of distance
            weight = repulsive_weight * (influence_radius - dist)**2 / (influence_radius - min_distance)**2
        
        # Accumulate weighted repulsive force
        repulsive_force['x'] += direction['x'] * weight
        repulsive_force['y'] += direction['y'] * weight
        repulsive_force['z'] += direction['z'] * weight
        
        total_weight += weight
    
    # Normalize repulsive force if there were any points within influence radius
    if total_weight > 0.001:
        repulsive_force['x'] /= total_weight
        repulsive_force['y'] /= total_weight
        repulsive_force['z'] /= total_weight
    
    # Combine attractive and repulsive forces
    resultant_force = {
        'x': attractive_force['x'] + repulsive_force['x'],
        'y': attractive_force['y'] + repulsive_force['y'],
        'z': attractive_force['z'] + repulsive_force['z']
    }
    
    # Normalize the resultant force
    magnitude = math.sqrt(resultant_force['x']**2 + resultant_force['y']**2 + resultant_force['z']**2)
    if magnitude > 0.001:
        resultant_force['x'] /= magnitude
        resultant_force['y'] /= magnitude
        resultant_force['z'] /= magnitude
    else:
        # If resultant force is zero (rare case), use pure attractive force
        resultant_force = attractive_force
    
    # At the end of the function, before returning the resultant_force:
# Ensure all values are native Python types
    resultant_force = {
        'x': ensure_native_types(resultant_force['x']),
        'y': ensure_native_types(resultant_force['y']),
        'z': ensure_native_types(resultant_force['z'])
    }

    return resultant_force

def move_swarm_by_velocity_with_potential_field(swarm_ids, swarm, end_positions, 
                                               speed=5.0, distance_threshold=2.0, 
                                               update_interval=0.5, influence_radius=5.0,
                                               min_distance=1.0):
    """
    Move swarm to end positions using velocity commands with potential field obstacle avoidance
    
    Args:
        swarm_ids: List of UAV IDs
        swarm: AirSim client object
        end_positions: Dictionary of target positions for each UAV
        speed: Velocity magnitude in m/s
        distance_threshold: Distance in meters to consider target reached
        update_interval: Time in seconds between velocity updates
        influence_radius: Maximum distance at which obstacles have influence
        min_distance: Minimum distance to obstacles (high repulsion inside this radius)
    """
    all_reached = False
    reached_targets = {uav: False for uav in swarm_ids}
    
    # Parameters for potential field
    attractive_weight = 1.0
    repulsive_weight = 2.0
    
    print("Moving swarm to end positions using potential field approach...")
    
    while not all_reached:
        joins = []
        
        for uav in swarm_ids:
            if reached_targets[uav]:
                continue
                
            # Get current position of the UAV
            state = swarm.getMultirotorState(vehicle_name=uav)
            current_pos = {
                'x': state.kinematics_estimated.position.x_val,
                'y': state.kinematics_estimated.position.y_val,
                'z': state.kinematics_estimated.position.z_val
            }
            
            # Compute distance to target
            distance = compute_distance(current_pos, end_positions[uav])
            
            # Check if the UAV reached its target
            if distance <= distance_threshold:
                print(f"{uav} reached destination within {distance:.2f}m")
                reached_targets[uav] = True
                # Hover in place once reached
                joins.append(swarm.hoverAsync(vehicle_name=uav))
                continue
            
            # Get LiDAR data
            try:
                lidar_points = get_lidar_data(swarm, uav)
                num_points = len(lidar_points)
                
                # Compute direction using potential field approach
                direction = compute_potential_field_direction(
                    current_pos, 
                    end_positions[uav], 
                    lidar_points,
                    attractive_weight=attractive_weight,
                    repulsive_weight=repulsive_weight,
                    influence_radius=influence_radius,
                    min_distance=min_distance
                )
                
                # Log information about obstacles if there are points
                if num_points > 0:
                    min_dist = np.min(np.sqrt(np.sum(lidar_points**2, axis=1)))
                    print(f"{uav} - Detected {num_points} points, closest at {min_dist:.2f}m")
                
            except Exception as e:
                print(f"Error processing LiDAR data for {uav}: {e}")
                # Fallback to direct path if LiDAR processing fails
                direction = compute_direction_vector(current_pos, end_positions[uav])
            
            # Scale direction by desired speed
            vx = direction['x'] * speed
            vy = direction['y'] * speed
            vz = direction['z'] * speed
            
            # Dynamic speed reduction when close to obstacles
            if 'lidar_points' in locals() and len(lidar_points) > 0:
                min_dist = np.min(np.sqrt(np.sum(lidar_points**2, axis=1)))
                if min_dist < influence_radius:
                    # Reduce speed when near obstacles (proportional to proximity)
                    speed_factor = max(0.3, min(1.0, min_dist / influence_radius))
                    vx *= speed_factor
                    vy *= speed_factor
                    vz *= speed_factor
                    print(f"{uav} - Reduced speed to {speed_factor*100:.0f}% due to obstacles")
            vx = ensure_native_types(vx)
            vy = ensure_native_types(vy)
            vz = ensure_native_types(vz)
            # Move UAV using velocity commands (duration = update_interval)
            joins.append(swarm.moveByVelocityAsync(vx, vy, vz, update_interval, vehicle_name=uav))
            
            print(f"{uav} - Distance to target: {distance:.2f}m, Direction: ({direction['x']:.2f}, {direction['y']:.2f}, {direction['z']:.2f})")
        
        # Wait for all movement commands to complete
        for f in joins:
            f.join()
            
        # Check if all UAVs have reached their targets
        all_reached = all(reached_targets.values())
        
        # Small delay before the next update
        time.sleep(0.1)
    
    print("All UAVs have reached their destinations!")

start_positions = {
        'uav0': {'x': -3.5, 'y': 3, 'z': -10},
        'uav1': {'x': -2, 'y': 1.5, 'z': -10},
        'uav2': {'x': 0, 'y': 0, 'z': -10},  # leader
        'uav3': {'x': 2, 'y': 1.5, 'z': -10},
        'uav4': {'x': 3.5, 'y': 3, 'z': -10},
    }
end_positions = {
        'uav0': {'x': -3.5, 'y': 3-210, 'z': -10},
        'uav1': {'x': -2, 'y': 1.5-210, 'z': -10},
        'uav2': {'x': 0, 'y': 0-210, 'z': -10},  # leader
        'uav3': {'x': 2, 'y': 1.5-210, 'z': -10},
        'uav4': {'x': 3.5, 'y': 3-210, 'z': -10},
    }

if __name__ == "__main__":
    swarm_ids = [f"uav{i}" for i in range(5)]
    swarm = airsim.MultirotorClient(ip='127.0.0.1')
    swarm.confirmConnection()
    
    # Enable API control and arm all drones
    for uav in swarm_ids:
        swarm.enableApiControl(True, vehicle_name=uav)
        swarm.armDisarm(True, vehicle_name=uav)
        
        # Check LiDAR for each drone
        try:
            lidar_data = swarm.getLidarData(vehicle_name=uav)
            print(f"LiDAR for {uav}: {len(lidar_data.point_cloud)/3:.0f} points")
        except Exception as e:
            print(f"Error checking LiDAR for {uav}: {e}")

    print('Waiting for the swarm to takeoff')    
    time.sleep(2)
    joins = []
 
    # Take off all drones
    for uav in swarm_ids:
        joins.append(swarm.takeoffAsync(vehicle_name=uav))
 
    for f in joins:
        f.join()

    # Get and print drone states
    states = []
    for uav in swarm_ids:
        state = swarm.getMultirotorState(vehicle_name=uav)
        states.append(pprint.pformat(state))
    
    for i in range(len(swarm_ids)):
        print(f"{swarm_ids[i]} State: {states[i]}")
    
    # Move to initial positions first (optional)
    print("Moving swarm to start positions")
    joins = []
    for uav in swarm_ids:
        pos = start_positions[uav]
        joins.append(swarm.moveToPositionAsync(pos['x'], pos['y'], pos['z'], 5, vehicle_name=uav))
    
    for f in joins:
        f.join()
    
    # Wait a moment to stabilize
    time.sleep(2)
    
    # Now move to end positions using velocity commands with potential field obstacle avoidance
    move_swarm_by_velocity_with_potential_field(
        swarm_ids,
        swarm, 
        end_positions, 
        speed=5.0, 
        distance_threshold=2.0,
        influence_radius=5.0,
        min_distance=1.0
    )
    
    # Optionally land the drones after reaching end positions
    print("Landing the swarm")
    joins = []
    for uav in swarm_ids:
        joins.append(swarm.landAsync(vehicle_name=uav))
    
    for f in joins:
        f.join()
    
    # Disarm and release control
    for uav in swarm_ids:
        swarm.armDisarm(False, vehicle_name=uav)
        swarm.enableApiControl(False, vehicle_name=uav)
    
    print("Mission completed")