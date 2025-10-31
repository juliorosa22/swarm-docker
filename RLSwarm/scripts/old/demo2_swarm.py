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
                                     attractive_weight=1.0, repulsive_weight=5.0,
                                     influence_radius=5.0, min_distance=1.5,
                                     use_stronger_repulsion=True):
    """
    Compute direction vector based on potential field approach with enhanced obstacle avoidance
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
    
    # Find the closest point
    distances = np.sqrt(np.sum(lidar_points**2, axis=1))
    min_dist_idx = np.argmin(distances)
    min_dist = distances[min_dist_idx]
    closest_point = lidar_points[min_dist_idx]
    
    # Check if we're very close to an obstacle (emergency avoidance)
    emergency_avoidance = False
    if min_dist < min_distance * 0.7:  # If we're dangerously close
        emergency_avoidance = True
        print(f"*** EMERGENCY AVOIDANCE - Obstacle at {min_dist:.2f}m ***")
    
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
            if use_stronger_repulsion:
                # More aggressive inverse square repulsion
                weight = repulsive_weight * (min_distance / max(0.1, dist))**2
            else:
                # Linear repulsion
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
    
    # In emergency avoidance, strongly prioritize repulsion over attraction
    if emergency_avoidance:
        # Ignore the goal direction when too close to obstacles
        resultant_force = {
            'x': repulsive_force['x'],
            'y': repulsive_force['y'],
            'z': repulsive_force['z']
        }
        
        # Add slight upward bias for safer avoidance
        resultant_force['z'] += 0.3  # Tendency to go up
    else:
        # Normal case: combine attractive and repulsive forces
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
        # If resultant force is zero (rare case), use pure repulsive force
        resultant_force = repulsive_force
    
    # Convert to native types to avoid msgpack errors
    resultant_force['x'] = float(resultant_force['x'])
    resultant_force['y'] = float(resultant_force['y'])
    resultant_force['z'] = float(resultant_force['z'])
    
    return resultant_force

def move_along_safe_path(swarm, uav_name, current_pos, goal_pos, lidar_points, speed=5.0, min_distance=1.5):
    """
    Move UAV using a smooth path with intermediate waypoints for obstacle avoidance
    """
    # Generate intermediate waypoints for smooth navigation
    waypoints = []
    
    # Get direction using potential field
    direction = compute_potential_field_direction(
        current_pos, goal_pos, lidar_points,
        repulsive_weight=5.0,
        influence_radius=5.0,
        min_distance=min_distance
    )
    
    # Current position as first waypoint
    waypoints.append(airsim.Vector3r(
        float(current_pos['x']), 
        float(current_pos['y']), 
        float(current_pos['z'])
    ))
    
    # Add intermediate waypoint in the safe direction
    intermediate_dist = 5.0  # 5 meter intermediate waypoint
    intermediate_point = airsim.Vector3r(
        float(current_pos['x'] + direction['x'] * intermediate_dist),
        float(current_pos['y'] + direction['y'] * intermediate_dist),
        float(current_pos['z'] + direction['z'] * intermediate_dist)
    )
    waypoints.append(intermediate_point)
    
    # Add final waypoint if not too close
    dist_to_goal = compute_distance(current_pos, goal_pos)
    if dist_to_goal > 10.0:  # Only add goal as waypoint if it's far enough
        waypoints.append(airsim.Vector3r(
            float(goal_pos['x']), 
            float(goal_pos['y']), 
            float(goal_pos['z'])
        ))
    
    # Convert to Vector3r path
    path = [airsim.Vector3r(float(p.x_val), float(p.y_val), float(p.z_val)) for p in waypoints]
    
    # Execute path following
    return swarm.moveOnPathAsync(
        path,
        float(speed),
        120,  # timeout in seconds
        airsim.DrivetrainType.ForwardOnly,
        airsim.YawMode(False, 0),
        1.0,  # lookahead distance
        1,    # adaptive lookahead
        vehicle_name=uav_name
    )

def move_to_safe_position(swarm, uav_name, current_pos, goal_pos, lidar_points, speed=5.0, min_distance=1.5):
    """
    Move UAV to a safe intermediate position based on potential field
    """
    # Compute safe direction using potential field
    direction = compute_potential_field_direction(
        current_pos, goal_pos, lidar_points,
        repulsive_weight=5.0,
        influence_radius=5.0,
        min_distance=min_distance
    )
    
    # Determine how far to move (adaptive based on obstacles)
    closest_obstacle_dist = float('inf')
    if len(lidar_points) > 0:
        closest_obstacle_dist = float(np.min(np.sqrt(np.sum(lidar_points**2, axis=1))))
    
    # Choose move distance (shorter when obstacles are near)
    move_distance = min(10.0, max(3.0, closest_obstacle_dist * 1.5))
    
    # Compute target position
    target_x = float(current_pos['x'] + direction['x'] * move_distance)
    target_y = float(current_pos['y'] + direction['y'] * move_distance)
    target_z = float(current_pos['z'] + direction['z'] * move_distance)
    
    # Move with smooth parameters
    return swarm.moveToPositionAsync(
        target_x, target_y, target_z,
        speed,
        60,  # timeout in seconds
        airsim.DrivetrainType.MaxDegreeOfFreedom,  # Allows more freedom of movement
        airsim.YawMode(False, 0),  # Don't maintain heading to goal
        -1,  # -1 means look at the goal point
        1,   # Accept looser final position tolerance
        vehicle_name=uav_name
    )

def move_at_constant_height(swarm, uav_name, current_pos, goal_pos, lidar_points, speed=5.0, min_distance=1.5):
    """
    Move UAV with velocity commands but maintain a constant height
    """
    # Compute safe direction using potential field (only considering x and y)
    direction = compute_potential_field_direction(
        current_pos, goal_pos, lidar_points,
        repulsive_weight=5.0,
        influence_radius=5.0,
        min_distance=min_distance
    )
    
    # Calculate velocity components (only x and y)
    vx = float(direction['x'] * speed)
    vy = float(direction['y'] * speed)
    
    # Use current height as target height (or adjust as needed)
    target_z = float(current_pos['z'])
    
    # Move with constant height
    return swarm.moveByVelocityZAsync(
        vx, vy, target_z,
        1.0,  # duration in seconds
        airsim.DrivetrainType.MaxDegreeOfFreedom,
        airsim.YawMode(False, 0),  
        vehicle_name=uav_name
    )

def move_swarm_with_smooth_navigation(swarm_ids, swarm, end_positions, 
                                     speed=5.0, distance_threshold=2.0,
                                     influence_radius=5.0, min_distance=1.5):
    """
    Move swarm to end positions using smoother navigation methods with potential field obstacle avoidance
    """
    all_reached = False
    reached_targets = {uav: False for uav in swarm_ids}
    
    # Track collision states
    collision_states = {uav: {'colliding': False, 'recovery_steps': 0} for uav in swarm_ids}
    max_recovery_steps = 10
    
    print("Moving swarm to end positions using smooth navigation...")
    
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
            
            # Check for collisions
            collision_info = swarm.simGetCollisionInfo(vehicle_name=uav)
            is_colliding = collision_info.has_collided
            
            # Update collision state
            if is_colliding:
                if not collision_states[uav]['colliding']:
                    print(f"*** {uav} COLLISION DETECTED - Starting recovery ***")
                    collision_states[uav]['colliding'] = True
                    collision_states[uav]['recovery_steps'] = 0
            else:
                # Only reset collision state after recovery is complete
                if collision_states[uav]['recovery_steps'] >= max_recovery_steps:
                    collision_states[uav]['colliding'] = False
            
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
                
                # Special behavior when recovering from collision
                in_recovery = collision_states[uav]['colliding'] and collision_states[uav]['recovery_steps'] < max_recovery_steps
                
                if in_recovery:
                    # During recovery, perform a vertical escape maneuver
                    # Move straight up first to get away from obstacles
                    recovery_height = current_pos['z'] - 3.0  # Move up 3 meters (remember z is negative in NED)
                    joins.append(swarm.moveToPositionAsync(
                        float(current_pos['x']), float(current_pos['y']), float(recovery_height),
                        2.0,  # Slow, careful movement
                        60,
                        airsim.DrivetrainType.MaxDegreeOfFreedom,
                        airsim.YawMode(False, 0),
                        vehicle_name=uav
                    ))
                    collision_states[uav]['recovery_steps'] += 1
                    print(f"{uav} - Recovery step {collision_states[uav]['recovery_steps']}/{max_recovery_steps}")
                elif distance > 20.0:
                    # For long distances, use moveOnPathAsync for smoother motion
                    joins.append(move_along_safe_path(
                        swarm, uav, current_pos, end_positions[uav], lidar_points, 
                        speed=speed, min_distance=min_distance
                    ))
                elif num_points > 0 and np.min(np.sqrt(np.sum(lidar_points**2, axis=1))) < min_distance:
                    # If obstacles are nearby, use more careful positioning
                    joins.append(move_to_safe_position(
                        swarm, uav, current_pos, end_positions[uav], lidar_points,
                        speed=speed * 0.7,  # Reduced speed for careful navigation
                        min_distance=min_distance
                    ))
                else:
                    # For normal movement without close obstacles, use moveByVelocityZAsync
                    # This maintains height while moving smoothly
                    direction = compute_potential_field_direction(
                        current_pos, end_positions[uav], lidar_points,
                        attractive_weight=1.0,
                        repulsive_weight=3.0,
                        influence_radius=influence_radius,
                        min_distance=min_distance
                    )
                    
                    # Calculate velocity but maintain target height
                    vx = float(direction['x'] * speed)
                    vy = float(direction['y'] * speed)
                    target_z = float(end_positions[uav]['z'])  # Maintain goal height
                    
                    joins.append(swarm.moveByVelocityZAsync(
                        vx, vy, target_z,
                        1.0,  # duration in seconds 
                        airsim.DrivetrainType.MaxDegreeOfFreedom,
                        airsim.YawMode(False, 0),
                        vehicle_name=uav
                    ))
                
                # Log information about obstacles if there are points
                if num_points > 0:
                    min_dist = float(np.min(np.sqrt(np.sum(lidar_points**2, axis=1))))
                    print(f"{uav} - Detected {num_points} points, closest at {min_dist:.2f}m")
                
            except Exception as e:
                print(f"Error during navigation for {uav}: {e}")
                # Fallback to simple position-based movement
                joins.append(swarm.moveToPositionAsync(
                    float(end_positions[uav]['x']),
                    float(end_positions[uav]['y']),
                    float(end_positions[uav]['z']),
                    speed,
                    60,
                    vehicle_name=uav
                ))
            
            print(f"{uav} - Distance to target: {distance:.2f}m")
        
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
    
    # Now move to end positions using velocity commands with smooth navigation
    move_swarm_with_smooth_navigation(
        swarm_ids,
        swarm, 
        end_positions, 
        speed=5.0, 
        distance_threshold=2.0,
        influence_radius=5.0,
        min_distance=1.5
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