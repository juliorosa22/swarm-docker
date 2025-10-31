"""
Simple script to connect to 5 UAVs in AirSim and send each to its end position.
This script demonstrates basic UAV control without using the full environment.
"""

import sys
import os
import time

# Add AirSim PythonClient to path (adjust if needed)
sys.path.append('C:\\Users\\julio\\OneDrive\\Documents\\Programming\\Drones\\Airsim\\PythonClient')

import airsim

def main():
    """Main function to control the UAV swarm."""
    
    print("=== UAV Swarm Control Script ===")
    
    # Define positions
    start_positions = [
        (-3.5, 3, -10),    # uav0
        (-2, 1.5, -10),    # uav1
        (0, 0, -10),       # uav2 (leader)
        (2, 1.5, -10),     # uav3
        (3.5, 3, -10)      # uav4
    ]
    
    end_positions = [
        (-3.5, 3-210, -10),    # uav0: (-3.5, -207, -10)
        (-2, 1.5-210, -10),    # uav1: (-2, -208.5, -10)
        (0, 0-210, -10),       # uav2: (0, -210, -10)
        (2, 1.5-210, -10),     # uav3: (2, -208.5, -10)
        (3.5, 3-210, -10)      # uav4: (3.5, -207, -10)
    ]
    
    # UAV names
    uav_names = ['uav0', 'uav1', 'uav2', 'uav3', 'uav4']
    
    try:
        # Connect to AirSim
        print("Connecting to AirSim...")
        client = airsim.MultirotorClient(ip='127.0.0.1')
        client.confirmConnection()
        print("✅ Connected to AirSim")
        
        # Enable API control for all UAVs
        print("\nEnabling API control for all UAVs...")
        for uav_name in uav_names:
            client.enableApiControl(True, uav_name)
            client.armDisarm(True, uav_name)
            print(f"✅ {uav_name} ready")
        
        # Take off all UAVs
        print("\nTaking off all UAVs...")
        for uav_name in uav_names:
            client.takeoffAsync(vehicle_name=uav_name).join()
            print(f"✅ {uav_name} took off")
        
        # Wait a moment after takeoff
        time.sleep(2)
        
        # Move each UAV to its end position
        print("\nMoving UAVs to end positions...")
        move_tasks = []
        
        for i, uav_name in enumerate(uav_names):
            x, y, z = end_positions[i]
            print(f"📍 Sending {uav_name} to position ({x}, {y}, {z})")
            
            # Send move command
            task = client.moveToPositionAsync(
                float(x), 
                float(y), 
                float(z), 
                5,  # velocity
                vehicle_name=uav_name
            )
            move_tasks.append((uav_name, task))
        
        # Wait for all movements to complete
        print("\nWaiting for movements to complete...")
        for uav_name, task in move_tasks:
            task.join()
            print(f"✅ {uav_name} reached destination")
        
        # Get final positions to verify
        print("\nFinal positions:")
        for uav_name in uav_names:
            pose = client.simGetVehiclePose(vehicle_name=uav_name)
            position = pose.position
            print(f"{uav_name}: ({position.x_val:.2f}, {position.y_val:.2f}, {position.z_val:.2f})")
        
        print("\n=== All UAVs successfully moved to end positions ===")
        
        # Optional: Land all UAVs
        print("\nLanding all UAVs...")
        for uav_name in uav_names:
            client.landAsync(vehicle_name=uav_name).join()
            print(f"✅ {uav_name} landed")
        
        # Disable API control
        for uav_name in uav_names:
            client.armDisarm(False, uav_name)
            client.enableApiControl(False, uav_name)
        
        print("✅ All UAVs landed and API control disabled")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()