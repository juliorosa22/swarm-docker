import numpy as np
import time
import airsim

class AirSimController:
    """
    Controller for interfacing with AirSim simulator.
    """
    
    def __init__(self, client_address=""):
        """
        Initialize the AirSim controller.
        
        Args:
            client_address (str, optional): AirSim client address. Defaults to "".
        """
        # Connect to AirSim
        self.client = airsim.MultirotorClient(client_address)
        self.connect()
        
    def connect(self):
        """
        Connect to AirSim client and initialize.
        """
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Set up initial flight mode
        self.client.takeoffAsync().join()
    
    def reset(self):
        """
        Reset the UAV state.
        """
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.5)  # Give some time for the simulator to reset
    
    def get_state(self):
        """
        Get the current state of the UAV.
        
        Returns:
            dict: UAV state information.
        """
        multirotor_state = self.client.getMultirotorState()
        
        state = {
            'position': multirotor_state.kinematics_estimated.position,
            'orientation': multirotor_state.kinematics_estimated.orientation,
            'linear_velocity': multirotor_state.kinematics_estimated.linear_velocity,
            'angular_velocity': multirotor_state.kinematics_estimated.angular_velocity,
            'timestamp': multirotor_state.timestamp
        }
        
        return state
    
    def execute_action(self, action):
        """
        Execute the given action in AirSim.
        
        Args:
            action (np.ndarray): [roll, pitch, yaw_rate, throttle] in range [-1, 1]
        """
        # Scale the action values to the proper ranges
        roll = float(action[0])        # -1 to 1
        pitch = float(action[1])       # -1 to 1
        yaw_rate = float(action[2])    # -1 to 1
        throttle = float(action[3])    # 0 to 1
        
        # Send commands to the UAV
        self.client.moveByRollPitchYawrateThrottleAsync(
            roll, pitch, yaw_rate, throttle, 0.05  # 50ms duration
        ).join()
    
    def get_image(self):
        """
        Get an RGB image from the AirSim camera.
        
        Returns:
            np.ndarray: RGB image array.
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene)
        ])
        
        if responses:
            img_response = responses[0]
            img1d = np.fromstring(img_response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(img_response.height, img_response.width, 3)
            return img_rgb
        
        return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def close(self):
        """
        Close the connection and release resources.
        """
        self.client.armDisarm(False)
        self.client.enableApiControl(False)