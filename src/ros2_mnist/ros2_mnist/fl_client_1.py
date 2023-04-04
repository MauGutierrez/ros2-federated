import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from example_interfaces.srv import Trigger

from cv_bridge import CvBridge
import os
import cv2
import numpy as np
import tensorflow as tf

class FederatedClientA(Node):
    def __init__(self):
        super().__init__('federated_client_a')
        
        self.model = None

        self.cli = self.create_client(Trigger, "get_model")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def build_model(self):
        self.model = tf.keras.models.load_model("my_model.h5")
        self.model.summary()

def main():
    rclpy.init()

    client = FederatedClientA()
    response = client.send_request()
    
    if response.message == "Init download":
        client.build_model()

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()