import os
import cv2
import json
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String

#___Global Variables:
TOPIC = 'weights_'
QUEUE_SIZE = 1
PERIOD = 0.5  # seconds

#__Classes:
class ModelSubscriber(Node):    
    def __init__(self, client=None, topic=TOPIC, queue=QUEUE_SIZE, period=PERIOD):
        super().__init__('weights_subscriber')

        # Variable to store the model weights
        self.weights_ = None
        
        self.client = client
        topic = topic + self.client
        # initialize subscriber
        self.subscription_ = self.create_subscription(
            String,
            topic,
            self.listener_callback,
            queue
        )
        self.subscription_
    
    def listener_callback(self, msg):
        self.get_logger().info("I heard " + self.client)
        if (msg.data != None) or (msg.data != ""):
            self.weights_ = msg.data

    def get_client_weights(self):
        return np.array(self.weights_)

if __name__ == '__main__':
    main()