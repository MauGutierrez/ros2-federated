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
class ModelPublisher(Node):    
    def __init__(self, client=None, weights=None, topic=TOPIC, queue=QUEUE_SIZE, period=PERIOD):
        super().__init__('weights_publisher')
        
        # initialize publisher
        self.client = client
        topic = topic + self.client
        self.weights = weights
        self.publisher_ = self.create_publisher(String, topic, queue)
        timer_period = period
        self.timer = self.create_timer(timer_period, self.timer_callback)

        
    def timer_callback(self):
        msg = String()
        msg.data = self.weights
        self.publisher_.publish(msg)
        self.get_logger().info('Weights ' + self.client + ' ready')


if __name__ == '__main__':
    main()