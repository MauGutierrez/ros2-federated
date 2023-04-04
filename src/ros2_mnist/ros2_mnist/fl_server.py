import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from example_interfaces.srv import Trigger

from cv_bridge import CvBridge
import os
import cv2
import numpy as np
import tensorflow as tf

class ModelPublisher(Node):
    def __init__(self):
        super().__init__('model_publisher')
        
        # Flag to init the download of the model
        self.model_ready = False

        # Initial model
        self.model = self.build_model()

        # Service to send the model
        self.service_ = self.create_service(
            Trigger, "get_model", self.callback_send_model
        )

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.save("my_model.h5")

        self.model_ready = True

        return model

    def callback_send_model(self, request, response):
        response.success = True

        if self.model_ready:
            response.message = "Init download"
        else:
            response.message = "Model not ready yet"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ModelPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()