import rclpy 
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from example_interfaces.srv import Trigger
from ros2_mnist.weights_subscriber import ModelSubscriber

import os
import numpy as np
import tensorflow as tf
import threading

class FederatedServer(Node):
    def __init__(self, clients):
        super().__init__('model_publisher')

        # Initial model
        self.model_config = self.build_model()
        
        # List to store the mean of the weights
        self.weights_mean = []

        # List to store the addition of the clients weights
        self.global_weights = []

        # Number of clients
        self.clients = clients

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
        
        config = model.to_json()
        
        return config

    def callback_send_model(self, request, response):
        response.success = True
        response.message = str(self.model_config)

        return response


def main(args=None):
    rclpy.init(args=args)
    
    clients = 1
    node = FederatedServer(clients)
    client_subscriber = ModelSubscriber("client_A")

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(client_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = node.create_rate(2)

    try:
        while rclpy.ok():
            rate.sleep()
            
            if client_subscriber.get_client_weights() != None:
                continue


    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()

if __name__ == '__main__':
    main()