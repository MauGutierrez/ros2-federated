import rclpy 
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights
from ros2_mnist.helper_functions import *

import os
import sys
import numpy as np
import tensorflow as tf
import json

class FederatedServer(Node):
    def __init__(self, n_clients):
        super().__init__('federated_server')
        
        # Initial model
        self.model_config = self.build_model()
        
        # List to store the mean of the weights
        self.fl_weights = None
        
        # Number of clients being executed
        self.n_clients = n_clients

        # Service to send the model
        self.build_service_ = self.create_service(
            Trigger, "get_model", self.callback_send_model
        )

        # Service to get the average of the weights
        self.update_weights_service_ = self.create_service(
            SendLocalWeights, "update_weights", self.callback_update_weights
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
    
    def get_average_of_weights(self, request):
        
        weights = request["weights"]
        weights = deserialize_array(weights)

        if self.fl_weights == None:
            self.fl_weights = np.copy(weights)
        else:
            self.fl_weights = self.fl_weights + weights

        new_weights = self.fl_weights / self.n_clients

        new_weights = serialize_array(new_weights)

        message = {
            "weights": new_weights
        }

        response = json.dumps(message)

        return response

    def callback_send_model(self, request, response):
        if self.model_config != None:
            response.success = True
            response.message = str(self.model_config)
        
        else:
            response.success = False
            response.message = "Model not ready yet"

        return response
    
    def callback_update_weights(self, request, response):
        request_message = json.loads(request.message_request)

        if request_message["weights"] != None:
            data = self.get_average_of_weights(request_message) 
            if data != None:
                response.success = True
                response.message_response = data
            else:
                response.success = False
                response.message_response = "Couldn't calculate average of weights"
        else:
            response.success = False
            response.message_response = "Wrong request"
        
        return response

def main(args=None):
    rclpy.init(args=args)
    
    n_clients = 1
    node = FederatedServer(n_clients)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()