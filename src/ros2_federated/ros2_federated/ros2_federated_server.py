import rclpy 
from rclpy.node import Node
from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights
from ros2_federated.helper_functions import *
from ros2_federated.keras_models import get_model_from_json

import os
import sys
import numpy as np
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class FederatedServer(Node):

    _clients_counter = 0
    
    def __init__(self, n_clients, selected_model):
        super().__init__('federated_server')
        
        # Initial model
        self.model_config = self.get_model_config(selected_model)
        
        # List to store the mean of the weights
        self.fl_weights = None

        # Number of clients being executed
        self.n_clients = n_clients

        # Service to send the model
        self.build_service_ = self.create_service(
            Trigger, "download_model", self.callback_send_model
        )

        # Service to get the average of the weights
        self.update_weights_service_ = self.create_service(
            SendLocalWeights, "add_weights", self.callback_add_weights
        )

        # Service to send the average of the weights
        self.send_weights_service_ = self.create_service(
            SendLocalWeights, "get_weights", self.callback_send_weights
        )

    def get_model_config(self, selected_model: str) -> object:
        config = get_model_from_json(selected_model)

        return config

    def callback_send_model(self, request, response):
        if self.model_config != None:
            response.success = True
            response.message = str(self.model_config)
        
        else:
            response.success = False
            response.message = "Model not ready yet"

        return response
    
    def callback_add_weights(self, request, response):
        request_message = json.loads(request.data)
        
        if request_message["weights"] != None:
            self.get_logger().info('Incoming request from: ' + request_message["client"])
            self.add_weights(request_message) 
            response.success = True
            response.message = "Weights added. Wait for all the other clients"

        else:
            response.success = False
            response.message = "Wrong request"
        
        return response
    
    def callback_send_weights(self, request, response):
        if (self._clients_counter % self.n_clients) != 0 and self._clients_counter > 0:
            response.success = False
            response.message = "Average of weights is not ready yet"
        
        else:
            data = self.get_average()
            response.success = True
            response.message = "OK"
            response.content = data
            self.weights = None
        
        return response
    
    def add_weights(self, request):
        
        self._clients_counter += 1
        
        weights_string = request["weights"]
        weights = deserialize_array(weights_string)

        if self.fl_weights == None:
            self.fl_weights = np.copy(weights)
        else:
            self.fl_weights = self.fl_weights + weights

        return True
    
    def get_average(self):
        new_weights = self.fl_weights / self.n_clients
        new_weights = serialize_array(new_weights)

        message = {
            "weights": new_weights
        }

        response = json.dumps(message)

        return response

def main(args=None):
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    #     except RuntimeError as e:
    #         print(e)
            
    rclpy.init(args=args)
    
    n_clients = int(sys.argv[1])
    model_selected = sys.argv[2] 

    node = FederatedServer(n_clients, model_selected)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()