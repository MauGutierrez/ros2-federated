import rclpy 
from rclpy.node import Node
from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights
from my_interfaces.srv import ConfigureAgent
from ros2_federated_server.helper_functions import serialize_array, deserialize_array
from ros2_federated_server.keras_models import get_model_from_json

import os
import sys
import numpy as np
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class FederatedServer(Node):

    _agents_counter = 0 # variable to control the iterations
    _n_agents = 0       # variable to store the number of agents
    _agents_list = []   # list to store the agents names
    _agents_ready = dict()
    _fl_weights = []     # List to store the mean of the weights
    
    def __init__(self):
        super().__init__('federated_server')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('dataset', rclpy.Parameter.Type.STRING)
            ]
        )

        self.selected_dataset = self.get_parameter('dataset').get_parameter_value().string_value

        # Initial model
        self.model_config = self.get_model_config(self.selected_dataset)

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

        # TODO
        # - Develop a service to add Agents to the network
        # - Check that every agent is added before or after the calculation of the average
        # - If an agent is added while the average, put it on wait to not affect the average
        
        # Service to add a new agent to the network
        self.add_agents_service_ = self.create_service(
            ConfigureAgent, "add_agent", self.callback_add_agent
        )

        self.wait_for_all_agents_ = self.create_service(
            ConfigureAgent, "wait_for_all_agents", self.callback_wait
        )

    def get_model_config(self, selected_model: str) -> object:
        self.get_logger().info('Model: ' + selected_model)
        config = get_model_from_json(selected_model)

        return config

    def callback_send_model(self, request, response):
        if self.model_config == None:
            response.success = False
            response.message = "Model not ready yet"
        else:
            response.success = True
            response.message = str(self.model_config)

        return response
    
    def callback_add_weights(self, request, response):
        request_message = json.loads(request.data)
        
        if "weights" not in request_message:
            response.success = False
            response.message = "Wrong request"
        else:
            self.add_weights(request_message) 
            response.success = True
            response.message = "Weights added. Wait for all the other clients"
        
        return response
    
    def callback_send_weights(self, request, response):
        self.get_logger().info(f'{request.data} Entering send_weights function.')
        if (self._agents_counter % self._n_agents) != 0:
            response.success = False
            response.message = "Average of weights is not ready yet"
        else:
            agent_name = request.data
            data = self.get_average()
            response.success = True
            response.message = "OK"
            response.content = data
            self._agents_ready[agent_name] = 1
        
        return response

    def callback_add_agent(self, request, response):
        agent_name = request.data
        
        if agent_name is None or agent_name == "":
            response.success = False
            response.message = "Wrong request"
        else:
            if agent_name in self._agents_list:
                response.success = False
                response.message = "Agent is already in the network. Use other name"
            else:
                self._agents_list.append(agent_name)
                self._agents_ready[agent_name] = 0
                self._n_agents = len(self._agents_list)
                response.success = True
                response.message = "OK"
        
        return response

    def callback_wait(self, request, response):
        if self.all_agents_ready() == False:
            response.success = False
            response.message = "Wait for all the agents to be ready"
        else:
            response.success = True
            response.message = "All agents are ready"

        return response
    
    def all_agents_ready(self):
        for agent in self._agents_ready:
            if self._agents_ready[agent] == 0:
                return False
        
        # self.get_logger().info(f'ALL AGENTS ARE READY')
        self._fl_weights = []

        return True

    def add_weights(self, request):
        self._agents_counter += 1
        agent_name = request["client"]
        self._agents_ready[agent_name] = 0
        weights_string = request["weights"]
        weights = deserialize_array(weights_string)

        self.get_logger().info(f'{request["client"]} Entering add_weights function.')
        self._fl_weights = self._fl_weights + weights
        self._fl_weights = np.array(self._fl_weights)

        return True
    
    def get_average(self):
        new_weights = self._fl_weights / self._n_agents
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
    node = FederatedServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()