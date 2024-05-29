import rclpy 
from rclpy.node import Node
from example_interfaces.srv import Trigger
from my_interfaces.srv import LocalValues
from my_interfaces.srv import ConfigureAgent

import os
import sys
import numpy as np
import json

class FederatedServer(Node):

    _n_agents = 0       # variable to store the number of agents
    _agents_list = []   # list to store the agents names
    _agents_buffer = [] # buffer to store the agents who already send their contribution
    _fl_loss = 0.0   # Float parameter to store the global loss 
    
    def __init__(self):
        super().__init__('federated_server')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('dataset', rclpy.Parameter.Type.STRING)
            ]
        )

        # Service to get the average of the weights
        self.update_weights_service_ = self.create_service(
            LocalValues, "add_to_global", self.callback_add_to_global
        )
        
        # Service to add a new agent to the network
        self.add_agents_service_ = self.create_service(
            ConfigureAgent, "add_agent", self.callback_add_agent
        )
    

    def callback_add_to_global(self, request, response):
        request_message = json.loads(request.data)
        agent_name = request_message["client"]

        if "loss" not in request_message:
            response.success = False
            response.message = "Wrong request"
        else:
            # State 1: agent hasn't send his collaboiration to global
            if len(self._agents_buffer) < self._n_agents and agent_name not in self._agents_buffer:
                self._agents_buffer.append(agent_name)
                self.add_loss(request_message)        
                response.success = True
                response.message = "Loss added"
            # State 2: agent is in the waiting buffer and the K agents has already participated
            if len(self._agents_buffer) == self._n_agents and agent_name in self._agents_buffer:
                agent_index = self._agents_buffer.index(agent_name)
                self._agents_buffer[agent_index] = -1
                response.success = True
                response.message = "OK"
                response.global_value = self.get_average()
                self.empty_agents_buffer()
            # State 3: do nothing since I already send my collaboration and there are missing agents
            else:
                response.success = True
                response.message = "Wait for all the other agents to get the global update"
        
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
                self._n_agents = len(self._agents_list)
                response.success = True
                response.message = "OK"
        
        return response

    def add_loss(self, request):
        client_loss = request["loss"]
        self.get_logger().info(f'{request["client"]} Entering add_loss function.')
        self._fl_loss = self._fl_loss + client_loss
    
    def get_average(self):
        new_value = self._fl_loss / self._n_agents

        return new_value
    
    def empty_agents_buffer(self):
        all_ready = True
        for i in range(len(self._agents_buffer)):
            if self._agents_buffer[i] != -1:
                all_ready = False
                break

        # If the last agent already received the global
        # clean the agents buffer and set to 0.0 the global value
        if all_ready:
            self._fl_loss = 0.0
            del self._agents_buffer[:]

def main(args=None):            
    rclpy.init(args=args)
    node = FederatedServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()