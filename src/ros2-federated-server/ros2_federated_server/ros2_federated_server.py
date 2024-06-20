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

    _agents_counter = 0
    _n_agents = 0
    _agents_list = []
    _agents_ready = dict() 
    _fl_loss = 0
    
    def __init__(self):
        super().__init__('federated_server')
        self.get_logger().info(f'Federated Server is running.')
        # self.declare_parameters(
        #     namespace='',
        #     parameters=[
        #         ('dataset', rclpy.Parameter.Type.STRING)
        #     ]
        # )

        # Service to get the average of the weights
        self.update_weights_service_ = self.create_service(
            LocalValues, "add_to_global", self.callback_add_to_global
        )
        
        # Service to add a new agent to the network
        self.add_agents_service_ = self.create_service(
            ConfigureAgent, "add_agent", self.callback_add_agent
        )

        # Service to send the new global value
        self.update_global_model_ = self.create_service(
            LocalValues, "get_global_value", self.callback_send_global
        )

        # Service to synchronize all the agents in the network
        self.wait_for_all_agents_ = self.create_service(
            ConfigureAgent, "wait_for_all_agents", self.callback_wait
        )
    

    def callback_add_to_global(self, request, response):
        request_message = json.loads(request.data)
        
        if "local_value" not in request_message:
            response.success = False
            response.message = "Wrong request"
        else:
            self.add_to_global_value(request_message) 
            response.success = True
            response.message = "Weights added. Wait for all the other clients"
        
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
                self.get_logger().info(f'{agent_name} has been added to the federated network.')
                self._agents_list.append(agent_name)
                self._agents_ready[agent_name] = 0
                self._n_agents = len(self._agents_list)
                response.success = True
                response.message = "OK"
        
        return response

    def callback_send_global(self, request, response):
        self.get_logger().info(f'{request.data} Entering send_global function.')
        if (self._agents_counter % self._n_agents) != 0:
            response.success = False
            response.message = "Global new value is not ready yet"
        else:
            agent_name = request.data
            data = self.get_average()
            response.success = True
            response.message = "OK"
            response.global_value = data
            self._agents_ready[agent_name] = 1
        
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
        
        self._fl_weights = []

        return True

    def add_to_global_value(self, request):
        self.get_logger().info(f'{request["client"]} Entering add_loss function.')
        self._agents_counter += 1
        agent = request["client"]
        agent_loss = request["local_value"]
        self._agents_ready[agent] = 0
        self._fl_loss = self._fl_loss + agent_loss

    def get_average(self):
        new_value = self._fl_loss / self._n_agents

        return new_value
    

def main(args=None):            
    rclpy.init(args=args)
    node = FederatedServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()