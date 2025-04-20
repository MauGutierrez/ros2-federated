import rclpy 
from rclpy.node import Node
from my_interfaces.srv import LocalValues
from my_interfaces.srv import ConfigureAgent

import numpy as np
import json
import time
import torch

from threading import Lock, Thread

class FederatedServerSync(Node):
    def __init__(self):
        self._agents_counter = 0
        self._n_agents = 0
        self._agents_list = []
        self._agents_ready = dict() 
        self._fl_loss = []
        self._lock = Lock()

        super().__init__('federated_server_sync')
        self.get_logger().info(f'Federated Sync Server is running.')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('time_threshold', rclpy.Parameter.Type.INTEGER)
            ]
        )

        # Time threshold
        self.time_threshold = self.get_parameter('time_threshold').value * 1

        # Timer
        self.start_time = 0

        # Service to get the average of the weights
        self.update_weights_service_ = self.create_service(
            LocalValues, "add_to_global", self.callback_add_to_global
        )
        
        # Service to add a new agent to the network
        self.add_agents_service_ = self.create_service(
            ConfigureAgent, "add_agent", self.callback_add_agent
        )

        # Service to remove an agent from the network
        self.remove_agents_service = self.create_service(
            ConfigureAgent, "remove_agent", self.callback_remove_agent
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
        # Protocol to prevent super later nodes
        if self.start_time > 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            if elapsed_time > self.time_threshold:
                response.success = False
                response.message = "Initialization phase is done. No new nodes can be added to the network."
            
                return response

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
                with self._lock:
                    if len(self._agents_list) == 0:
                        self.start_time = time.time()
                        
                    self._agents_list.append(agent_name)
                    self._agents_ready[agent_name] = 0
                    self._n_agents = len(self._agents_list)

                self.get_logger().info(f'Agents in network: {self._n_agents}.')
                response.success = True
                response.message = "OK"
        
        return response

    def callback_send_global(self, request, response):
        if (self._agents_counter % self._n_agents) != 0:
            response.success = False
            response.message = "Global new value is not ready yet"
        else:
            self.get_logger().info(f'callback_send_global :: Sending average to: {request.data}.')
            agent_name = request.data
            data = self.get_average()
            response.success = True
            response.message = "OK"
            response.global_value = data
            with self._lock:
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

    def callback_remove_agent(self, request, response):
        agent_name = request.data
        if agent_name is None or agent_name == "":
            response.success = False
            response.message = "Wrong request"
        else:
            if agent_name not in self._agents_list:
                response.success = False
                response.message = "Agent is not in the network. Use the correct name."
            else:
                self.get_logger().info(f'{agent_name} has been removed from the federated network.')
                with self._lock:
                    self._agents_list.remove(agent_name)
                    del self._agents_ready[agent_name]
                    self._n_agents = len(self._agents_list)
                response.success = True
                response.message = "OK"
                
        return response

    def all_agents_ready(self):
        for agent in self._agents_ready:
            if self._agents_ready[agent] == 0:
                return False
        
        self._fl_loss = []

        return True

    def add_to_global_value(self, request):
        self.get_logger().info(f'add_to_global :: {request["client"]} is entering.')
        agent = request["client"]
        agent_loss = request["local_value"]
        # self.get_logger().info(f'add_to_global :: Loss from {request["client"]}: {agent_loss}')
        
        
        with self._lock:
            self._agents_counter += 1
            self._agents_ready[agent] = 0
            agent_loss = [torch.tensor(vector).float() for vector in agent_loss]
            
            self._fl_loss = self._fl_loss + agent_loss
            self._fl_loss = np.array(self._fl_loss)
        
        self.get_logger().info(f'add_to_global :: Counter {self._agents_counter}')

    def get_average(self):
        with self._lock:
            new_global = self._fl_loss / self._n_agents
            new_arr = [vector.tolist() for vector in new_global]

            message = {
                "weights": new_arr
            }

            response = json.dumps(message)

            return response
    
    def run(self):
        rclpy.spin(self)

    

def main(args=None):            
    rclpy.init(args=args)
    server = FederatedServerSync()
    # rclpy.spin(node)
    server.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()