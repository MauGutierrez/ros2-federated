
import rclpy

from my_interfaces.srv import LocalValues
from my_interfaces.srv import ConfigureAgent
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup


class SyncConnection(Node):
    def __init__(self, agent_name):
        super().__init__('federated_' + agent_name)

        wait_cb = MutuallyExclusiveCallbackGroup()
        get_values_cb = MutuallyExclusiveCallbackGroup()
        # Client to request the addition of the weights
        self.loss_cli = self.create_client(LocalValues, "add_to_global")
        while not self.loss_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.loss_req = LocalValues.Request()

        # Client to request the addition of a new agent in the network
        self.add_agent_cli = self.create_client(ConfigureAgent, "add_agent")
        while not self.add_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.add_agent_req = ConfigureAgent.Request()

        self.wait_agent_cli = self.create_client(ConfigureAgent, "wait_for_all_agents", callback_group=wait_cb)
        while not self.wait_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.wait_agent_req = ConfigureAgent.Request()

        # Client to request the new global value
        self.update_cli = self.create_client(LocalValues, "get_global_value", callback_group=get_values_cb)
        while not self.update_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.update_req = LocalValues.Request()

        # Client to request the remove of an agent in the network
        self.remove_agent_cli = self.create_client(ConfigureAgent, "remove_agent")
        while not self.remove_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.remove_agent_req = ConfigureAgent.Request()

    def add_agent_to_network(self, name):
        self.add_agent_req.data = name
        self.future_add_agent = self.add_agent_cli.call_async(self.add_agent_req)
        rclpy.spin_until_future_complete(self, self.future_add_agent)
        return self.future_add_agent.result()
    
    def add_local_weights_request(self, message):
        self.loss_req.data = message
        self.future_add_local = self.loss_cli.call_async(self.loss_req)
        rclpy.spin_until_future_complete(self, self.future_add_local)
        return self.future_add_local.result()

    def wait(self):
        self.future_wait = self.wait_agent_cli.call_async(self.wait_agent_req)
        rclpy.spin_until_future_complete(self, self.future_wait, timeout_sec=5.0)
        return self.future_wait.result()
    
    def get_new_weights_request(self, message):
        self.update_req.data = message
        self.future_get_global = self.update_cli.call_async(self.update_req)
        rclpy.spin_until_future_complete(self, self.future_get_global, timeout_sec=5.0)
        return self.future_get_global.result()
    
    def remove_agent_from_network(self, name):
        self.remove_agent_req.data = name
        self.future_remove = self.remove_agent_cli.call_async(self.remove_agent_req)
        rclpy.spin_until_future_complete(self, self.future_remove)
        return self.future_remove.result()
    

class AsyncConnection(Node):
    def __init__(self, agent_name):
        super().__init__('federated_' + agent_name)

        wait_cb = MutuallyExclusiveCallbackGroup()
        get_values_cb = MutuallyExclusiveCallbackGroup()
        # Client to request the addition of the weights
        self.loss_cli = self.create_client(LocalValues, "add_to_global")
        while not self.loss_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.loss_req = LocalValues.Request()

        # Client to request the addition of a new agent in the network
        self.add_agent_cli = self.create_client(ConfigureAgent, "add_agent")
        while not self.add_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.add_agent_req = ConfigureAgent.Request()

        # Client to request the new global value
        self.update_cli = self.create_client(LocalValues, "get_global_value", callback_group=get_values_cb)
        while not self.update_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.update_req = LocalValues.Request()

        # Client to request the remove of an agent in the network
        self.remove_agent_cli = self.create_client(ConfigureAgent, "remove_agent")
        while not self.remove_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.remove_agent_req = ConfigureAgent.Request()


        self.wait_agent_cli = self.create_client(ConfigureAgent, "wait_for_buffer", callback_group=wait_cb)
        while not self.wait_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.wait_agent_req = ConfigureAgent.Request()

    def add_agent_to_network(self, name):
        self.add_agent_req.data = name
        self.future_add_agent = self.add_agent_cli.call_async(self.add_agent_req)
        rclpy.spin_until_future_complete(self, self.future_add_agent)
        return self.future_add_agent.result()
    
    def add_local_weights_request(self, message):
        self.loss_req.data = message
        self.future_add_local = self.loss_cli.call_async(self.loss_req)
        rclpy.spin_until_future_complete(self, self.future_add_local)
        return self.future_add_local.result()
    
    def get_new_weights_request(self, message):
        self.update_req.data = message
        self.future_get_global = self.update_cli.call_async(self.update_req)
        rclpy.spin_until_future_complete(self, self.future_get_global, timeout_sec=5.0)
        return self.future_get_global.result()
    
    def remove_agent_from_network(self, name):
        self.remove_agent_req.data = name
        self.future_remove = self.remove_agent_cli.call_async(self.remove_agent_req)
        rclpy.spin_until_future_complete(self, self.future_remove)
        return self.future_remove.result()
    
    def wait(self):
        self.future_wait = self.wait_agent_cli.call_async(self.wait_agent_req)
        rclpy.spin_until_future_complete(self, self.future_wait, timeout_sec=5.0)
        return self.future_wait.result()