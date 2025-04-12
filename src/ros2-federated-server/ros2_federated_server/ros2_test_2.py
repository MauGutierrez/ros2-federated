import rclpy
from rclpy.node import Node

from my_interfaces.srv import ConfigureAgent

class TestService(Node):
    def __init__(self):
        super().__init__('test_service')

        # Client to request the addition of a new agent in the network
        self.add_agent_cli = self.create_client(ConfigureAgent, "add_agent")
        while not self.add_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.add_agent_req = ConfigureAgent.Request()

    def add_agent_to_network(self, name):
        self.add_agent_req.data = name
        
        return self.add_agent_cli.call_async(self.add_agent_req)


def main():
    # Init ROS
    rclpy.init()

    test_srv = TestService()
    
    future = test_srv.add_agent_to_network("test 2")
    rclpy.spin_until_future_complete(test_srv, future)
    
    response = future.result()

    if response.success is True:
        test_srv.get_logger().info('Node has been added to the network.')
    else:
        test_srv.get_logger().info(response.message)
    
    test_srv.destroy_node()
    # Explicity destroy nodes 
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()