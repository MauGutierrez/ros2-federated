import argparse
import rclpy
from ros2_federated_server.ros2_federated_server_async import FederatedServerAsync
from ros2_federated_server.ros2_federated_server_sync import FederatedServerSync


def main():
    parser = argparse.ArgumentParser(description="ROS 2 launcher.")
    parser.add_argument("--mode", choices=["sync", "async"], required=True, help="Select the connection mode between sync and async.")
    args, unkown = parser.parse_known_args()
    
    rclpy.init(args=unkown)

    if args.mode == "async":
        server = FederatedServerAsync()
    elif args.mode == "sync":
        server = FederatedServerSync()
    
    server.run()

    rclpy.shutdown()

if __name__ == '__main__':
    main()