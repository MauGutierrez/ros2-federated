import rclpy
import os
import sys
import time
import json

from rclpy.node import Node 
from ament_index_python.packages import get_package_share_directory
from ros2_federated_agents.ros2_federated_client import FederatedClient
from ros2_federated_agents.ros2_topics import Topics

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ROS TOPICS
CLIENT_NAME = "client_1"
CAMERA = "/camera_robot"
STOP_SIGNAL = "/stop_signal"

def main():
    # Init ROS topics
    topics = Topics()
    topics.CLIENT_NAME = CLIENT_NAME
    topics.CAMERA = CAMERA
    topics.STOP_SIGNAL = STOP_SIGNAL

    # Load general settings saved in json file
    settings = os.path.join(get_package_share_directory('ros2_federated_agents'), 'config/settings.json')

    with open(settings) as fp:
        content = json.load(fp)
        # Get hyperparameters from settings file
        hyperparameters = content['model_hyperparameters']
        iterations = hyperparameters['iterations']

        rclpy.init()

        # Init client object to handle communication with server
        client = FederatedClient(hyperparameters, topics)
        
        # Load test dataset
        client.load_local_dataset()

        # Preproces dataset
        client.preproces_dataset()
        
        # Start counting the execution time
        st = time.time()
        # Send an initial request to add the agent to the network
        response = client.add_agent_to_network()
        if response.success == False:
            exit(0)

        # Send a request to download the model from federated server 
        response = client.download_model_request()
        if response.success == True:
            # Model configuration from json
            model_config = response.message
            
            # Build the deep learning model
            client.build_local_model(model_config)

            for _ in range(iterations):        
                # Train the deep learning model
                client.train_local_model()

                # Serialize and get model weights
                data = client.serialize_model_weights()
                
                # Send a request to update the local weights
                response = client.add_local_weights_request(data)
                
                if response.success == True:
            
                    while rclpy.ok():
                        request = client.get_new_weights_request()
                        if request.success is False:
                            print(request.message)
                        else:
                            # Set new local weights that have been received from federated server
                            client.set_new_weights(request.content)
                            break
                    
                    while rclpy.ok():
                        request = client.wait_for_all_agents()
                        if request.success is False:
                            print(request.message)
                        else:
                            break
            
            # Train the deep learning model
            client.train_local_model(finish_train=True)

            # Stop timer and save time
            et = time.time()

            # Evaluate deep learning model
            client.evaluate_model()

            # Deep Learning Model is ready to predict
            client.model_is_ready = True
            
            # Subscribe to the camera images node
            while rclpy.ok():
                if client.stop_flag is False:
                    rclpy.spin_once(client)
                
                else:
                    client.save_model_information(st, et)
                    break
                    
        # Explicity destroy nodes 
        client.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()