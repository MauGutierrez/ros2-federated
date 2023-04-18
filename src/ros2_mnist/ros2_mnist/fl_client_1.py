import rclpy 
from rclpy.node import Node 
from ament_index_python.packages import get_package_share_directory
from example_interfaces.srv import Trigger
from ros2_mnist.weights_publisher import ModelPublisher

import os
import cv2
import json
import numpy as np
import tensorflow as tf

# Global Variables
CLIENT_NAME = "client_A"

class FederatedClientA(Node):
    def __init__(self, model_config, client_name):
        super().__init__('federated_client_a')
        
        self.model = None

        self.client_name = client_name

        # JSON file to store configuration parameters
        self.model_config = model_config

        # Dataset
        self.dataset = None

        self.X_test = None
        self.X_train = None

        self.y_test = None
        self.y_train = None

        self.cli = self.create_client(Trigger, "get_model")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def preproces_data(self, dataset):
        self.dataset = dataset
        (self.X_train, self.y_train), (self.X_test, y_test) = self.dataset
        self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0
        

    def build_model(self, model_config):
        self.model = tf.keras.models.model_from_json(model_config)
        self.model.compile(
            optimizer=self.model_config['optimizer'],
            loss=self.model_config['loss'],
            metrics=self.model_config['metrics']
        )
        self.model.summary()
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.model_config['epochs'])
    
    def serialize_model_weights(self):
        self.model_weights = np.array(self.model.get_weights())

        weights = np.array_str(self.model_weights)

        print(weights)

        return weights

def main():
    # Load model hyperparameters
    settings = os.path.join(get_package_share_directory('ros2_mnist'), "settings.json")

    # Load test dataset of mnist
    mnist = tf.keras.datasets.mnist
    dataset = mnist.load_data()

    with open(settings) as fp:
        content = json.load(fp)
        hyperparameters = content['model_hyperparameters']

        rclpy.init()

        # Init client object to handle communication with server
        client = FederatedClientA(hyperparameters, CLIENT_NAME)

        # Preproces the dataset
        client.preproces_data(dataset)

        # Send an initial request to download the model from federated server 
        response = client.send_request()
        if response.success == True:
            # Model configuration from json
            model_config = response.message

            # Build the deep learning model
            client.build_model(model_config)

            # Train the deep learning model
            client.train_model()

            # Serialize and get model weights
            weights = client.serialize_model_weights()

            # Init Model publisher
            publisher = ModelPublisher(CLIENT_NAME, weights)
            
            rclpy.spin(publisher)

            # Explicity destroy nodes 
            client.destroy_node()
            publisher.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()