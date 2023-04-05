import rclpy 
from rclpy.node import Node 
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from example_interfaces.srv import Trigger

from cv_bridge import CvBridge
import os
import cv2
import json
import numpy as np
import tensorflow as tf

class FederatedClientA(Node):
    def __init__(self, model_config):
        super().__init__('federated_client_a')
        
        self.model = None

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
        

    def build_model(self):
        self.model = tf.keras.models.load_model("my_model.h5")
        self.model.compile(
            optimizer=self.model_config['optimizer'],
            loss=self.model_config['loss'],
            metrics=self.model_config['metrics']
        )
        self.model.summary()
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.model_config['epochs'])

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
        client = FederatedClientA(hyperparameters)

        # Preproces the dataset
        client.preproces_data(dataset)

        # Send an initial request to download the model from federated server 
        response = client.send_request()
        if response.message == "Init download":
            # Build the deep learning model
            client.build_model()

            # Train the deep learning model
            client.train_model()

        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()