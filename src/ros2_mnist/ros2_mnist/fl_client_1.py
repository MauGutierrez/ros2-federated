import rclpy 
from rclpy.node import Node 
from ament_index_python.packages import get_package_share_directory
from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights
from sensor_msgs.msg import Image
from ros2_mnist.helper_functions import *
from keras.datasets import cifar10
from keras.datasets import mnist

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import cv2
import json
import numpy as np
import tensorflow as tf

# Global Variables
CLIENT_NAME = "client_1"
TOPIC_CAMERA = "/camera/Right"

class FederatedClientA(Node):
    def __init__(self, model_config, client_name):
        super().__init__('federated_client_1')
        
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

        self.subscription_camera_topic = self.create_subscription(Image, TOPIC_CAMERA, self.callback_camera, 30)

        # Client to request the initial model
        self.model_cli = self.create_client(Trigger, "get_model")
        while not self.model_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.model_req = Trigger.Request()

        # Client to request the update of the weights
        self.weights_cli = self.create_client(SendLocalWeights, "update_weights")
        while not self.weights_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.weights_req = SendLocalWeights.Request()

    def build_model_request(self):
        self.future = self.model_cli.call_async(self.model_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def update_weights_request(self, message):
        self.weights_req.message_request = message
        self.future = self.weights_cli.call_async(self.weights_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def preproces_data(self, dataset):
        self.dataset = dataset
        (self.X_train, self.y_train), (self.X_test, y_test) = self.dataset

        # From the original dataset, only number from 0 to 4 are selected
        indexes = np.array([i for i in range(0, len(self.y_train)) if self.y_train[i] >= 0 and self.y_train[i] <= 4])
        labels = np.array([label for label in self.y_train if label >= 0 and label <= 4])
        self.X_train = np.array([self.X_train[index] for index in indexes])
        self.y_train = labels.copy()

        self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0

    def build_local_model(self, model_config):
        self.model = tf.keras.models.model_from_json(model_config)
        self.model.compile(
            optimizer=self.model_config['optimizer'],
            loss=self.model_config['loss'],
            metrics=self.model_config['metrics']
        )
        self.model.summary()
    
    def train_model(self):
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            batch_size=self.model_config['batch_size'],
            epochs=self.model_config['epochs']
        )
    
    def monitor_training(self):
        return self.history.history['loss']
    
    def serialize_model_weights(self):
        self.model_weights = self.model.get_weights()

        weights = serialize_array(self.model_weights)
        message = {
            "client": self.client_name,
            "weights": weights
        }

        message = json.dumps(message)

        return message
    
    def set_new_weights(self, weights):
        data = json.loads(weights)
        new_weights = data["weights"]
        new_weights = deserialize_array(new_weights)

        self.model.set_weights(new_weights)
    
    def select_dataset(self, selected_dataset):
        dataset = None
        if selected_dataset == "mnist":
            dataset = mnist.load_data()
        
        elif selected_dataset == "cifar10":
            dataset = cifar10.load_data()
        
        else:
            print("No Dataset found. Try with a different one.")
            exit(1)
        
        return dataset

    def callback_camera(self, image_msg):
        # self.node.get_logger().info('I heard: "%s"' % image_msg.data)
        self.node.get_logger().info('I heard topic: "%s"' % TOPIC_CAMERA)

def main():
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    #     except RuntimeError as e:
    #         print(e)

    # Load model hyperparameters
    settings = os.path.join(get_package_share_directory('ros2_mnist'), "settings.json")

    with open(settings) as fp:
        content = json.load(fp)
        hyperparameters = content['model_hyperparameters']
        iterations = hyperparameters['iterations']

        rclpy.init()

        # Init client object to handle communication with server
        client = FederatedClientA(hyperparameters, CLIENT_NAME)

        # Load test dataset
        dataset = client.select_dataset(sys.argv[1])

        # Preproces the dataset
        client.preproces_data(dataset)

        # Send an initial request to download the model from federated server 
        response = client.build_model_request()
        if response.success == True:
            # Model configuration from json
            model_config = response.message
            
            # Build the deep learning model
            client.build_local_model(model_config)

            for i in range(iterations):        
                # Train the deep learning model
                client.train_model()

                # Serialize and get model weights
                request = client.serialize_model_weights()
                
                # Send a request to update the local weights
                response = client.update_weights_request(request)

                if response.success == True:
                    # Set new local weights that have been received from federated server
                    client.set_new_weights(response.message_response)
            
            # Subscribe to the camera images node
            rclpy.spin(client)

        # Explicity destroy nodes 
        client.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()