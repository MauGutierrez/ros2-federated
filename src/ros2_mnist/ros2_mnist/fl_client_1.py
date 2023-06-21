import rclpy 
from rclpy.node import Node 
from ament_index_python.packages import get_package_share_directory
from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights
from my_interfaces.msg import UnityImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
import matplotlib.pyplot as plt
import zipfile
import pandas
import io
from PIL import Image

# Global Variables
CLIENT_NAME = "client_1"
TOPIC_CAMERA = "/camera_robot/Right"

class FederatedClientA(Node):
    def __init__(self, model_config, client_name):
        super().__init__('federated_client_1')
        
        self.model = None

        self.client_name = client_name

        # JSON file to store configuration parameters
        self.model_config = model_config

        self.train_dataset = None
        self.test_dataset = None

        self.X_test = []
        self.X_train = []

        self.y_test = []
        self.y_train = []

        self._cv_bridge = CvBridge()

        # Flag to indicate that the deep learning model is ready to predict 
        self.model_ready = False

        self.subscription_camera_topic = self.create_subscription(UnityImage, TOPIC_CAMERA, self.callback_camera, 1)

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

    def extract_data_from_zip(self, imgs, x_array = [], y_array = [], dataset = []):
        archive = zipfile.ZipFile(imgs, 'r')

        for (img_path, label) in zip(dataset[0], dataset[1]):
            file_path = img_path.replace('dataset/', '') 
            
            img_file = archive.read(file_path)
            bytes_io = io.BytesIO(img_file)
            img = Image.open(bytes_io)
            image_as_array = np.array(img, np.uint8)

            x_array.append(image_as_array)
            y_array.append(label)

    def preproces_data(self, dataset):
        imgs_train = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/train.zip")
        imgs_test = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/test.zip")

        # Extract data from train dataset
        self.extract_data_from_zip(imgs_train, self.X_train, self.y_train, self.train_dataset)
        
        # Extract data from test dataset
        self.extract_data_from_zip(imgs_test, self.X_test, self.y_test, self.test_dataset)
        
        if dataset == "mnist":
            # From the original dataset, only numbers from 0 to 4 are selected
            indexes = np.array([i for i in range(0, len(self.y_train)) if self.y_train[i] >= 0 and self.y_train[i] <= 4])
            labels = np.array([label for label in self.y_train if label >= 0 and label <= 4])
            self.X_train = np.array([self.X_train[index] for index in indexes])
            self.y_train = labels.copy()
        else:
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)

        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

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
    
    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print('\nTest accuracy:', test_acc)

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
        if selected_dataset == "mnist":
            dataset_train = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/train.csv")
            dataset_test = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/test.csv")

            # Train dataset
            df = pandas.read_csv(dataset_train)
            images_train = df['image_paths']
            labels_train = df['labels']

            # Test dataset
            df = pandas.read_csv(dataset_test)
            images_test = df['image_paths']
            labels_test = df['labels']
            
            self.train_dataset = (images_train, labels_train)
            self.test_dataset = (images_test, labels_test)

        elif selected_dataset == "cifar10":
            self.dataset = cifar10.load_data()
        
        else:
            print("No Dataset found. Try with a different one.")
            exit(1)

    def callback_camera(self, image_msg):        
        if self.model_ready:
            # Get the label of the image
            label = image_msg.label
            self.get_logger().info('I heard: "%s"' % label)

            # Convert image to cv_bridge
            cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg.unity_image, "bgr8")
            cv_image_rotated = cv2.rotate(cv_image, cv2.ROTATE_180)
            cv_image_flipped = cv2.flip(cv_image_rotated, 1)

            cv_image_gray = cv2.cvtColor(cv_image_flipped, cv2.COLOR_RGB2GRAY)
            ret, cv_image_binary = cv2.threshold(cv_image_gray, 128, 255, cv2.THRESH_BINARY_INV)  
            cv_image_28 = cv2.resize(cv_image_binary, (28 , 28))     
            np_image = np.reshape(cv_image_28, (1, 28, 28, 1))

            prediction = self.model.predict(np_image)

            answer = np.argmax(prediction, 1)
            self.get_logger().info('Number is: %d' % answer)
            # predict the number
            cv2.imshow("camera 1", cv_image_flipped)
            cv2.waitKey(1)

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
        
        dataset_selected = sys.argv[1]

        # Load test dataset
        client.select_dataset(dataset_selected)

        # Preproces the dataset
        client.preproces_data(dataset_selected)

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
            
            # Evaluate deep learning model
            client.evaluate_model()

            # Deep Learning Model is ready to predict
            client.model_ready = True
            # Subscribe to the camera images node
            rclpy.spin(client)

        # Explicity destroy nodes 
        client.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()