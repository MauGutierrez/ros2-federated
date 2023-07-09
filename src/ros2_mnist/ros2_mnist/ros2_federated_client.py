import rclpy 
from rclpy.node import Node 
from ament_index_python.packages import get_package_share_directory
from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights
from my_interfaces.msg import UnityImage
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ros2_mnist.helper_functions import *
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix
from ros2_mnist.ros2_topics import Topics
from PIL import Image

import os
import sys
import cv2
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
import seaborn as sn
import io
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class FederatedClient(Node):
    def __init__(self, model_config, topics: Topics):
        super().__init__(topics.CLIENT_NAME)

        self.model = None
        self.client_name = topics.CLIENT_NAME
        self.model_config = model_config
        self.dataset_name = None
        self.train_dataset = None
        self.test_dataset = None
        self.X_test = []
        self.X_train = []
        self.y_test = []
        self.y_train = []
        self._cv_bridge = CvBridge()
        self.model_is_ready = False
        self.predictions_list = []
        self.labels_list = []
        self.stop_flag = False

        self.subscription_camera_topic = self.create_subscription(UnityImage, topics.CAMERA, self.callback_camera, 1)
        self.subscription_stop_validation = self.create_subscription(Bool, topics.STOP_SIGNAL, self.callback_stop, 1)

        # Client to request the initial model
        self.model_cli = self.create_client(Trigger, "download_model")
        while not self.model_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.model_req = Trigger.Request()

        # Client to request the addition of the weights
        self.weights_cli = self.create_client(SendLocalWeights, "add_weights")
        while not self.weights_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.weights_req = SendLocalWeights.Request()

        # Client to request the update of the weights
        self.update_cli = self.create_client(SendLocalWeights, "get_weights")
        while not self.update_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.update_req = SendLocalWeights.Request()

    def download_model_request(self):
        self.future = self.model_cli.call_async(self.model_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def add_local_weights_request(self, message):
        self.weights_req.data = message
        self.future = self.weights_cli.call_async(self.weights_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def get_new_weights_request(self):
        self.future = self.update_cli.call_async(self.update_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def callback_camera(self, image_msg):        
        if self.model_is_ready:
            if image_msg.label.data != '':
                # Get the label of the image
                label = int(image_msg.label.data)
                
                self.get_logger().info('I heard: "%s"' % label)

                # Convert image to cv_bridge
                cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg.unity_image, "bgr8")
                cv_image_rotated = cv2.rotate(cv_image, cv2.ROTATE_180)
                cv_image_flipped = cv2.flip(cv_image_rotated, 1)

                cv_image_gray = cv2.cvtColor(cv_image_flipped, cv2.COLOR_BGR2GRAY)
                cv_image_28 = cv2.resize(cv_image_gray, (28 , 28))     
                np_image = cv_image_28.reshape((1, 28, 28, 1))

                prediction = self.model.predict(np_image)

                answer = np.argmax(prediction, 1)[0]
                
                self.labels_list.append(label)
                self.predictions_list.append(answer)

                self.get_logger().info('Number is: %d' % answer)
            else:
                self.get_logger().info('Waiting for images')
    
    def callback_stop(self, stop_flag):
        self.stop_flag = stop_flag.data

    def load_local_dataset(self, selected_dataset: str) -> None:
        self.dataset_name = selected_dataset

        if self.dataset_name == "mnist":
            # csv files that contain the train and test dataset
            dataset_train = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/train.csv")
            dataset_test = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/test.csv")
            
            # imgs from the dataset
            imgs_train = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/train.zip")
            imgs_test = os.path.join(get_package_share_directory('ros2_mnist'), "dataset/test.zip")

            # Read train and test CSV that contain the image path and labels
            df = pd.read_csv(dataset_train)
            images_train = df['image_paths']
            labels_train = df['labels']

            df = pd.read_csv(dataset_test)
            images_test = df['image_paths']
            labels_test = df['labels']
            
            self.train_dataset = (images_train, labels_train)
            self.test_dataset = (images_test, labels_test)

            # Extract data from train and test dataset and get the images and labels
            self.extract_data_from_zip(imgs_train, self.X_train, self.y_train, self.train_dataset)            
            self.extract_data_from_zip(imgs_test, self.X_test, self.y_test, self.test_dataset)

        elif self.dataset_name == "cifar10":
            self.dataset = cifar10.load_data()
        
        else:
            self.get_logger().info("No Dataset found. Try with a different one.")
            exit(1)

    def extract_data_from_zip(self, imgs, x_array = [], y_array = [], dataset = []) -> None:
        archive = zipfile.ZipFile(imgs, 'r')

        for (img_path, label) in zip(dataset[0], dataset[1]):
            file_path = img_path.replace('dataset/', '') 
            
            img_file = archive.read(file_path)
            bytes_io = io.BytesIO(img_file)
            img = Image.open(bytes_io)
            image_as_array = np.array(img, np.uint8)

            x_array.append(image_as_array)
            y_array.append(label)

    def preproces_dataset(self) -> None:
        if self.dataset_name == "mnist":
            # Get a random portion of the dataset
            # to simulate that the dataset is different for every client
            n = len(self.y_train)
            indexes = random.sample(range(n), int(n/2))

            self.X_train = np.array(self.X_train)[indexes]
            self.y_train = np.array(self.y_train)[indexes]

        else:
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)

        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0

    def build_local_model(self, model_config: object) -> None:
        self.model = tf.keras.models.model_from_json(model_config)
        self.model.compile(
            optimizer=self.model_config['optimizer'],
            loss=self.model_config['loss'],
            metrics=self.model_config['metrics']
        )
        self.model.summary()
    
    def train_local_model(self) -> None:
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            batch_size=self.model_config['batch_size'],
            epochs=self.model_config['epochs']
        )
    
    def serialize_model_weights(self) -> object:
        self.model_weights = self.model.get_weights()
        weights_serialized = serialize_array(self.model_weights)
        
        message = {
            "client": self.client_name,
            "weights": weights_serialized
        }

        message = json.dumps(message)

        return message
    
    def set_new_weights(self, weights: object) -> None:
        data = json.loads(weights)
        new_weights = data["weights"]
        new_weights = deserialize_array(new_weights)

        self.model.set_weights(new_weights)

    def evaluate_model(self) -> None:
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.get_logger().info(f'Test accuracy: is: {test_acc}')

    def get_confusion_matrix(self) -> None:
        labels = set(self.labels_list) | set(self.predictions_list)
        labels = list(labels)

        self.labels_list = np.array(self.labels_list)
        self.predictions_list = np.array(self.predictions_list)

        confusion = confusion_matrix(self.labels_list, self.predictions_list)
        plt.figure(figsize = (10,7))
        cfm_plot = sn.heatmap(confusion, xticklabels=labels, yticklabels=labels, annot=True)
        cfm_plot.figure.savefig(self.client_name + ".png")