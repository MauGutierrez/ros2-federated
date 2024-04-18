import cv2
import math
import numpy as np
import random
import rclpy
import time

from cv_bridge import CvBridge
from gym.spaces import Discrete
from my_interfaces.srv import PositionService
from my_interfaces.srv import InitUnityObjects
from rclpy.node import Node

import torch
import torchvision.transforms as transforms

IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 60
DELTA_DISTANCE = 1.0000000

class Coordinates():
    
    def __init__(self, pos_x, pos_y, pos_z) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z

class UnityObject(Node):

    def __init__(self):
        super().__init__('unity_object')
        self.init_cli = self.create_client(InitUnityObjects, 'init_unity_objects')
        while not self.init_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.inital_req = InitUnityObjects.Request()

        self.actions_cli = self.create_client(PositionService, 'move_unity_object')
        while not self.actions_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.action_req = PositionService.Request()

    def request_init_unity_objects(self):
        self.future = self.init_cli.call_async(self.inital_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def request_action_to_unity(self, action):
        # self.get_logger().info(f'My action: {action}')
        self.action_req.action = action
        self.future = self.actions_cli.call_async(self.action_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

class UnityEnv():
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self) -> None:
        self.unity_obj = UnityObject()
        self.objective_coordinates = Coordinates(0.0, 0.0, 0.0)
        self.agent_coordinates = Coordinates(0.0, 0.0, 0.0)
        self.collisions = 0
        self.delta = DELTA_DISTANCE
        self.avg_distance = 0.0
        self.measruements_counter = 0
        self.action_space = Discrete(3)
        self._cv_bridge = CvBridge()

    def reset(self):
        # Restart the initial coordinates of the object
        # Restart the initial coordinates of the agent
        # Restart the flag to detect collisons
        # Get the initial observation
        response = self.unity_obj.request_init_unity_objects()
        # Get the initial observation Image
        if response.success is True:
            # Start counting the time of an episode
            self.episode_start = time.time()
            # Environment observation
            observation, _ = self.__unity_image_formater(response.unity_image)
            # Initial coordinates of the objective
            self.objective_coordinates.pos_x = response.objective.pos_x
            self.objective_coordinates.pos_y = response.objective.pos_y
            self.objective_coordinates.pos_z = response.objective.pos_z
            # self.unity_obj.get_logger().info(f'{ response.objective.pos_x}, {response.objective.pos_y}, {response.objective.pos_z}')
            # Initial coordinates of the agent
            self.agent_coordinates.pos_x = response.agent.pos_x
            self.agent_coordinates.pos_y = response.agent.pos_y
            self.agent_coordinates.pos_z = response.agent.pos_z
            # Reset the number of collisions
            self.collisions = 0
            # Reset the counter of measurements
            self.measruements_counter = 0
            # Reset the average distances 
            self.avg_distance = self.__euclidean_distance(self.agent_coordinates, self.objective_coordinates)
        else:
            self.unity_obj.get_logger().warning('Initialization of Unity objects failed.')
        
        return observation, None
    
    def step(self, action):
        # Here we need to put the logic to execute a step in Unity
        # to do so, we sill select the action, and send it back to unity
        # Once the action has been executed, we need to receive the image
        # with the result of the image, we need to obtain the information
        # of the object. If it has already reached the objective, or if it has collide

        # We must return the observation, the reward, done and info

        # Actions 
        # 0 - forward
        # 1 - rotate left
        # 2 - rotate right
        response = self.unity_obj.request_action_to_unity(action)
        # Get the Image, coordinates and collisions from unity object
        observation, object_detected = self.__unity_image_formater(response.unity_image)
        object_coordinates = response.output
        object_collision = response.collision
        done = False
        # Get the current distance between A and B
        current_distance = self.__euclidean_distance(object_coordinates, self.objective_coordinates)
        # Increment the counter of measurements
        self.measruements_counter += 1

        # If there was a collision, it means a negative reward
        # and it has to stop this episode
        if object_collision is True:
            self.collisions += 1
            done = True
            reward = -2
        
        # If we have reached the time limit for every episode, it's a terminal state
        # and a small positive reward since it didn't collide but it didn't arrive to the objective
        elif self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            reward = 0.5

        # If we have reached the objective, it means a terminal state 
        elif (current_distance < self.delta):
            done = True
            reward = 2
        
        # If we are already in a terminal state, just return 
        if done:
            return observation, reward, done, None

        # If the agent is not correctly oriented, it means a negative reward
        if object_detected is False:
            done = False
            reward = -1
        
        # If the agent has the correct orientation
        else:
            done = False
            # If current distance between agent and object is bigger than the average distance
            # add a small negative reward
            if self.avg_distance < current_distance:
                reward = -0.5
            else:
                reward = 1

        # Update the average distance with the current distance
        self.avg_distance = ((self.avg_distance+current_distance) / self.measruements_counter)

        return observation, reward, done, None
    
    def __euclidean_distance(self, point_a, point_b) -> float:
        distance = math.sqrt((point_b.pos_x-point_a.pos_x)**2 + (point_b.pos_y-point_a.pos_y)**2 + (point_b.pos_z-point_a.pos_z)**2)

        return distance

    def __unity_image_formater(self, unity_img):
        # Convert image to cv_bridge
        cv_image = self._cv_bridge.imgmsg_to_cv2(unity_img, "bgr8")
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        image_rotated = cv2.rotate(image_gray, cv2.ROTATE_180)
        image_flipped = cv2.flip(image_rotated, 1)
        
        # Detect the object in the image
        blur_frame = cv2.GaussianBlur(image_flipped, (13, 13), 0)
        circles = cv2.HoughCircles(blur_frame, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=10, maxRadius=80)
        detected = False
        if circles is not None:
            detected = True
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cv2.circle(image_flipped, (circle[0], circle[1]), 1, (0,100,100), 3)
                cv2.circle(image_flipped, (circle[0], circle[1]), circle[2], (255, 0, 255), 3)
        

        img = cv2.resize(image_flipped, (84 , 84))
        img = img.reshape((1, 84, 84))

        return img, detected