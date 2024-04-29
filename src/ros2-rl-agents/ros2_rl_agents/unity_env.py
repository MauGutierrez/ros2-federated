import cv2
import math
import numpy as np
import random
import rclpy
import time

from cv_bridge import CvBridge
from collections import deque
from gym.spaces import Discrete
from my_interfaces.srv import PositionService
from my_interfaces.srv import InitUnityObjects
from rclpy.node import Node

import torch
import torchvision.transforms as transforms


SECONDS_PER_EPISODE = 45.0
DELTA_DISTANCE = 1.5000000
DELTA_ANGLE = 8.0

class Coordinates():
    
    def __init__(self, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z

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
        self.action_req.action = action
        self.future = self.actions_cli.call_async(self.action_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

class UnityEnv():
    def __init__(self, action_space: int, num_stack:int) -> None:
        self.unity_obj = UnityObject()
        self.objective_coordinates = Coordinates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.agent_coordinates = Coordinates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.collisions = 0
        self.delta = DELTA_DISTANCE
        self.action_space = Discrete(action_space)
        self._cv_bridge = CvBridge()
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

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
            observation = self.__unity_image_formater(response.unity_image)
            # Initial coordinates of the objective
            self.objective_coordinates.pos_x = response.objective.pos_x
            self.objective_coordinates.pos_y = response.objective.pos_y
            self.objective_coordinates.pos_z = response.objective.pos_z
            self.objective_coordinates.rot_x = response.objective.rot_x
            self.objective_coordinates.rot_y = response.objective.rot_y
            self.objective_coordinates.rot_z = response.objective.rot_z
            # Initial coordinates of the agent
            self.agent_coordinates.pos_x = response.agent.pos_x
            self.agent_coordinates.pos_y = response.agent.pos_y
            self.agent_coordinates.pos_z = response.agent.pos_z
            self.agent_coordinates.rot_x = response.agent.rot_x
            self.agent_coordinates.rot_y = response.agent.rot_y
            self.agent_coordinates.rot_z = response.agent.rot_z
            # Reset the number of collisions
            self.collisions = 0
        else:
            self.unity_obj.get_logger().warning('Initialization of Unity objects failed.')
            observation = []
        
        stacked_observations = np.array([self.frames.append(observation) for _ in range(self.num_stack)], dtype=np.float64)
        
        return stacked_observations, None
    
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
        observation = self.__unity_image_formater(response.unity_image)
        self.frames.append(observation)
        object_coordinates = response.output
        object_collision = response.collision
        # Get the current angle between the agent and the object
        orientation_angle = response.vision_angle
        done = False
        # Get the current distance between A and B
        current_distance = self.__euclidean_distance(object_coordinates, self.objective_coordinates)

        # If there was a collision, it means a negative reward
        # and it has to stop this episode
        if object_collision:
            self.collisions += 1
            done = True
            reward = -100

        # If we have reached the objective, it means a terminal state 
        elif current_distance < self.delta:
            done = True
            reward = 1
        
        if done:
            return np.array(self.frames, dtype=np.float64), reward, done, None    

        # This will help the agent to learn to rotate and see the object
        if orientation_angle < DELTA_ANGLE:
            done = False
            reward = 0.5

        else:
            done = False
            reward = -0.5

        # If we have reached the time limit for every episode, it's a terminal state
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return np.array(self.frames, dtype=np.float64), reward, done, None
    
    def __euclidean_distance(self, point_a, point_b) -> float:
        vector_a = np.array((point_a.pos_x, point_a.pos_y, point_a.pos_z))
        vector_b = np.array((point_b.pos_x, point_b.pos_y, point_b.pos_z))
        dist = np.linalg.norm(vector_a - vector_b)
        return dist

    def __unity_image_formater(self, unity_img):
        # Convert image to cv_bridge
        cv_image = self._cv_bridge.imgmsg_to_cv2(unity_img, "bgr8")
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        image_rotated = cv2.rotate(image_gray, cv2.ROTATE_180)
        image_flipped = cv2.flip(image_rotated, 1)
        img = cv2.resize(image_flipped, (84 , 84))
        img = img.reshape((84, 84, 1))

        return img