import cv2
import math
import numpy as np
import random
import rclpy
import time
import array as arr

from cv_bridge import CvBridge
from collections import deque
from gym.spaces import Discrete
from my_interfaces.srv import PositionService
from my_interfaces.srv import InitUnityObjects
from rclpy.node import Node
from PIL import Image

import torch
import torchvision.transforms as transforms


# SECONDS_PER_EPISODE = 15.0
DELTA_DISTANCE = 2.0000
DELTA_ANGLE = 10.0

class Coordinates():
    
    def __init__(self, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z

class UnityObject(Node):

    def __init__(self, agent_name):
        super().__init__('unity_object_' + agent_name)
        self.init_cli = self.create_client(InitUnityObjects, 'init_unity_objects_1')
        while not self.init_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.inital_req = InitUnityObjects.Request()

        self.actions_cli = self.create_client(PositionService, 'move_unity_object_1')
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
    def __init__(self, action_space: int, agent_name:str, n_steps: int) -> None:
        self.unity_obj = UnityObject(agent_name)
        self.objective_coordinates = Coordinates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.agent_coordinates = Coordinates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.collisions = 0
        self.action_space = Discrete(action_space)
        self._cv_bridge = CvBridge()
        # self.num_stack = num_stack
        # self.frames = deque(maxlen=num_stack)
        self.distances = arr.array('d', [])
        self.angles = arr.array('d', [])
        # self.width = width
        # self.height = height
        self.n_steps = n_steps
        self.initial_distance = 0
        self.initial_angle = 0
        self.steps = 0

    def reset(self):
        # Restart the initial coordinates of the object
        # Restart the initial coordinates of the agent
        # Restart the flag to detect collisons
        # Get the initial observation
        response = self.unity_obj.request_init_unity_objects()
        # Get the initial observation Image
        if response.success is True:
            # Start counting the time of an episode
            # self.episode_start = time.time()
            # Environment observation
            # observation = self.__unity_image_formater(response.unity_image)
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
            # Get the initial angle between the agent and the objective
            self.initial_angle = response.vision_angle
            self.initial_distance = self.__euclidean_distance(self.agent_coordinates, self.objective_coordinates)
            self.collisions = 0

        else:
            self.unity_obj.get_logger().warning('Initialization of Unity objects failed.')
            # observation = []
        
        # for _ in range(self.num_stack):
        #     self.frames.append(observation)
        
        # stacked_observations = np.array(self.frames, dtype=np.float64, copy=True)
        return np.array([
            self.agent_coordinates.pos_x, self.agent_coordinates.pos_z, 
            self.objective_coordinates.pos_x, self.objective_coordinates.pos_z, 
            self.initial_angle, self.initial_distance, 0], dtype=np.float64), None
    
    
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
        # observation = self.__unity_image_formater(response.unity_image)
        # self.frames.append(observation)
        object_coordinates = response.output
        object_collision = response.collision
        # Get the current angle between the agent and the object
        current_angle = response.vision_angle
        current_distance = self.__euclidean_distance(object_coordinates, self.objective_coordinates)

        done = False
        goal = 0
        collision = 0
        if current_distance < DELTA_DISTANCE:
            done = True
            reward = 1
            goal = 1
        # If there was a collision, it means a negative reward
        # and it has to stop this episode
        elif object_collision:
            self.collisions += 1
            collision = 1
            done = True
            reward = -1
        elif self.initial_distance > current_distance and self.initial_angle > current_angle:
            reward = 0.1
        else:
            reward = -0.01
        
        info = {
            "collision": collision,
            "goal": goal
        }

        # return np.array(self.frames, dtype=np.float64), reward, done, None
        return np.array([
            object_coordinates.pos_x, object_coordinates.pos_z, 
            self.objective_coordinates.pos_x, self.objective_coordinates.pos_z, 
            current_angle, current_distance, goal], dtype=np.float64), reward, done, info
    
    def __euclidean_distance(self, point_a, point_b) -> float:
        vector_a = np.array((point_a.pos_x, 0.0, point_a.pos_z))
        vector_b = np.array((point_b.pos_x, 0.0, point_b.pos_z))
        dist = np.linalg.norm(vector_a - vector_b)
        return dist

    def __unity_image_formater(self, unity_img):
        # Convert image to cv_bridge
        cv_image = self._cv_bridge.imgmsg_to_cv2(unity_img, "bgr8")
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        image_rotated = cv2.rotate(image_gray, cv2.ROTATE_180)
        image_flipped = cv2.flip(image_rotated, 1)
        img = cv2.resize(image_flipped, (self.height, self.width))

        return img