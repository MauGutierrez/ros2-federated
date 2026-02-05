import cv2
import math
import numpy as np
import random
import requests
import rclpy
import time
import array as arr
import json

from cv_bridge import CvBridge
from collections import deque
from gym.spaces import Discrete
# from my_interfaces.srv import PositionService
# from my_interfaces.srv import InitUnityObjects
from rclpy.node import Node
from PIL import Image

import torch
import torchvision.transforms as transforms

from dataclasses import dataclass
from typing import Optional

@dataclass
class Coordinates:
    pos_x: float
    pos_y: float
    pos_z: float
    rot_x: float
    rot_y: float
    rot_z: float
    rot_w: float

@dataclass
class UnityData:
    agent_coords: Coordinates
    target_coords: Optional[Coordinates]  # optional because of `omitempty` in Go
    collision: bool
    success: bool
    vision_angle: float

@dataclass
class UnityResponse:
    agent_id: str
    result: int
    data: UnityData



# SECONDS_PER_EPISODE = 15.0
DELTA_DISTANCE = 2.0000
DELTA_ANGLE = 10.0

# class Coordinates():
    
#     def __init__(self, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z) -> None:
#         self.pos_x = pos_x
#         self.pos_y = pos_y
#         self.pos_z = pos_z
#         self.rot_x = rot_x
#         self.rot_y = rot_y
#         self.rot_z = rot_z

class UnityNetwork():
    def __init__(self, agent_name):
        self.agent_id = agent_name
        self.url = "http://192.168.68.107:8080/ros"

    def format_response(self, response):
        data = json.loads(response)

        # Convert dicts into dataclasses 
        agent_coords = Coordinates(**data["data"]["agent_coords"]) 
        target_coords = Coordinates(**data["data"]["target_coords"]) 
        unity_data = UnityData( 
            agent_coords=agent_coords, 
            target_coords=target_coords, 
            collision=data["data"]["collision"], 
            success=data["data"]["success"], 
            vision_angle=data["data"]["vision_angle"] 
        ) 
        
        payload = UnityResponse( 
            agent_id=data["agent_id"], 
            result=data["result"], 
            data=unity_data 
        )

        return payload

    def request_init_unity_objects(self):
        payload = {
            "agent_id": self.agent_id,
            "task": "INIT"
        }

        resp = None
        try:
            resp = requests.post(
                self.url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10
            )

            resp = self.format_response(resp.text)
        except Exception as e:
            print("Request Failed")
            # self.get_logger().error(f"Request failed: {e}")
        
        return resp
    
    def request_action_to_unity(self, action):
        payload = {
            "agent_id": self.agent_id,
            "task": "MOVE",
            "data": {"action": action}
        }

        resp = None
        try:
            resp = requests.post(
                self.url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10
            )

            resp = self.format_response(resp.text)

        except Exception as e:
            print("Request Failed")
            # self.get_logger().error(f"Request failed: " {e})
        
        return resp


class UnityEnv(Node):
    def __init__(self, action_space: int, n_steps: int) -> None:
        super().__init__('federated_agent')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('agent_name', rclpy.Parameter.Type.STRING),
                ('checkpoint', rclpy.Parameter.Type.STRING)
            ]
        )

        self.agent_name = self.get_parameter('agent_name').value
        self.checkpoint = self.get_parameter('checkpoint').value
        self.unity_obj = UnityNetwork(self.agent_name)
        # self.objective_coordinates = Coordinates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # self.agent_coordinates = Coordinates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
        if response.data.success is True:
            # Start counting the time of an episode
            # self.episode_start = time.time()
            # Environment observation
            # observation = self.__unity_image_formater(response.unity_image)
            # Initial coordinates of the objective
            # self.objective_coordinates.pos_x = response.objective.pos_x
            # self.objective_coordinates.pos_y = response.objective.pos_y
            # self.objective_coordinates.pos_z = response.objective.pos_z
            # self.objective_coordinates.rot_x = response.objective.rot_x
            # self.objective_coordinates.rot_y = response.objective.rot_y
            # self.objective_coordinates.rot_z = response.objective.rot_z
            # Initial coordinates of the agent
            # self.agent_coordinates.pos_x = response.agent.pos_x
            # self.agent_coordinates.pos_y = response.agent.pos_y
            # self.agent_coordinates.pos_z = response.agent.pos_z
            # self.agent_coordinates.rot_x = response.agent.rot_x
            # self.agent_coordinates.rot_y = response.agent.rot_y
            # self.agent_coordinates.rot_z = response.agent.rot_z
            # Get the initial angle between the agent and the objective
            self.objective_coordinates = response.data.target_coords
            self.initial_angle = response.data.vision_angle
            self.initial_distance = self.__euclidean_distance(response.data.agent_coords, self.objective_coordinates)
            self.collisions = 0

        else:
            print("Problem with request.")
            # self.get_logger().warning('Initialization of Unity objects failed.')
            # observation = []
        
        # for _ in range(self.num_stack):
        #     self.frames.append(observation)
        
        # stacked_observations = np.array(self.frames, dtype=np.float64, copy=True)
        return np.array([
            response.data.agent_coords.pos_x, response.data.agent_coords.pos_z, 
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
        object_coordinates = response.data.agent_coords
        object_collision = response.data.collision
        # Get the current angle between the agent and the object
        current_angle = response.data.vision_angle
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