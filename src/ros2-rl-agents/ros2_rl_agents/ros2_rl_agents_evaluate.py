import datetime
import json
import numpy as np
import os
import random
import rclpy
import torch

from pathlib import Path
from ros2_rl_agents.unity_env import UnityEnv
from ros2_rl_agents.unity_agent import UnityAgent
from ros2_rl_agents.metrics import MetricLogger
from ament_index_python.packages import get_package_share_directory

NAME = "agent_1"
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


OBSERVATION_SPACE = 7
ACTION_SPACE = 3
NUM_EPISODES = 200
# TESTING_LOOP = 200
BATCH_SIZE = 64
SEED = 42
TESTING = True
# checkpoint = Path('checkpoints_train/agent_1/2026-02-05T01-25-27/ros_net_1.chkpt')

def create_checkpoints_folder(agent_name: str):
    save_dir = Path('checkpoints') / agent_name / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    return save_dir

def main():
    # Init ROS
    rclpy.init()

    settings_path = os.path.join(get_package_share_directory('ros2_rl_agents'), 'config/settings.json')
    
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    
    # Sync or Async mode    
    connection_mode = settings["connection_mode"]

    # Setup UnityEnv environment
    env = UnityEnv(action_space=ACTION_SPACE, n_steps=20, testing=TESTING)
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    agent_name = env.agent_name
    checkpoint = Path(env.checkpoint)

    save_dir = create_checkpoints_folder(agent_name)
    logger = MetricLogger(save_dir)

    # Setup Unity Agent
    agent = UnityAgent(agent_name=NAME, state_dim=OBSERVATION_SPACE, action_dim=n_actions, connection_mode=connection_mode, save_dir=save_dir, checkpoint=checkpoint, testing=TESTING)   
    
    agent.exploration_rate = agent.exploration_rate_min

    episodes = NUM_EPISODES

    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes):
        state, _ = env.reset()
        collision = 0
        goal = 0
        not_completed = False

        # Play the game!
        # for i in range(TESTING_LOOP):
        while True:
            action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)

            # agent.cache(state, next_state, action, reward, done)

            logger.log_step(reward, None, None)

            state = next_state

            if done:
                collision = info["collision"]
                goal = info["goal"]
                break

            # if i == TESTING_LOOP - 1:
            #     not_completed = True
            #     break
        

        logger.log_raw(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step,
            collision=collision,
            goal=goal,
            not_completed=not_completed
        )
        
        # 11. Update the exploration rate after every episode
        # agent.update_exploration_rate()
    

    # Explicity destroy nodes 
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()