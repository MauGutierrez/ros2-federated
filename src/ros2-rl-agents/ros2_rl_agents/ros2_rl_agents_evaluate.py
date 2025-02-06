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
save_dir = Path('checkpoints') / NAME / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
checkpoint = Path('checkpoints_agents/agent_1/2025-02-02T02-41-42/ros_net_1.chkpt')
# checkpoint = None

OBSERVATION_SPACE = 6
ACTION_SPACE = 3
NUM_EPISODES = 800
TESTING_LOOP = 200
BATCH_SIZE = 64
SEED = 42
TESTING = True
logger = MetricLogger(save_dir, testing=TESTING)

def main():
    # Init ROS
    rclpy.init()

    # Load general settings saved in json file
    # settings = os.path.join(get_package_share_directory('ros2_rl_agents'), 'config/settings.json')

    # Setup UnityEnv environment
    env = UnityEnv(action_space=ACTION_SPACE, agent_name=NAME, n_steps=10)
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Setup Unity Agent
    agent = UnityAgent(agent_name=NAME, state_dim=OBSERVATION_SPACE, action_dim=n_actions, save_dir=save_dir, checkpoint=checkpoint, testing=TESTING)   
    
    agent.exploration_rate = agent.exploration_rate_min

    episodes = NUM_EPISODES

    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes+1):
        state, _ = env.reset()

        # Play the game!
        for i in range(TESTING_LOOP):
            action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)

            # agent.cache(state, next_state, action, reward, done)

            logger.log_step(reward, None, None)

            state = next_state

            if done:
                collision = info["collision"]
                goal = info["goal"]
                logger.log_episode(collision, goal)
                break

            if i == TESTING_LOOP - 1:
                logger.log_episode(collision=0, goal=0, not_completed=True)
                break
        

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=agent.exploration_rate,
                step=agent.curr_step
            )
        
        # 11. Update the exploration rate after every episode
        # agent.update_exploration_rate()
    

    # Explicity destroy nodes 
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()