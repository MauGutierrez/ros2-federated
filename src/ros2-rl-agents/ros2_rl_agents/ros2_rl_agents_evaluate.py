import datetime
import json
import os
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
checkpoint = Path('checkpoints/agent_1/2024-10-09T01-01-38/ros_net_10.chkpt')
logger = MetricLogger(save_dir)
SKIP_FRAMES = 3
FRAME_STACK = 4
ACTION_SPACE = 3
NUM_EPISODES = 100
HEIGHT = 84
WIDTH = 84
TESTING = True


def main():
    # Init ROS
    rclpy.init()

    # Load general settings saved in json file
    # settings = os.path.join(get_package_share_directory('ros2_rl_agents'), 'config/settings.json')

    # Setup UnityEnv environment
    env = UnityEnv(action_space=ACTION_SPACE, num_stack=FRAME_STACK, height=HEIGHT, width=WIDTH, agent_name=NAME)
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Setup Unity Agent
    agent = UnityAgent(agent_name=NAME, state_dim=(FRAME_STACK, HEIGHT, WIDTH), action_dim=n_actions, save_dir=save_dir, checkpoint=checkpoint, testing=TESTING)   
    
    agent.exploration_rate = agent.exploration_rate_min

    episodes = NUM_EPISODES

    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes):
        state, _ = env.reset()

        # Play the game!
        while True:
            action = agent.act(state)
            
            next_state, reward, done, _ = env.step(action)

            agent.cache(state, next_state, action, reward, done)

            logger.log_step(reward, None, None)

            state = next_state

            if done:
                break
        
        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=agent.exploration_rate,
                step=agent.curr_step
            )
        
        # 11. Update the exploration rate after every episode
        agent.update_exploration_rate()
    

    # Explicity destroy nodes 
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()