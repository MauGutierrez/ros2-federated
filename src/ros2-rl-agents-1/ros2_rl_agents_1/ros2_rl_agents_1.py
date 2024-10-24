import datetime
import json
import os
import rclpy
import torch

from pathlib import Path
from ros2_rl_agents_1.unity_env import UnityEnv
from ros2_rl_agents_1.unity_agent import UnityAgent
from ros2_rl_agents_1.metrics import MetricLogger
from ament_index_python.packages import get_package_share_directory

NAME = "agent_2"
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
save_dir = Path('checkpoints') / NAME / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
checkpoint = Path('checkpoints/agent_1/2024-10-06T14-43-07/ros_net_3.chkpt')
logger = MetricLogger(save_dir)
SKIP_FRAMES = 3
FRAME_STACK = 4
ACTION_SPACE = 3
NUM_EPISODES = 800
HEIGHT = 84
WIDTH = 84


def main():
    # Init ROS
    rclpy.init()

    # Load general settings saved in json file
    # settings = os.path.join(get_package_share_directory('ros2_rl_agents_1'), 'config/settings.json')

    # Setup UnityEnv environment
    env = UnityEnv(action_space=ACTION_SPACE, num_stack=FRAME_STACK, height=HEIGHT, width=WIDTH, agent_name=NAME)
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Setup Unity Agent
    agent = UnityAgent(agent_name=NAME, state_dim=(FRAME_STACK, HEIGHT, WIDTH), action_dim=n_actions, save_dir=save_dir, checkpoint=checkpoint)  
    
    # Add the agent to the federated network
    agent.add_agent_to_federated_network()

    episodes = NUM_EPISODES

    ### for Loop that train the model num_episodes times by playing the game
    for e in range(episodes):
        state, _ = env.reset()

        # Play the game!
        while True:

            # 3. Show environment (the visual) [WIP]
            # env.render()
            
            # 4. Run agent on the state
            action = agent.act(state)
            
            # 5. Agent performs action
            next_state, reward, done, _ = env.step(action)

            # 6. Remember
            agent.cache(state, next_state, action, reward, done)

            # 7. Learn
            q, loss = agent.learn()

            # 8. Logging
            logger.log_step(reward, loss, q)

            # 9. Update state
            state = next_state

            # 10. Check if end of game
            if done:
                break
        
        logger.log_episode()

        # 11. Update the exploration rate after every episode
        agent.update_exploration_rate()
        
        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=agent.exploration_rate,
                step=agent.curr_step
            )
            # 12. Update the optimizer with the average
            agent.update_optimizer()
    
    # 13. Remove agent from network
    agent.remove_agent_from_federated_network()

    # 14. Save the model
    agent.save()

    # Explicity destroy nodes 
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()