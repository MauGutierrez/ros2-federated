import datetime
import json
import os
import rclpy
import torch

from rclpy.node import Node
from collections import namedtuple
from pathlib import Path
from ros2_rl_agents.neural_net import Net
from ros2_rl_agents.unity_env import UnityEnv
from ros2_rl_agents.unity_agent import UnityAgent
from ros2_rl_agents.metrics import MetricLogger

from ament_index_python.packages import get_package_share_directory

from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
logger = MetricLogger(save_dir)

# class FederatedAgent(Node):
#     def __init__(self):
#         super().__init__('federated_agent')

#         # Client to request the initial model
#         self.model_cli = self.create_client(Trigger, "download_model")
#         while not self.model_cli.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('service not available, waiting again...')
#         self.model_req = Trigger.Request()

#         # Client to request the addition of the weights
#         self.weights_cli = self.create_client(SendLocalWeights, "add_weights")
#         while not self.weights_cli.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('service not available, waiting again...')
#         self.weights_req = SendLocalWeights.Request()

#         # Client to request the update of the weights
#         self.update_cli = self.create_client(SendLocalWeights, "get_weights")
#         while not self.update_cli.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('service not available, waiting again...')
#         self.update_req = SendLocalWeights.Request()

#     def download_model_request(self):
#         self.future = self.model_cli.call_async(self.model_req)
#         rclpy.spin_until_future_complete(self, self.future)
#         return self.future.result()
    
#     def add_local_weights_request(self, message):
#         self.weights_req.data = message
#         self.future = self.weights_cli.call_async(self.weights_req)
#         rclpy.spin_until_future_complete(self, self.future)
#         return self.future.result()
    
#     def get_new_weights_request(self):
#         self.future = self.update_cli.call_async(self.update_req)
#         rclpy.spin_until_future_complete(self, self.future)
#         return self.future.result()
    
def main():
    # Init ROS
    rclpy.init()

    # Init client object to handle communication with server
    # agent = FederatedAgent()

    # Load general settings saved in json file
    settings = os.path.join(get_package_share_directory('ros2_rl_agents'), 'config/settings.json')

    # Setup UnityEnv environment
    env = UnityEnv()
    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Setup Unity Agent
    agent = UnityAgent(state_dim=(1, 84, 84), action_dim=n_actions, save_dir=save_dir, checkpoint=None)  
    
    episodes = 300

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

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=agent.exploration_rate,
                step=agent.curr_step
            )
    
    # Explicity destroy nodes 
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()