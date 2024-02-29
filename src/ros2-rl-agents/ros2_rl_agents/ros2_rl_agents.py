import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import rclpy
import time

from rclpy.node import Node
from collections import namedtuple
from itertools import count
from ros2_rl_agents.DQN import DQN
from ros2_rl_agents.ReplayMemory import ReplayMemory
from ros2_rl_agents.helper_functions import select_action, optimize_model, plot_durations, serialize_array, deserialize_array
from ament_index_python.packages import get_package_share_directory

from example_interfaces.srv import Trigger
from my_interfaces.srv import SendLocalWeights

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TRANSITION = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class FederatedAgent(Node):
    def __init__(self):
        super().__init__('federated_agent')

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
    
def main():
    # Init ROS
    rclpy.init()

    # Init client object to handle communication with server
    agent = FederatedAgent()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up Matplotlib
    plt.ion()

    # Load general settings saved in json file
    settings = os.path.join(get_package_share_directory('ros2_rl_agents'), 'config/settings.json')

    # Setup CartPole environment
    env = gym.make("CartPole-v1")
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    episode_durations = []
    
    with open(settings) as fp:
        content = json.load(fp)
        # Get hyperparameters from settings file
        hyperparameters = content['model_hyperparameters']

        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        batch_size = hyperparameters['BATCH_SIZE'] 
        gamma = hyperparameters['GAMMA']
        eps_start = hyperparameters['EPS_START']
        eps_end = hyperparameters['EPS_END']
        eps_decay = hyperparameters['EPS_DECAY']
        tau = hyperparameters['TAU']
        lr = hyperparameters['LR']

        if torch.cuda.is_available():
            num_episodes = 300
        else:
            num_episodes = 300

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
        memory = ReplayMemory(10000, TRANSITION)

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                # Select an action
                action = select_action(state, eps_end, eps_start, eps_decay, policy_net, env, device)
                # Execute the action
                observation, reward, terminated, truncated, _ = env.step(action.item())
                # Convert the reward in a vector
                reward = torch.tensor([reward], device=device)
                # Determine the duration of the episode
                done = terminated or truncated

                # If the game has terminated, next step is set to None (Last step)
                if terminated:
                    next_state = None
                # Else, next step is the observation
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(memory, TRANSITION, device, optimizer, policy_net, target_net, batch_size, gamma)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations(episode_durations, show_result=True)
                    break
            
            # Update the weights at the end of every episode
            
            # Convert pytorch tensor to numpy array
            serialized_policy_net_weights = serialize_array(policy_net.parameters())
            
            # Store numpy array in json message to be sent to the server 
            message = {
                "client": "agent_1",
                "weights": serialized_policy_net_weights
            }

            message = json.dumps(message)
            
            # Send a request to update the local weights
            response = agent.add_local_weights_request(message)

            # Send a request to the federated server
            if response.success == True:
                while rclpy.ok():
                    request = agent.get_new_weights_request()
                    if request.success is False:
                        print(request.message)
                        time.sleep(1)
                    else:
                        # Set new local weights that have been received from federated server
                        data = json.loads(request.content)
                        new_weights = data["weights"]
                        new_weights = deserialize_array(new_weights, device)
                        
                        # Assign new weights to policy net
                        for param, weight in zip(policy_net.parameters(), new_weights):
                            param.data = weight

                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        target_net_state_dict = target_net.state_dict()
                        policy_net_state_dict = policy_net.state_dict()
                        
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
                        target_net.load_state_dict(target_net_state_dict)
                        break

        print('Complete')
        plot_durations(episode_durations, show_result=True)
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()