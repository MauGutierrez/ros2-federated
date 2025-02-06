import json
import rclpy
import random, numpy as np
import torch

from collections import deque
from example_interfaces.srv import Trigger
from my_interfaces.srv import LocalValues
from my_interfaces.srv import ConfigureAgent
from pathlib import Path
from rclpy.node import Node
from ros2_rl_agents_1.neural_net import Net
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup


class FederatedConnection(Node):
    def __init__(self, agent_name):
        super().__init__('federated_' + agent_name)

        wait_cb = MutuallyExclusiveCallbackGroup()
        get_values_cb = MutuallyExclusiveCallbackGroup()
        # Client to request the addition of the weights
        self.loss_cli = self.create_client(LocalValues, "add_to_global")
        while not self.loss_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.loss_req = LocalValues.Request()

        # Client to request the addition of a new agent in the network
        self.add_agent_cli = self.create_client(ConfigureAgent, "add_agent")
        while not self.add_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.add_agent_req = ConfigureAgent.Request()

        self.wait_agent_cli = self.create_client(ConfigureAgent, "wait_for_all_agents", callback_group=wait_cb)
        while not self.wait_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.wait_agent_req = ConfigureAgent.Request()

        # Client to request the new global value
        self.update_cli = self.create_client(LocalValues, "get_global_value", callback_group=get_values_cb)
        while not self.update_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.update_req = LocalValues.Request()

        # Client to request the remove of an agent in the network
        self.remove_agent_cli = self.create_client(ConfigureAgent, "remove_agent")
        while not self.remove_agent_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.remove_agent_req = ConfigureAgent.Request()

    def add_agent_to_network(self, name):
        self.add_agent_req.data = name
        self.future_add_agent = self.add_agent_cli.call_async(self.add_agent_req)
        rclpy.spin_until_future_complete(self, self.future_add_agent)
        return self.future_add_agent.result()
    
    def add_local_weights_request(self, message):
        self.loss_req.data = message
        self.future_add_local = self.loss_cli.call_async(self.loss_req)
        rclpy.spin_until_future_complete(self, self.future_add_local)
        return self.future_add_local.result()

    def wait_for_all_agents(self):
        self.future_wait = self.wait_agent_cli.call_async(self.wait_agent_req)
        rclpy.spin_until_future_complete(self, self.future_wait, timeout_sec=5.0)
        return self.future_wait.result()
    
    def get_new_weights_request(self, message):
        self.update_req.data = message
        self.future_get_global = self.update_cli.call_async(self.update_req)
        rclpy.spin_until_future_complete(self, self.future_get_global, timeout_sec=5.0)
        return self.future_get_global.result()
    
    def remove_agent_from_network(self, name):
        self.remove_agent_req.data = name
        self.future_remove = self.remove_agent_cli.call_async(self.remove_agent_req)
        rclpy.spin_until_future_complete(self, self.future_remove)
        return self.future_remove.result()

class UnityAgent:
    def __init__(self, agent_name, state_dim, action_dim, save_dir=None, checkpoint=None, testing=None):
        self.agent_name = agent_name
        if testing is None:
            self.federated_connection = FederatedConnection(self.agent_name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e4   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.device = 'cpu'

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = Net(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.device = 'cuda'
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()
        self.start_optimizer = False


    def update_exploration_rate(self):
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

    
    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # if self.curr_step % self.save_every == 0:
        #     self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None
        
        self.start_optimizer = True

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"ros_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"Net saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
    


    def update_optimizer(self):
        if self.start_optimizer is False:
            return None
        
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        batch_gradients = self.accumulate_gradients()
        
        message = {
            "client": self.agent_name,
            "local_value": batch_gradients
        }

        message = json.dumps(message)

        response = self.federated_connection.add_local_weights_request(message)
        if response.success is False:
            print("Failure")
        else:
            while rclpy.ok():
                response = self.federated_connection.get_new_weights_request(self.agent_name)
                if response is None:
                    continue
                elif response.success is True:
                    # Update loss with new global value
                    # and set torch.no_grad() to keep the same grad_fn
                    with torch.no_grad():
                        json_data = json.loads(response.global_value)
                        new_global = json_data["weights"]
                        new_loss = [torch.tensor(vector).float() for vector in new_global]
                        for param, grad in zip(self.net.online.parameters(), new_loss):
                            grad = grad.to(self.device)
                            param.grad = grad
                        self.optimizer.step()
                    break
            
            while rclpy.ok():
                response = self.federated_connection.wait_for_all_agents()
                if response is None:
                    continue
                elif response.success is True:
                    break    
    
    def accumulate_gradients(self):
        self.optimizer.zero_grad()
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.loss_fn(td_est, td_tgt)
        loss.backward()

        return [param.grad.clone().detach().cpu().numpy().tolist() for param in self.net.online.parameters()]


    def add_agent_to_federated_network(self):
        response = self.federated_connection.add_agent_to_network(self.agent_name)
        if response.success is False:
            exit(0)
    
    def remove_agent_from_federated_network(self):
        response = self.federated_connection.remove_agent_from_network(self.agent_name)
        if response.success is False:
            exit(0)