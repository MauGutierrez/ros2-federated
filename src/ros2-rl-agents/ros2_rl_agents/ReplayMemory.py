from collections import deque
import random

class ReplayMemory(object):
    def __init__(self, capacity, transition):
        self.memory = deque([], maxlen=capacity)
        self.transition = transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)