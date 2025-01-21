import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        """
        Save an experience to memory.
        """
        self.memory.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
