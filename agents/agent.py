# agent.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import traci  # Import traci to interact with SUMO

class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        # Define your neural network architecture here
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_dim)

        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        """Saves an experience to the replay memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly samples a batch of experiences."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, tl_id, action_space, state_size, model, neighbors):
        self.tl_id = tl_id
        self.action_space = action_space
        self.state_size = state_size
        self.model = model  # Each agent has its own model
        self.memory = ReplayMemory(50000)  # Each agent has its own replay memory
        self.neighbors = neighbors
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        self.batch_size = 64  # Adjust as needed
        self.old_state = None
        self.old_action = None

    def get_state(self, env):
        """
        Get the current state for this agent from the environment.
        """
        # Use the queue length and waiting time for each controlled lane
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        state = []
        for lane in controlled_lanes:
            # Get the number of vehicles halting (speed below 0.1 m/s)
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            # Get the total waiting time on this lane
            waiting_time = traci.lane.getWaitingTime(lane)
            state.extend([queue_length, waiting_time])
        # Convert state to numpy array
        state = np.array(state, dtype=np.float32)
        # Ensure state size matches expected size
        if len(state) < self.state_size:
            # Pad with zeros if necessary
            padding = np.zeros(self.state_size - len(state), dtype=np.float32)
            state = np.concatenate((state, padding))
        elif len(state) > self.state_size:
            # Truncate if necessary
            state = state[:self.state_size]
        # Print state shape for debugging
        print(f"Agent {self.tl_id} state shape: {state.shape}")
        return state

    def choose_action(self, state):
        """
        Choose an action based on the current state and epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            # Explore: choose a random action
            action = random.choice(self.action_space)
        else:
            # Exploit: choose the action with max Q-value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def learn(self):
        """
        Learn from a batch of experiences.
        """
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples to learn from
        minibatch = self.memory.sample(self.batch_size)

        # Convert lists of experiences into numpy arrays
        try:
            batch_states = np.array([experience[0] for experience in minibatch], dtype=np.float32)
            batch_actions = np.array([experience[1] for experience in minibatch], dtype=np.int64)
            batch_rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
            batch_next_states = np.array([experience[3] for experience in minibatch], dtype=np.float32)
            batch_dones = np.array([experience[4] for experience in minibatch], dtype=np.float32)
        except Exception as e:
            print(f"Error converting minibatch to arrays: {e}")
            return None

        # Convert numpy arrays to tensors
        states = torch.from_numpy(batch_states)
        actions = torch.from_numpy(batch_actions)
        rewards = torch.from_numpy(batch_rewards)
        next_states = torch.from_numpy(batch_next_states)
        dones = torch.from_numpy(batch_dones)

        # Compute target Q-values
        with torch.no_grad():
            target_q_values = rewards + self.gamma * torch.max(self.model(next_states), dim=1)[0] * (1 - dones)

        # Compute current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Return loss value for logging
        return loss.item()
