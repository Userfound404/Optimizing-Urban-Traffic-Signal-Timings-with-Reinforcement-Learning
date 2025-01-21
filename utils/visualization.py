# utils/visualization.py
import matplotlib.pyplot as plt
import os
import numpy as np

class Visualization:
    def __init__(self, path):
        self.path = path

    def plot_rewards(self, rewards):
        plt.figure()
        plt.plot(rewards, label='Rewards')
        # Calculate moving average
        moving_avg = self._moving_average(rewards, window_size=10)
        plt.plot(moving_avg, label='Moving Average (10 episodes)', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.savefig(os.path.join(self.path, 'rewards.png'))
        plt.close()

    def plot_losses(self, losses):
        plt.figure()
        plt.plot(losses, label='Losses')
        # Calculate moving average
        moving_avg = self._moving_average(losses, window_size=10)
        plt.plot(moving_avg, label='Moving Average (10 episodes)', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per Episode')
        plt.legend()
        plt.savefig(os.path.join(self.path, 'losses.png'))
        plt.close()

    def plot_epsilons(self, epsilons):
        plt.figure()
        plt.plot(epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay over Episodes')
        plt.savefig(os.path.join(self.path, 'epsilon_decay.png'))
        plt.close()

    def _moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
