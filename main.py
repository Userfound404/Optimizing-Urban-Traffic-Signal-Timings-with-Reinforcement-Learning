# main.py

import os
import sys
import traci
import random
import numpy as np
import torch
import logging
import datetime
import traceback
from agent import Agent, DQNModel  # Import from agent.py
from Environment import Environment  # Import from environment.py
from utils.visualization import Visualization  # Ensure this module is available
import matplotlib.pyplot as plt  # For visualization

def main():
    # Configure logging
    logging.basicConfig(filename='simulation.log', level=logging.INFO)

    # Define a boolean to toggle GUI visibility
    use_gui = True  # Set to True to see the simulation, False to run without GUI

    # Set up SUMO
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    sumo_binary = 'sumo-gui' if use_gui else 'sumo'

    # Path to the SUMO configuration file
    sumo_config = 'sumo_config/boston.sumocfg'  # Adjust the path as needed

    # Update SUMO command to use the configuration file
    sumo_cmd = [sumo_binary, '-c', sumo_config, '--start', '--no-step-log', 'true']

    # Define simulation parameters
    steps_per_episode = 30  # Adjust based on your simulation length
    num_episodes = 30
    total_simulation_steps = steps_per_episode * num_episodes

    # Create a directory to save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    visualization = Visualization(path=results_dir)

    # Start the SUMO simulation to access the network and get traffic light IDs
    traci.start(sumo_cmd)
    logging.info("SUMO simulation started.")
    tls = traci.trafficlight.getIDList()  # Get traffic light IDs from the network

    # Initialize agents
    agents = {}
    for agent_id in tls:
        logics = traci.trafficlight.getAllProgramLogics(agent_id)
        if not logics or len(logics[0].phases) == 0:
            print(f"Skipping traffic light {agent_id} due to no phases.")
            continue
        num_phases = len(logics[0].phases)
        action_space = list(range(num_phases))
        controlled_lanes = traci.trafficlight.getControlledLanes(agent_id)
        state_size = 2 * len(controlled_lanes)  # Adjust as needed
        model = DQNModel(input_dim=state_size, output_dim=num_phases)
        agents[agent_id] = Agent(
            tl_id=agent_id,
            action_space=action_space,
            state_size=state_size,
            model=model,
            neighbors=[],
            controlled_lanes=controlled_lanes
        )

    print(f"Initialized {len(agents)} agents.")

    # Initialize environment with traffic light IDs
    env = Environment(sumo_cmd=sumo_cmd, max_steps=total_simulation_steps, tl_ids=list(agents.keys()))

    # Metrics for visualization
    episode_rewards = {agent_id: [] for agent_id in agents}
    episode_losses = {agent_id: [] for agent_id in agents}
    episode_epsilons = {agent_id: [] for agent_id in agents}
    average_queue_lengths = []  # To store average queue lengths per episode

    # Training loop
    current_episode = 1
    steps_in_current_episode = 0

    try:
        while env.step < total_simulation_steps:
            if steps_in_current_episode == 0:
                logging.info(f"Starting episode {current_episode}/{num_episodes}")
                # Reset agent-specific variables for the new episode
                for agent in agents.values():
                    agent.old_state = None
                    agent.old_action = None
                # Reset episode metrics
                episode_reward = {agent_id: 0 for agent_id in agents}
                episode_loss = {agent_id: 0 for agent_id in agents}
                loss_count = {agent_id: 0 for agent_id in agents}
                # Initialize variables for average queue length
                total_queue_length = 0
                queue_length_samples = 0

            # For each agent, get state, choose action, and apply it
            for agent_id, agent in agents.items():
                state = agent.get_state(env)
                action = agent.choose_action(state)
                env.set_action(agent_id, action)
                reward = env.get_reward(agent_id)
                next_state = agent.get_state(env)
                agent.memory.push((state, action, reward, next_state, False))
                loss = agent.learn()
                agent.old_state = state
                agent.old_action = action

                # Collect metrics
                episode_reward[agent_id] += reward
                if loss is not None:
                    episode_loss[agent_id] += loss
                    loss_count[agent_id] += 1

                # Collect queue length data
                queue_length = env.get_total_queue_length(agent_id)
                total_queue_length += queue_length
                queue_length_samples += 1

            # Step the simulation
            env.step_simulation()
            steps_in_current_episode += 1

            # Check if episode is done
            if steps_in_current_episode >= steps_per_episode:
                logging.info(f"Episode {current_episode} completed.")
                # Compute average loss and average queue length per agent
                avg_queue_length = total_queue_length / queue_length_samples if queue_length_samples > 0 else 0
                average_queue_lengths.append(avg_queue_length)

                for agent_id in agents:
                    avg_loss = episode_loss[agent_id] / loss_count[agent_id] if loss_count[agent_id] > 0 else 0
                    # Print metrics per agent
                    print(f"Agent {agent_id} - Episode {current_episode}/{num_episodes}, Total Reward: {episode_reward[agent_id]}, Average Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
                    # Save metrics
                    episode_rewards[agent_id].append(episode_reward[agent_id])
                    episode_losses[agent_id].append(avg_loss)
                    episode_epsilons[agent_id].append(agent.epsilon)

                print(f"Average Queue Length in Episode {current_episode}: {avg_queue_length:.2f}")

                current_episode += 1
                steps_in_current_episode = 0
                if current_episode > num_episodes:
                    break  # Exit the loop if all episodes are completed

        # After training, save the trained models
        for agent_id, agent in agents.items():
            model_path = os.path.join(results_dir, f"trained_model_{agent_id}.pth")
            torch.save(agent.model.state_dict(), model_path)
        logging.info("Training completed and models saved.")

        # Generate visualization plots
        # Plot average queue length over episodes
        plt.figure()
        plt.plot(range(1, num_episodes + 1), average_queue_lengths, marker='o')
        plt.title('Average Queue Length Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Queue Length')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'average_queue_length.png'))
        plt.close()

        # Optionally, plot other metrics such as total reward or average loss per episode
        # Aggregate rewards and losses across agents if needed
        total_rewards_per_episode = [sum(episode_rewards[agent_id][i] for agent_id in agents) for i in range(num_episodes)]
        average_losses_per_episode = [np.mean([episode_losses[agent_id][i] for agent_id in agents if episode_losses[agent_id][i] > 0]) for i in range(num_episodes)]

        # Plot total rewards over episodes
        plt.figure()
        plt.plot(range(1, num_episodes + 1), total_rewards_per_episode, marker='o')
        plt.title('Total Reward Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'total_reward.png'))
        plt.close()

        # Plot average loss over episodes
        plt.figure()
        plt.plot(range(1, num_episodes + 1), average_losses_per_episode, marker='o')
        plt.title('Average Loss Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'average_loss.png'))
        plt.close()

    except Exception as e:
        logging.error(f"An error occurred during simulation: {e}")
        traceback.print_exc()
        print(f"An error occurred: {e}")

    finally:
        traci.close()
        logging.info("SUMO simulation closed.")

if __name__ == "__main__":
    main()
