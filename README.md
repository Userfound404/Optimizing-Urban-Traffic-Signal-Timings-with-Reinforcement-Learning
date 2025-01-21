# Optimizing-Urban-Traffic-Signal-Timings-with-Reinforcement-Learning
Code for the publication of Optimizing Urban Traffic Signal Timings with Reinforcement Learning: A Comprehensive Simulation Study

This project focuses on optimizing urban traffic management using multi-agent deep reinforcement learning (RL). By employing agents to control traffic lights dynamically, we aim to reduce congestion, minimize waiting times, and improve traffic flow in a simulated environment created with SUMO (Simulation of Urban Mobility).

**This work has been presented at AISC 2024, University of North Carolina Greensboro.**

## Table of Contents
- Introduction
- Motivation
- How It Works
- Features
- Project Structure
- Requirements
- Setup and Usage
- Results
- Future Work
- Contributing
- License

## Introduction

Traffic congestion is a global challenge, resulting in economic losses, wasted fuel, and environmental pollution. Traditional traffic light systems rely on static timings that fail to adapt to changing traffic patterns. This project leverages Deep Q-Learning (DQL) and multi-agent systems to create adaptive traffic light controllers capable of real-time decision-making.

Each traffic light is controlled by an independent agent, which learns to optimize signal timings based on traffic flow data. By training in a simulated environment, the agents gradually improve their decision-making to reduce overall congestion.

## Motivation

The rise of smart cities and IoT (Internet of Things) has highlighted the need for intelligent traffic systems. Reinforcement learning provides a powerful approach to learn policies that optimize real-world systems, such as traffic lights. This project aims to:

- Showcase the potential of multi-agent reinforcement learning for traffic control.
- Provide a scalable and modular framework for traffic simulation and optimization.
- Encourage further research in applying AI to urban mobility problems.

## How It Works

Simulation Environment: The project uses SUMO, a microscopic traffic simulator, to create a realistic urban traffic environment. SUMO generates vehicles, road networks, and traffic signals based on configuration files.

Reinforcement Learning Setup:

- Agents: Each traffic light intersection is controlled by a reinforcement learning agent.
- State Space: Includes the number of vehicles, average speeds, and waiting times at each signal.
- Action Space: Represents possible traffic light phase changes (e.g., switching between green and red signals).
- Reward Function: Encourages actions that minimize waiting times and queue lengths.
- Training Process: Agents interact with the environment, collect state information, and take actions to maximize long-term rewards. The training is performed using Deep Q-Learning.
- Visualization: After training, the results are visualized to analyze the impact of the learned policies on traffic metrics such as queue lengths and average waiting times.

## Features

- Multi-agent system for decentralized traffic control.
- Deep Q-Learning implementation using PyTorch.
- Real-time simulation with SUMO integration.
- Visualization of training metrics and performance results.
- Modular and scalable design to extend to larger networks.

## Project Structure

multi_agent_traffic/

├── agent.py                  # Agent logic for Deep Q-Learning

├── environment.py            # SUMO environment for traffic simulation

├── main.py                   # Main script to train and evaluate agents

├── utils/

│   ├── __init__.py           # Initialization file for utilities

│   ├── visualization.py      # Code to generate plots and save results

├── sumo_config/              # SUMO configuration files

│   ├── boston.sumocfg        # SUMO simulation configuration

│   ├── osm.net.xml.gz        # Compressed SUMO network file

│   ├── osm.poly.xml.gz       # Compressed SUMO polygon file

└── results/                  # Directory for plots, metrics, and logs

## Requirements
System Requirements:
- Python 3.8+
- SUMO 1.20+
- Python Packages: Install the required Python dependencies:
`!pip install numpy torch matplotlib sumolib traci`

SUMO Installation: Follow the SUMO Installation Guide to set up SUMO on your machine.

Setup and Usage

1. Clone the Repository:

- git clone `https://github.com/Userfound404/Optimizing-Urban-Traffic-Signal-Timings-with-Reinforcement-Learning.git`
- cd Optimizing-Urban-Traffic-Signal-Timings-with-Reinforcement-Learning

2. Configure SUMO: Place the required SUMO files (boston.sumocfg, osm.net.xml.gz, osm.poly.xml.gz) in the sumo_config/ directory.

3. Run the Simulation: To start the training and evaluation process: `python main.py`

4. View Results: Visualization files and logs will be saved in the results/ directory.

## Results

The results demonstrate the effectiveness of the reinforcement learning agents in reducing traffic congestion. Key metrics include:

- Average Queue Length: Reduction in the number of vehicles waiting at signals.
- Average Waiting Time: Decrease in the average time vehicles spend idling.
- Total Reward: Improvement in overall system performance.

Example plots from the results/ folder:

Average Queue Length Over Episodes: 


![average_queue_length](https://github.com/user-attachments/assets/277901a4-11f9-4f48-85fd-8a24a13e2e5d)

Total Reward Over Episodes: 


![total_reward](https://github.com/user-attachments/assets/aa03262f-9cb2-4ba5-bb65-ee1ef692536a)

## Future Work

Potential improvements include:

- Implementing advanced RL algorithms (e.g., PPO, A3C).
- Adding cooperative strategies for agents to communicate and share information.
- Testing the system on larger, more complex networks.
- Incorporating real-world traffic data for validation.

## Contributing

Contributions are welcome! Please:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with a detailed description.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this project, but please provide appropriate attribution.
readme assisted by AI.
