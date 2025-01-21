# environment.py

import traci

class Environment:
    def __init__(self, sumo_cmd, max_steps, tl_ids):
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.tl_ids = tl_ids
        self.step = 0

    def reset(self):
        """
        Reset the simulation to the initial state.
        """
        traci.load(self.sumo_cmd[1:])  # Reload the simulation without restarting SUMO
        self.step = 0

    def step_simulation(self):
        """
        Advance the simulation by one step.
        """
        traci.simulationStep()
        self.step += 1

    def is_done(self):
        """
        Check if the simulation has reached the maximum number of steps.
        """
        return self.step >= self.max_steps

    def get_reward(self, tl_id):
        """
        Calculate the reward for the given traffic light ID.
        """
        # Use the negative sum of waiting times at the traffic light's controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        total_waiting_time = 0.0
        for lane in controlled_lanes:
            # Get the list of vehicles on the lane
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            # Sum the waiting times of the vehicles
            lane_waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
            total_waiting_time += lane_waiting_time
        # Negative reward to minimize waiting time
        reward = -total_waiting_time
        return reward

    def set_action(self, tl_id, action):
        """
        Apply the action to the traffic light.
        """
        # Map action to traffic light phase
        # Assuming action corresponds to a valid phase index
        traci.trafficlight.setPhase(tl_id, action)

    def get_total_queue_length(self, tl_id):
        """
        Get the total queue length for the traffic light's controlled lanes.
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        total_queue_length = 0
        for lane in controlled_lanes:
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            total_queue_length += queue_length
        return total_queue_length

    def close(self):
        """
        Close the simulation.
        """
        traci.close()
