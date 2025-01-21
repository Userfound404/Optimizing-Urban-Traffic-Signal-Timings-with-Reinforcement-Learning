import os
import sys


def set_sumo(gui, sumocfg_file_name):
    """
    Configure SUMO command-line arguments.
    """
    # Ensure SUMO_HOME is set
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    from sumolib import checkBinary

    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    sumo_cmd = [sumo_binary, "-c", sumocfg_file_name]

    return sumo_cmd
