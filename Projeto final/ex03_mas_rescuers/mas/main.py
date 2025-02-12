import sys
import os
import time

# Import necessary classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

def main(data_folder_name, config_ag_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
        
    # Instantiate the environment
    env = Env(data_folder)
    
    # Instantiate the master rescuer
    # This agent unifies the maps and instantiates other 3 agents
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4)  # 4 is the number of explorer agents

    # Instantiate explorers and associate them with the master rescuer
    # Explorers need to know the rescuer to send the map, so the rescuer is instantiated first
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        Explorer(env, explorer_file, master_rescuer)

    # Run the environment simulator
    env.run()

if __name__ == '__main__':
    """To get data from a different folder than the default, called 'data',
       pass it via command line arguments."""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_400v_90x90")
    
    config_ag_folder_name = os.path.join("ex03_mas_rescuers", "cfg_1")
    main(data_folder_name, config_ag_folder_name)