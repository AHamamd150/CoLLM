import yaml
import numpy as np
########################
def read_conf(file_): 
    with open(file_, 'r') as file:
        data = yaml.safe_load(file)
        
    #######Read general setup ###########
    output_dir = data["Output_dir"]
    DEFAULT_MODEL = data["DEFAULT_MODEL"]
    MAX_RETRIES = data["MAX_RETRIES"]
    input_test  = data["Input_file"]
    user_input = data["User_input"]
    return (output_dir,
              DEFAULT_MODEL,
              MAX_RETRIES,
              input_test,
              user_input)
