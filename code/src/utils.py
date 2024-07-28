import numpy as np
import os

def get_data_dir(folder): # for the data directory
    cur_dir = os.path.dirname(os.path.abspath(__file__)) # should be util
    #directory one level above
    home_dir = os.path.dirname(cur_dir)
    #check if the data directory is exist
    if not os.path.exists(os.path.join(home_dir, folder)): #customize to cifar
        os.makedirs(os.path.join(home_dir, folder))
    #assign the data directory
    data_dir = os.path.join(home_dir, folder) # this is to get the data directory
    return data_dir