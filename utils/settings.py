import datetime
import torch
import random
import numpy as np
import os

def exp_name(args, exclude_lst, seed, seperator='_'):
    args_dict = vars(args)
    name = ''
    start = True
    for arg in args_dict:
        if arg in exclude_lst:
            continue
        if start:
            name += arg + '=' + str(args_dict[arg])
        else:
            name += seperator + arg + '=' + str(args_dict[arg])
        start = False
    
    cur_time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    name = name + '/seed=' + str(seed) + seperator + cur_time_str
    return name

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_dir(dir_url):
    if not os.path.exists(dir_url):
        os.makedirs(dir_url)


