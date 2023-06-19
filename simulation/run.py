import numpy as np
import pandas as pd
import os
import sys
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def run_baseline_parallel(config):

    cmdLine = './waf --run \"scratch/dcn-simulator {config}\"'.format(config=config)
    print(cmdLine)
    os.system(cmdLine)


if __name__ == "__main__":

    os.chdir(sys.path[0])

    config_temp = "{config_dir}/config{config_id}_{trace}.txt"
    config_dir = "data/config/encore/"

    config_file = config_dir + 'all_configs.csv'
    configs = pd.read_csv(config_file)[:1000]

    args = []
    for i, config in configs.iterrows():
        config_id = int(config['config'])
        trace_id = config['trace']
        conf = config_temp.format(config_dir=os.path.join(config_dir), config_id=config_id, trace=trace_id)
        args.append(conf)

    pool = ThreadPool(12)
    pool.map(run_baseline_parallel, args)
    pool.close()
    pool.join() 
