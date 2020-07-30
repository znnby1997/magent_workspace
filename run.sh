#!/usr/bin/env bash

# seed=0, episode_num=5000, threads=1 
# python -u shell_main.py --train_opp 1 > /mnt/znn/data/log_file/opp_train_log.txt  2>&1 &

# seed=1, episode_num=5000, threads=2
python -u shell_main.py --train_opp 1 --seed 1 > /mnt/znn/data/log_file/opp_train_log_v1.txt  2>&1 &