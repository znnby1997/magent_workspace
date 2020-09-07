#!/usr/bin/env bash

# seed=0, episode_num=5000, threads=1 
# python -u shell_main.py --train_opp 1 > /mnt/znn/data/log_file/opp_train_log.txt  2>&1 &

# seed=1, episode_num=5000, threads=2
# python -u shell_main.py --train_opp 1 --seed 1 > /mnt/znn/data/log_file/opp_train_log_v1.txt  2>&1 &

# id=1 seed=0 episode_num=10000 threads=1 net_type='dyan-mean'
# python -u shell_main.py --net_type 'dyan' --episode_num 10000 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_mean_v0.txt  2>&1 &

# id=2 seed=0 episode_num=5000 threads=1 net_type='dyan-mean' batch_size=4500
# python -u shell_main.py --net_type 'dyan' --batch_size 4500 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_mean_v1.txt  2>&1 &

# id=3 seed=0 episode_num=5000 threads=1 net_type='dyan-sum' batch_size=5000
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v0.txt  2>&1 &

# id=4 seed=0 episode_num=5000 threads=1 net_type='dyan-max' batch_size=5000
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'max' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_max_v0.txt  2>&1 &

# id=5 seed=0 episode_num=5000 threads=1 net_type='gruga' batch_size=5000
# python -u shell_main.py --net_type 'gruga' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v0.txt  2>&1 &

# id=6 seed=0 episode_sum=5000 threads=1 net_type='dyan-mean' batch_size=5000
# python -u shell_main.py --net_type 'dyan' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_mean_v2.txt  2>&1 &

# test dyan-sum
# python -u shell_main.py --test_model 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/dyan_20200801105530.th' > /mnt/znn/data/log_file/test_log_sum_v0.txt  2>&1 &


# test cuda
# python -u shell_main.py --net_type 'alw' --episode_num 1000 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_test.txt  2>&1 &

# id=7 seed=0 episode_sum=10000 threads=1 net_type='dyan-sum' batch_size=5000
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --episode_num 10000 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v1.txt  2>&1 &

# id=9 seed=0 episode_sum=5000 threads=1 net_type='alw_att_net' nonlin='softmax' batch_size=5000
# python -u shell_main.py --net_type 'alw' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v0.txt  2>&1 &

# id=10 seed=0 episode_num=5000 threads=1 net_type='alw' nonlin='softmax' batch_size=5000
# python -u shell_main.py --net_type 'alw' --learning_rate 0.00001 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v2.txt  2>&1 &

# id=11 seed=0 episode_num=5000 threads=1 net_type='alw' nonlin='softmax' batch_size=4000
# python -u shell_main.py --net_type 'alw' --learning_rate 0.00001 --batch_size 4000 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v3.txt  2>&1 &

# id=12 seed=0 episode_num=5000 threads=1 net_type='gru' batch_size=5000
# python -u shell_main.py --net_type 'gruga' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v0.txt  2>&1 &

# id=13 test dyan-sum
# python -u shell_main.py --test_model 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/dyan_20200804003404.th' > /mnt/znn/data/log_file/test_log_dyan_sum_v1.txt  2>&1 &

# id=14 seed=0 episode_num=5000 threads=1 net_type='gru' batch_size=5000
# python -u shell_main.py --net_type 'gruga' --learning_rate 0.00005 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v1.txt  2>&1 &

# id=15 seed=0 episode_num=5000 threads=1 net_type='gru' batch_size=5000
# python -u shell_main.py --net_type 'gruga' --learning_rate 0.00001 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v2.txt  2>&1 &

# id=16 seed=0 episode_num=5000 threads=1 net_type='gru' batch_size=4000
# python -u shell_main.py --net_type 'gruga' --learning_rate 0.00001 --batch_size 4000 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v3.txt  2>&1 &

# id=17 test alw-soft
# python -u shell_main.py --test_model 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/alw_20200804133242.th' > /mnt/znn/data/log_file/test_log_alw_soft_v2.txt  2>&1 &

# id = 18 seed=0 episode_num=5000 threads=1 net_type='dyan-sum' batch_size=5000
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --learning_rate 0.00001 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v2.txt  2>&1 &

# id = 19 seed=0 episode_num=5000 threads=1 net_type='dot_scale' batch_size=5000
# python -u shell_main.py --net_type 'dot_scale' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v0.txt  2>&1 &

# id = 20 seed=0 episode_num=5000 threads=1 net_type='alw' batch_size=5000
# python -u shell_main.py --net_type 'alw' --em_dim 64  --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v4.txt  2>&1 &

# id = 21 seed=0 episode_num=5000 threads=1 net_type='alw' batch_size=5000 减少一层网络
# python -u shell_main.py --net_type 'dyan'  --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v5.txt  2>&1 &

# id = 22 test dyan-sum
# python -u shell_main.py --test_model 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/dyan_20200807000057.th' > /mnt/znn/data/log_file/test_log_dyan_sum_v2.txt  2>&1 &

# id = 23 seed=0 episode_num=5000 threads=1 net_type='gruga' batch_size=5000 em_dim=16
# python -u shell_main.py --net_type 'gruga' --em_dim 16 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v4.txt  2>&1 &

# id = 24 seed=0 episode_num=5000 threads=1 net_type='gruga' batch_size=5000 em_dim=64
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v5.txt  2>&1 &

# test gruga
# python -u shell_main.py --test_model 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/gruga_20200806234302.th' > /mnt/znn/data/log_file/test_log_gruga_v3.txt  2>&1 &

# test epoch train
# python -u shell_main.py --net_type 'dyan' --epoch_train 1 --print_info 0 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v3.txt  2>&1 &

# id = 25 seed=0 episode_num=5000 threads=1 net_type='gruga' batch_szie=5000 em_dim=128
# python -u shell_main.py --net_type 'gruga' --em_dim 128 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v6.txt  2>&1 &

# id = 26 seed=0 episode_num=5000 threads=1 net_type='alw' batch_szie=5000 em_dim=128
# python -u shell_main.py --net_type 'alw' --em_dim 128 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v6.txt  2>&1 &

# test distance sort obs and dyan_group
# python -u shell_main.py --net_type 'dyan_group' --aggregate_form 'max' --distance_sort 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_group_v1.txt  2>&1 &


# id = 27 seed=0 episode_num=5000 threads=1 net_type='gruga' batch_size=3000 em_dim=64
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --batch_size 3000 --capacity 50000 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v7.txt  2>&1 &


# id = 28 seed=0 episode_num=5000 threads=1 net_type='dyan_group' batch_size=5000 em_dim=32
# python -u shell_main.py --net_type 'dyan_group' --distance_sort 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_group_v0.txt  2>&1 &

# test new code
# # id = 29 seed=0 episode_num=5000 threads=1 net_type='gruga' batch_size=5000 em_dim=64
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v8.txt  2>&1 &

# id = 30 seed = 0 episode_num = 5000 threads = 1 net_type='alw' batch_size=5000 em_dim=64
# python -u shell_main.py --net_type 'alw' --em_dim 64 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v7.txt  2>&1 &

# dyan-sum 5组实验 epoch 100 episode_num_per_epoch 100 batch_size=5000 hidden_dim=32 lr=1e-4
# seed=0
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v4.txt  2>&1 &
# seed=1
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --seed 1 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v5.txt  2>&1 &
# seed=2
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --seed 2 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v6.txt  2>&1 &
# seed=3
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --seed 3 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v7.txt  2>&1 &
# seed=4
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'sum' --seed 4 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_sum_v8.txt  2>&1 &

# test dyan group
# python -u shell_main.py --test_model 1 --distance_sort 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/dyan_group_20200811134008.th' > /mnt/znn/data/log_file/test_log_dyan_group_v0.txt  2>&1 &

# id = 31 seed=0 episode_num=5000 threads=1 net_type='dyan_group' batch_size=5000 em_dim=32
# python -u shell_main.py --net_type 'dyan_group' --aggregate_form 'sum' --distance_sort 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_group_v4.txt  2>&1 &

# dyan-mean 5组实验
# seed=0
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'mean' --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_mean_v2.txt  2>&1 &
# seed=1
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'mean' --seed 1 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_mean_v3.txt  2>&1 &
# seed=2
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'mean' --seed 2 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_mean_v4.txt  2>&1 &
# seed=3
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'mean' --seed 3 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_mean_v5.txt  2>&1 &
# seed=4
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'mean' --seed 4 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_mean_v6.txt  2>&1 &

# dyan-max 5组实验
# seed=0
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'max' --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_max_v1.txt  2>&1 &
# seed=1
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'max' --seed 1 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_max_v2.txt  2>&1 &
# seed=2
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'max' --seed 2 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_max_v3.txt  2>&1 &
# seed=3
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'max' --seed 3 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_max_v4.txt  2>&1 &
# seed=4
# python -u shell_main.py --net_type 'dyan' --aggregate_form 'max' --seed 4 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_max_v5.txt  2>&1 &

# gruga 5组实验
# seed=0
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v9.txt  2>&1 &
# seed=1
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --seed 1 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v10.txt  2>&1 &
# seed=2
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --seed 2 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v11.txt  2>&1 &
# seed=3
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --seed 3 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v12.txt  2>&1 &
# seed=4
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --seed 4 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v13.txt  2>&1 &

# scale dot-product 调参
# python -u shell_main.py --net_type 'dot_scale' --em_dim 64 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v1.txt  2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --em_dim 128 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v2.txt  2>&1 &
# scale dot 加入self-info作为key
# python -u shell_main.py --net_type 'dot_scale' --em_dim 128 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v3.txt  2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --em_dim 48 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v4.txt  2>&1 &

# alw 5组实验
# seed=0
# python -u shell_main.py --net_type 'alw' --em_dim 64 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v8.txt  2>&1 &
# seed=1
# python -u shell_main.py --net_type 'alw' --em_dim 64 --seed 1 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v9.txt  2>&1 &
# seed=2
# python -u shell_main.py --net_type 'alw' --em_dim 64 --seed 2 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v10.txt  2>&1 &
# seed=3
# python -u shell_main.py --net_type 'alw' --em_dim 64 --seed 3 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v11.txt  2>&1 &
# seed=4
# python -u shell_main.py --net_type 'alw' --em_dim 64 --seed 4 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v12.txt  2>&1 &

# group att agg
# python -u shell_main.py --net_type 'gaa' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v1.txt  2>&1 &

# scale-dot 5组实验
# seed=0
# python -u shell_main.py --net_type 'dot_scale' --em_dim 64 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_test.txt  2>&1 &
# seed=1
# python -u shell_main.py --net_type 'dot_scale' --em_dim 64 --seed 1 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v6.txt  2>&1 &
# seed=2
# python -u shell_main.py --net_type 'dot_scale' --em_dim 64 --seed 2 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v7.txt  2>&1 &
# seed=3
# python -u shell_main.py --net_type 'dot_scale' --em_dim 64 --seed 3 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v8.txt  2>&1 &
# seed=4
# python -u shell_main.py --net_type 'dot_scale' --em_dim 64 --seed 4 --epoch_train 1 --print_info 0 --epoch_num 100 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_test.txt  2>&1 &

# test gaa
# python -u shell_main.py --test_model 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/gaa_20200818141205.th' > /mnt/znn/data/log_file/test_log_gaa_v0.txt  2>&1 &


# dyan-group
# 手工设置group mask
# python -u shell_main.py --net_type 'dyan_group' --distance_sort 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_dyan_group_test.txt  2>&1 &
# test model
# python -u shell_main.py --test_model 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/dyan_group_20200820173047.th' > /mnt/znn/data/log_file/test_log_dyan_group_v0.txt  2>&1 &

# none
# python -u shell_main.py --net_type 'none' --update_model_rate 20 --hidden_dim 16 --capacity 5000 --batch_size 32 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_none_v11.txt  2>&1 &
# alw concatenation
# python -u shell_main.py --net_type 'alw' --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_v0.txt  2>&1 &
# scale dot concatenation
# python -u shell_main.py --net_type 'dot_scale' --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v10.txt  2>&1 &
# gruga concatenation em_dim = 32
# python -u shell_main.py --net_type 'gruga' --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v10.txt  2>&1 &
# gruga concatenation em_dim = 64
# python -u shell_main.py --net_type 'gruga' --em_dim 64 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga_v11.txt  2>&1 &

# gaa魔改版(调参中)
# python -u shell_main.py --net_type 'gaa' --em_dim 64 --hidden_dim 64 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v5.txt  2>&1 &

# gruga改进版
# no concatenation
# python -u shell_main.py --net_type 'gruga2' --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga2_v0.txt  2>&1 &
# concatenation
# python -u shell_main.py --net_type 'gruga2' --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga2_v1.txt  2>&1 &

# none参数进一步测试
# python -u shell_main.py --net_type 'none' --epoch_train 1 --print_info 0 --hidden_dim 64 --batch_size 5000 --capacity 100000 --learning_rate 0.00005 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_none_v12.txt  2>&1 &

# none跑(剩下的4个seed)
# seed = 1
# python -u shell_main.py --net_type 'none' --seed 1 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_none_v15.txt  2>&1 &
# seed = 2
# python -u shell_main.py --net_type 'none' --seed 2 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_none_v16.txt  2>&1 &
# seed = 3
# python -u shell_main.py --net_type 'none' --seed 3 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_none_v16.txt  2>&1 &
# seed = 4
# python -u shell_main.py --net_type 'none' --seed 4 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_none_v17.txt  2>&1 &

# gaa新参数尝试
# python -u shell_main.py --net_type 'gaa' --em_dim 16 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v6.txt  2>&1 &
# 测试可视化debug
# python -u shell_main.py --net_type 'gaa' --em_dim 16 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v7.txt  2>&1 &
# python -u shell_main.py --net_type 'gaa' --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v8.txt  2>&1 &
# python -u shell_main.py --net_type 'gaa' --em_dim 32 --hidden_dim 16 --batch_size 64 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v9.txt  2>&1 &
# python -u shell_main.py --net_type 'gaa' --em_dim 32 --hidden_dim 16 --batch_size 1024 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v11.txt  2>&1 &
# python -u shell_main.py --net_type 'gaa' --em_dim 32 --hidden_dim 32 --batch_size 1024 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v12.txt  2>&1 &

# alw跑
# python -u shell_main.py --net_type 'alw' --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v13.txt  2>&1 &
# python -u shell_main.py --net_type 'alw' --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v14.txt  2>&1 &
# python -u shell_main.py --net_type 'alw' --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v15.txt  2>&1 &

# scale-dot
# python -u shell_main.py --net_type 'dot_scale' --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v11.txt  2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v12.txt  2>&1 &

# alw 跑五组第一组试跑
# seed = 0
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v15.txt  2>&1 &
# seed = 1
# python -u shell_main.py --net_type 'alw' --seed 1 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v16.txt  2>&1 &
# seed = 2
# python -u shell_main.py --net_type 'alw' --seed 2 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v17.txt  2>&1 &
# seed = 3
# python -u shell_main.py --net_type 'alw' --seed 3 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v18.txt  2>&1 &
# seed = 4
# python -u shell_main.py --net_type 'alw' --seed 4 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v19.txt  2>&1 &

# dot-scale 跑五组第一组试跑
# seed = 0
# python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v13.txt  2>&1 &
# seed = 1
# python -u shell_main.py --net_type 'dot_scale' --seed 1 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v14.txt  2>&1 &
# seed = 2
# python -u shell_main.py --net_type 'dot_scale' --seed 2 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v15.txt  2>&1 &
# seed = 3
# python -u shell_main.py --net_type 'dot_scale' --seed 3 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v16.txt  2>&1 &
# seed = 4
# python -u shell_main.py --net_type 'dot_scale' --seed 4 --epoch_train 1 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v17.txt  2>&1 &
# 聚合尝试
python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --print_info 0 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v19.txt  2>&1 &


# gaa试跑
# seed = 0
# python -u shell_main.py --net_type 'gaa' --epoch_train 1 --print_info 0 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v13.txt  2>&1 &
# seed = 1
# python -u shell_main.py --net_type 'gaa' --seed 1 --epoch_train 1 --print_info 0 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v14.txt  2>&1 &
# seed = 2
# python -u shell_main.py --net_type 'gaa' --seed 2 --epoch_train 1 --print_info 0 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v15.txt  2>&1 &
# seed = 3
# python -u shell_main.py --net_type 'gaa' --seed 3 --epoch_train 1 --print_info 0 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v16.txt  2>&1 &
# seed = 4
# python -u shell_main.py --net_type 'gaa' --seed 4 --epoch_train 1 --print_info 0 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v17.txt  2>&1 &

# gaa尝试统计胜率
# seed = 0
# python -u shell_main.py --net_type 'gaa' --episode_num 10000 --print_info 0 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v18.txt  2>&1 &

# alw统计胜率
# seed = 0
# python -u shell_main.py --net_type 'alw' --episode_num 10000 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_alw_soft_v20.txt  2>&1 &

# scale-dot统计胜率
# seed = 0
# python -u shell_main.py --net_type 'dot_scale' --episode_num 10000 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --concatenation 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_scale_dot_v18.txt  2>&1 &

# none统计胜率
# seed = 0
# python -u shell_main.py --net_type 'none' --episode_num 10000 --print_info 0 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_none_v18.txt  2>&1 &

# gaa模型测试
# python -u shell_main.py --test_model 1 --print_mask 1 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' --test_model_url '/mnt/znn/data/model/gaa_20200831225322.th' > /mnt/znn/data/log_file/test_log_gaa_v1.txt  2>&1 &

# gruga新参数尝试
# python -u shell_main.py --net_type 'gruga2' --episode_num 10000 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gruga2_v4.txt  2>&1 &
# python -u shell_main.py --net_type 'gaa' --episode_num 50000 --em_dim 64 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 --opp_policy '/mnt/znn/data/model/opp_20200729235611.th' > /mnt/znn/data/log_file/train_log_gaa_v19.txt  2>&1 &
