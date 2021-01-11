#!/usr/bin/env bash

# random_init_pos env

# test ian+alw_soft
# python -u shell_main.py --test_model 1 --basic_model 'dqn' --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --seed 0 --net_flag 'ian' --csv_url '/mnt/znn/data/csv/' --test_log_url '/mnt/znn/data/csv/' --test_model_url '/mnt/znn/data/model/ian_20201125082505_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/log_file/test_ian_alw_soft_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --basic_model 'dqn' --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --seed 1 --net_flag 'ian' --csv_url '/mnt/znn/data/csv/' --test_log_url '/mnt/znn/data/csv/' --test_model_url '/mnt/znn/data/model/ian_20201125082752_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/log_file/test_ian_alw_soft_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --basic_model 'dqn' --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --seed 2 --net_flag 'ian' --csv_url '/mnt/znn/data/csv/' --test_log_url '/mnt/znn/data/csv/' --test_model_url '/mnt/znn/data/model/ian_20201125082758_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/log_file/test_ian_alw_soft_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --basic_model 'dqn' --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --seed 3 --net_flag 'ian' --csv_url '/mnt/znn/data/csv/' --test_log_url '/mnt/znn/data/csv/' --test_model_url '/mnt/znn/data/model/ian_20201125082803_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/log_file/test_ian_alw_soft_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --basic_model 'dqn' --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --seed 4 --net_flag 'ian' --csv_url '/mnt/znn/data/csv/' --test_log_url '/mnt/znn/data/csv/' --test_model_url '/mnt/znn/data/model/ian_20201125082808_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/log_file/test_ian_alw_soft_4.txt  2>&1 &
# sleep 2

# 5v5v5 测试a2c none
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --max_train_steps 100000 --update_interval 20 --learning_rate 1e-3 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v0.txt  2>&1 &
# a2c none 熵值调大，增大探索
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --max_train_steps 100000 --update_interval 20 --print_log 1 --learning_rate 1e-3 --beta 0.1 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v1.txt  2>&1 &
# a2c none 增大update interval
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --max_train_steps 100000 --update_interval 200 --print_log 1 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v2.txt  2>&1 &
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --max_train_steps 200000 --update_interval 1000 --print_log 1 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v3.txt  2>&1 &

# 5v5v5 测试ppo none
# python -u shell_main.py --net_type 'none' --basic_model 'ppo' --seed 0 --net_flag 'none' --epoch_num 200 --t_horizon 10 --eps_clip 0.2 --learning_rate 2e-3 --beta 1e-3 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/ppo/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/ppo/log_file/train_ppo_none_5v5v5_v0.txt  2>&1 &
# 增大t_horizon
# python -u shell_main.py --net_type 'none' --basic_model 'ppo' --seed 0 --net_flag 'none' --epoch_num 200 --t_horizon 200 --eps_clip 0.2 --learning_rate 2e-3 --beta 1e-3 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/ppo/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/ppo/log_file/train_ppo_none_5v5v5_v1.txt  2>&1 &


# 10v10v10 20 * 20
# none
# python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 200 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 10000 --random_init_pos 1 --agent_num 10 --map_size 20 --noisy_agent_num 10 --opp_policy '/mnt/znn/data/opp_model/opp_20201126030141/final_model.th' > /mnt/znn/data/log_file/train_none_10v10v10_0.txt 2>&1 &
# for ((seed=1;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 10000 --random_init_pos 1 --agent_num 10 --map_size 20 --noisy_agent_num 10 --opp_policy '/mnt/znn/data/opp_model/opp_20201126030141/final_model.th' > /mnt/znn/data/log_file/train_none_10v10v10_$seed.txt 2>&1 &
#     sleep 5
# done

# alw_soft
# python -u shell_main.py --net_type 'alw' --nonlin 'softmax' --concatenation 1 --epoch_train 1 --epoch_num 200 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 10000 --random_init_pos 1 --agent_num 10 --map_size 20 --noisy_agent_num 10 --opp_policy '/mnt/znn/data/opp_model/opp_20201126030141/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_10v10v10_0.txt 2>&1 &
# for ((seed=1;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'alw' --nonlin 'softmax' --concatenation 1 --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 10000 --random_init_pos 1 --agent_num 10 --map_size 20 --noisy_agent_num 10 --opp_policy '/mnt/znn/data/opp_model/opp_20201126030141/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_10v10v10_$seed.txt 2>&1 &
#     sleep 5
# done

# ian+alw_soft
# python -u shell_main.py --net_type 'ian' --epoch_train 1 --epoch_num 200 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 10000 --random_init_pos 1 --agent_num 10 --map_size 20 --noisy_agent_num 10 --opp_policy '/mnt/znn/data/opp_model/opp_20201126030141/final_model.th' > /mnt/znn/data/log_file/train_ian_10v10v10_0.txt 2>&1 &
# for ((seed=1;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'ian' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 10000 --random_init_pos 1 --agent_num 10 --map_size 20 --noisy_agent_num 10 --opp_policy '/mnt/znn/data/opp_model/opp_20201126030141/final_model.th' > /mnt/znn/data/log_file/train_ian_10v10v10_$seed.txt 2>&1 &
#     sleep 5
# done

# 5v5v5 test a2c with multi-trajectory
# 该参数设置有效果
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --epoch_num 1000 --test_rate 20 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v1.txt  2>&1 &
# entropy_weight太大了
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --epoch_num 1000 --test_rate 20 --learning_rate 1e-3 --beta 0.1 --capacity 5000 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v2.txt  2>&1 &

# 增大步数
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed $seed --net_flag 'none' --epoch_num 2000 --test_rate 20 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_2000e_$seed.txt  2>&1 &
#     sleep 5
# done

# 减小数据量
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --epoch_num 1000 --test_rate 10 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v3.txt  2>&1 &
# sleep 5
# 增大数据量
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --net_flag 'none' --epoch_num 1000 --test_rate 30 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_5v5v5_v4.txt  2>&1 &

# 5v5v5 test ppo with multi-trajectory
# python -u shell_main.py --net_type 'none' --basic_model 'ppo' --seed 0 --net_flag 'none' --epoch_num 1000 --episodes_per_epoch 20 --eps_clip 0.2 --learning_rate 2e-3 --beta 1e-3 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/ppo/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/ppo/log_file/train_ppo_none_5v5v5_v3.txt  2>&1 &
# python -u shell_main.py --net_type 'none' --basic_model 'ppo' --seed 0 --net_flag 'none' --print_log 1 --epoch_num 1000 --episodes_per_epoch 20 --eps_clip 0.2 --learning_rate 2e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/ppo/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/ppo/log_file/train_ppo_none_5v5v5_v4.txt  2>&1 &
# python -u shell_main.py --net_type 'none' --basic_model 'ppo' --seed 0 --net_flag 'none' --print_log 1 --epoch_num 1000 --episodes_per_epoch 20 --eps_clip 0.2 --learning_rate 2e-3 --beta 0.1 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5  --csv_url '/mnt/znn/data/ppo/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/ppo/log_file/train_ppo_none_5v5v5_v5.txt  2>&1 &

# 5v5v5 a2c
# alw_softmax
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'alw' --basic_model 'a2c' --seed $seed --net_flag 'alw' --nonlin 'softmax' --concatenation 1 --epoch_num 2000 --test_rate 20 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_alw_soft_5v5v5_2000e_$seed.txt  2>&1 &
#     sleep 5
# done
# python -u shell_main.py --net_type 'ian' --basic_model 'a2c' --seed 0 --net_flag 'ian' --epoch_num 2000 --test_rate 20 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_ian_5v5v5_2000e_0.txt  2>&1 &
# ian_alw
# for ((seed=1;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'ian' --basic_model 'a2c' --seed $seed --net_flag 'ian' --epoch_num 2000 --test_rate 20 --learning_rate 1e-3 --beta 0.02 --random_init_pos 1 --agent_num 5 --map_size 15 --noisy_agent_num 5 --print_info 0 --csv_url '/mnt/znn/data/a2c/csv/' --opp_policy '/mnt/znn/data/opp_model/opp_20201121194852/episode_5999.th' > /mnt/znn/data/a2c/log_file/train_a2c_ian_5v5v5_2000e_$seed.txt  2>&1 &
#     sleep 5
# done

############################ 以下是重构后代码实验 ######################################