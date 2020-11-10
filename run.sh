#!/usr/bin/env bash

# 尝试新的环境设置以及新的模型接入
# 训练对手
# python -u shell_main.py --train_opp 1 --episode_num 10000 --hidden_dim 16 --batch_size 128 --capacity 5000 --update_model_rate 20 > /mnt/znn/data/log_file/train_opp_policy_0.txt  2>&1 &
# batch_size 512
# python -u shell_main.py --train_opp 1 --episode_num 10000 --hidden_dim 16 --batch_size 512 --capacity 5000 --update_model_rate 20 > /mnt/znn/data/log_file/train_opp_policy_1.txt  2>&1 &
# batch_size 1024
# python -u shell_main.py --train_opp 1 --episode_num 10000 --hidden_dim 16 --batch_size 1024 --capacity 5000 --update_model_rate 20 > /mnt/znn/data/log_file/train_opp_policy_2.txt  2>&1 &
# capacity 10000
# python -u shell_main.py --train_opp 1 --episode_num 10000 --hidden_dim 16 --batch_size 1024 --capacity 10000 --update_model_rate 20 > /mnt/znn/data/log_file/train_opp_policy_3.txt  2>&1 &
# capacity 50000
# python -u shell_main.py --train_opp 1 --episode_num 5000 --hidden_dim 16 --batch_size 32 --capacity 5000 --update_model_rate 20 > /mnt/znn/data/log_file/opp_policy.txt  2>&1 &
# python -u shell_main.py --test_opp 1  --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_opp_policy_0.txt  2>&1 &
# python -u shell_main.py --train_opp 1 --episode_num 5000 --hidden_dim 32 --batch_size 5000 --capacity 100000 --update_model_rate 100 > /mnt/znn/data/log_file/train_opp_policy_5.txt  2>&1 &

# 开始实验环节
# baseline
# python -u shell_main.py --net_type 'none' --epoch_train 1 --ssed 1 --print_info 0 --hidden_dim 32 --batch_size 5000 --capacity 100000 --update_model_rate 100 --noisy 1 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_0.txt 2>&1 &
# 验证attention的好处
# python -u shell_main.py --net_type 'hand_weight' --epoch_train 1 --seed 4 --print_info 0 --hidden_dim 32 --batch_size 5000 --capacity 100000 --update_model_rate 100 --noisy 1 --concatenation 0 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_hand_agg_v4.txt 2>&1 &
# AAAI 2020中的分组聚合方式
# python -u shell_main.py --net_type 'gw' --epoch_train 1 --print_info 0 --hidden_dim 32 --batch_size 5000 --capacity 100000 --update_model_rate 100 --noisy 1 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_gw_0.txt 2>&1 &
# 测试模型
# python -u shell_main.py --test_model 1 --noisy 1 --test_model_url '/mnt/znn/data/model/hand_weight_20200921234923_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hw_0.txt  2>&1 &
# 3种baseline
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --seed 4 --print_info 0 --hidden_dim 32 --batch_size 5000 --capacity 100000 --noisy 1 --concatenation 1 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_sig_v4.txt 2>&1 &

# 引入latent variable z的尝试
# python -u shell_main.py --net_type 'ssd' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 512 --capacity 100000 --noisy 1 --beta 0.002 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_ssd_v8.txt 2>&1 &

# group mask
# python -u shell_main.py --net_type 'gn' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 1 --group_num 1 3 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_gn_bs256_v1.txt 2>&1 &
# 补充跑batch_size=512的几个baseline
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 512 --capacity 100000 --noisy 1 --group_num 2 5 --nonlin 'softmax' --need_diff 1 --beta 0.00002 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_entr_bs512_v1.txt 2>&1 &

# test
# python -u shell_main.py --test_model 1 --noisy 1 --group_num 2 5 --test_model_url '/mnt/znn/data/model/alw_20201026090255_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_alw_soft_bs512_0.txt  2>&1 &

# 无noisy
# python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 500 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_bs256_v0.txt 2>&1 &
# python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 500 --seed 1 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_bs256_v1.txt 2>&1 &
# python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 500 --seed 2 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_bs256_v2.txt 2>&1 &
# python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 500 --seed 3 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_bs256_v3.txt 2>&1 &
# python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 500 --seed 4 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_bs256_v4.txt 2>&1 &
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --epoch_num 500 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 1 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_concat_bs256_v0.txt 2>&1 &
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --epoch_num 500 --seed 1 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 1 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_concat_bs256_v1.txt 2>&1 &
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --epoch_num 500 --seed 2 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 1 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_concat_bs256_v2.txt 2>&1 &
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --epoch_num 500 --seed 3 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 1 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_concat_bs256_v3.txt 2>&1 &
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --epoch_num 500 --seed 4 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 1 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_concat_bs256_v4.txt 2>&1 &

# python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --epoch_num 500 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_dot_scale_bs256_v0.txt 2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --epoch_num 500 --seed 1 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_dot_scale_bs256_v1.txt 2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --epoch_num 500 --seed 2 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_dot_scale_bs256_v2.txt 2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --epoch_num 500 --seed 3 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_dot_scale_bs256_v3.txt 2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --epoch_num 500 --seed 4 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_dot_scale_bs256_v4.txt 2>&1 &

# python -u shell_main.py --net_type 'gruga' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_gruga_bs256_v0.txt 2>&1 &

# hand process group
# python -u shell_main.py --net_type 'hand_group' --epoch_train 1 --epoch_num 500 --seed 4 --print_info 0 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_hand_group_bs256_v4.txt 2>&1 &


# a2c训练
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --print_info 1 --hidden_dim 32 --max_train_steps 10000 --noisy 0 --group_num 2 5 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/a2c/log_file/test_v0.txt 2>&1 &

# none
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'none' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_bs256_$seed.txt 2>&1 &
#     sleep 2
# done

# alw_soft
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'alw' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 1 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_soft_bs256_$seed.txt 2>&1 &
#     sleep 2
# done

# scale_dot
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_dot_scale_bs256_$seed.txt 2>&1 &
#     sleep 2
# done

# gruga2
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'gruga2' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_gruga2_bs256_$seed.txt 2>&1 &
#     sleep 2
# done

# gruga
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'gruga' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_gruga_bs256_$seed.txt 2>&1 &
#     sleep 2
# done

# handgroup ig_num=5
# for ((seed=0;seed<5;seed+=1))
# do
#     python -u shell_main.py --net_type 'hand_group' --epoch_train 1 --epoch_num 200 --seed $seed --print_info 0 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_hand_group_ig5_$seed.txt 2>&1 &
#     sleep 2
# done

# a2c调参
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --max_train_steps 500000 --noisy 0 --learning_rate 1e-3 --beta 0.1 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_v1.txt 2>&1 &
# python -u shell_main.py --net_type 'alw' --basic_model 'a2c' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --max_train_steps 500000 --noisy 0 --learning_rate 1e-3 --beta 0.1 --nonlin 'softmax' --concatenation 1 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/a2c/log_file/train_a2c_alw_soft_v1.txt 2>&1 &
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --max_train_steps 100000 --noisy 0 --learning_rate 1e-3 --beta 0.5  --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/a2c/log_file/train_a2c_alw_soft_v1.txt 2>&1 &
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --max_train_steps 100000 --noisy 0 --learning_rate 1e-3 --beta 0.02  --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_v3.txt 2>&1 &
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --max_train_steps 100000 --noisy 0 --learning_rate 1e-3 --beta 0.1  --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/a2c/log_file/train_a2c_none_v4.txt 2>&1 &


# ppo测试
# python -u shell_main.py --net_type 'alw' --basic_model 'ppo' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --epoch_num 1 --noisy 0 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/ppo/log_file/test_ppo_none_v0.txt 2>&1 &
python -u shell_main.py --net_type 'none' --basic_model 'ppo' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --epoch_num 200 --noisy 0 --eps_clip 0.2 --t_horizon 10 --learning_rate 2e-3 --beta 1e-3 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/ppo/log_file/train_ppo_none_v2.txt 2>&1 &
sleep 2
python -u shell_main.py --net_type 'none' --basic_model 'ppo' --seed 0 --print_info 0 --print_log 1 --hidden_dim 32 --epoch_num 200 --noisy 0 --eps_clip 0.2 --t_horizon 10 --learning_rate 2e-3 --beta 1e-2 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/ppo/log_file/train_ppo_none_v3.txt 2>&1 &

# 测试DQN
# none
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --print_info 0 --net_flag 'none' --test_model_url '/mnt/znn/data/model/none_20201102153223_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --print_info 0 --net_flag 'none' --test_model_url '/mnt/znn/data/model/none_20201102153224_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --print_info 0 --net_flag 'none' --test_model_url '/mnt/znn/data/model/none_20201102153226_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --print_info 0 --net_flag 'none' --test_model_url '/mnt/znn/data/model/none_20201102153228_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --print_info 0 --net_flag 'none' --test_model_url '/mnt/znn/data/model/none_20201102153230_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_4.txt  2>&1 &
# sleep 2

# alw_sig
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --test_model_url '/mnt/znn/data/model/none_20201102153223_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --test_model_url '/mnt/znn/data/model/none_20201102153223_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --test_model_url '/mnt/znn/data/model/none_20201102153223_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --test_model_url '/mnt/znn/data/model/none_20201102153223_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --test_model_url '/mnt/znn/data/model/none_20201102153223_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_none_0.txt  2>&1 &
# sleep 2
# alw_soft
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --print_info 0 --net_flag 'alw_soft' --test_model_url '/mnt/znn/data/model/alw_20201102153658_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_alw_soft_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --print_info 0 --net_flag 'alw_soft' --test_model_url '/mnt/znn/data/model/alw_20201102153700_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_alw_soft_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --print_info 0 --net_flag 'alw_soft' --test_model_url '/mnt/znn/data/model/alw_20201102153702_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_alw_soft_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --print_info 0 --net_flag 'alw_soft' --test_model_url '/mnt/znn/data/model/alw_20201102153704_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_alw_soft_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --print_info 0 --net_flag 'alw_soft' --test_model_url '/mnt/znn/data/model/alw_20201102153706_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_alw_soft_4.txt  2>&1 &
# sleep 2
# scale_dot
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --print_info 0 --net_flag 'scale_dot' --test_model_url '/mnt/znn/data/model/dot_scale_20201102154115_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_scale_dot_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --print_info 0 --net_flag 'scale_dot' --test_model_url '/mnt/znn/data/model/dot_scale_20201102154117_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_scale_dot_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --print_info 0 --net_flag 'scale_dot' --test_model_url '/mnt/znn/data/model/dot_scale_20201102154119_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_scale_dot_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --print_info 0 --net_flag 'scale_dot' --test_model_url '/mnt/znn/data/model/dot_scale_20201102154121_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_scale_dot_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --print_info 0 --net_flag 'scale_dot' --test_model_url '/mnt/znn/data/model/dot_scale_20201102154123_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_scale_dot_4.txt  2>&1 &
# sleep 2
# gruga
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --print_info 0 --net_flag 'gruga' --test_model_url '/mnt/znn/data/model/gruga_20201104094325_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --print_info 0 --net_flag 'gruga' --test_model_url '/mnt/znn/data/model/gruga_20201104094325_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --print_info 0 --net_flag 'gruga' --test_model_url '/mnt/znn/data/model/gruga_20201104094326_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --print_info 0 --net_flag 'gruga' --test_model_url '/mnt/znn/data/model/gruga_20201104094328_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --print_info 0 --net_flag 'gruga' --test_model_url '/mnt/znn/data/model/gruga_20201104094328_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga_4.txt  2>&1 &
# sleep 2
# gruga2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --print_info 0 --net_flag 'gruga2' --test_model_url '/mnt/znn/data/model/gruga2_20201103112236_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga2_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --print_info 0 --net_flag 'gruga2' --test_model_url '/mnt/znn/data/model/gruga2_20201103112238_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga2_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --print_info 0 --net_flag 'gruga2' --test_model_url '/mnt/znn/data/model/gruga2_20201103112239_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga2_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --print_info 0 --net_flag 'gruga2' --test_model_url '/mnt/znn/data/model/gruga2_20201103112240_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga2_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --print_info 0 --net_flag 'gruga2' --test_model_url '/mnt/znn/data/model/gruga2_20201103112241_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_gruga2_4.txt  2>&1 &
# sleep 2
# hand_group(ig=10)
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --print_info 0 --net_flag 'hg_10' --test_model_url '/mnt/znn/data/model/hand_group_20201102011612_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg10_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --print_info 0 --net_flag 'hg_10' --test_model_url '/mnt/znn/data/model/hand_group_20201102011707_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg10_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --print_info 0 --net_flag 'hg_10' --test_model_url '/mnt/znn/data/model/hand_group_20201102011721_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg10_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --print_info 0 --net_flag 'hg_10' --test_model_url '/mnt/znn/data/model/hand_group_20201102011734_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg10_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --print_info 0 --net_flag 'hg_10' --test_model_url '/mnt/znn/data/model/hand_group_20201102011746_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg10_4.txt  2>&1 &
# sleep 2
# hang_group(ig=5)
# python -u shell_main.py --test_model 1 --noisy 0 --seed 0 --print_info 0 --net_flag 'hg_5' --test_model_url '/mnt/znn/data/model/hand_group_20201104101113_0.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg5_0.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 1 --print_info 0 --net_flag 'hg_5' --test_model_url '/mnt/znn/data/model/hand_group_20201104101114_1.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg5_1.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 2 --print_info 0 --net_flag 'hg_5' --test_model_url '/mnt/znn/data/model/hand_group_20201104101115_2.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg5_2.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 3 --print_info 0 --net_flag 'hg_5' --test_model_url '/mnt/znn/data/model/hand_group_20201104101116_3.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg5_3.txt  2>&1 &
# sleep 2
# python -u shell_main.py --test_model 1 --noisy 0 --seed 4 --print_info 0 --net_flag 'hg_5' --test_model_url '/mnt/znn/data/model/hand_group_20201104101117_4.th' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/test_hg5_4.txt  2>&1 &
# sleep 2