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
# python -u shell_main.py --net_type 'none' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 32 --capacity 100000 --noisy 0 --group_num 2 5 --nonlin 'softmax' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_none_bs32_v0.txt 2>&1 &
# python -u shell_main.py --net_type 'alw' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 32 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 1 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_alw_sig_concat_bs32_v0.txt 2>&1 &
# python -u shell_main.py --net_type 'dot_scale' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_dot_scale_bs256_v0.txt 2>&1 &
# python -u shell_main.py --net_type 'gruga' --epoch_train 1 --seed 0 --print_info 1 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --concatenation 0 --nonlin 'sigmoid' --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_gruga_bs256_v0.txt 2>&1 &

# hand process group
python -u shell_main.py --net_type 'hand_group' --epoch_train 1 --epoch_num 500 --seed 0 --print_info 0 --hidden_dim 32 --batch_size 256 --capacity 100000 --noisy 0 --group_num 2 5 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/log_file/train_hand_group_bs256_v2.txt 2>&1 &


# a2c训练
# python -u shell_main.py --net_type 'none' --basic_model 'a2c' --seed 0 --print_info 1 --hidden_dim 32 --max_train_steps 10000 --noisy 0 --group_num 2 5 --opp_policy '/mnt/znn/data/opp_model/opp_20200915121500/final_model.th' > /mnt/znn/data/a2c/log_file/test_v0.txt 2>&1 &
