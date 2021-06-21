#!/usr/bin/env bash

############################ 以下是重构后代码实验 ######################################
# 测试bug
# dqn所有网络均可运行
# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 1 --root_url './' --batch_size 64 --capacity 1000 --update_rate 10 --episodes_per_epoch 2 --episodes_per_test 1 --epoch_num 10 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > ./test.txt  2>&1 &
# a2c所有网络均可运行
# python -u train_agents.py --model_tag 'a2c' --net 'odfsn_self' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 1 --root_url './' --batch_size 64 --capacity 1000 --update_rate 10 --episodes_per_epoch 2 --episodes_per_test 1 --epoch_num 10 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > ./test.txt  2>&1 &
# ppo所有网络均可运行
# python -u train_agents.py --model_tag 'ppo' --net 'odfsn_self' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 1 --root_url './' --batch_size 64 --capacity 1000 --update_rate 10 --episodes_per_epoch 2 --episodes_per_test 1 --epoch_num 10 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > ./test.txt  2>&1 &

# dqn none
# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'none' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 1 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_none_seed_$seed.txt  2>&1 &
#     sleep 5
# done
# dqn oan
# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'oan' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 1 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_oan_seed_$seed.txt  2>&1 &
#     sleep 5
# done

# dqn odfsn_base v1
# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 1 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v1_seed_$seed.txt  2>&1 &
#     sleep 5
# done
# dqn odfsn_base v2
# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 1 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v2_seed_$seed.txt  2>&1 &
#     sleep 5
# done

# dqn odfsn_self v1
# for ((seed=0;seed<=3;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v1_seed_$seed.txt  2>&1 &
#     sleep 5
# done

# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed 4 --gpu_id 2 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v1_seed_4.txt  2>&1 &
# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed 5 --gpu_id 3 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v1_seed_5.txt  2>&1 &


# dqn odfsn_self v2
# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 1 --threads_num 1 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v2_seed_$seed.txt  2>&1 &
#     sleep 5
# done

# a2c none(参数不行)
# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'a2c' --net 'none' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 1 --threads_num 1 --root_url '/mnt/znn/data/magent_data/data/' --capacity 256 --e_coef 0.02 --episodes_per_epoch 20 --episodes_per_test 20 --epoch_num 2000 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/a2c_none_seed_$seed.txt  2>&1 &
#     sleep 5
# done
# python -u train_agents.py --model_tag 'a2c' --net 'none' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 1 --threads_num 1 --root_url '/mnt/znn/data/magent_data/data/' --capacity 256 --e_coef 0.02 --episodes_per_epoch 5 --episodes_per_test 20 --epoch_num 6000 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/a2c_none_seed_0v1.txt  2>&1 &

# a2c none
# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'a2c' --net 'none' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --capacity 100000 --e_coef 0.02 --episodes_per_epoch 20 --episodes_per_test 20 --epoch_num 2000 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/a2c_none_seed_$seed_v1.txt  2>&1 &
#     sleep 5
# done
# python -u train_agents.py --model_tag 'a2c' --net 'none' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 2 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --capacity 1000 --e_coef 0.02 --episodes_per_epoch 20 --episodes_per_test 20 --epoch_num 20000 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/a2c_none_seed_0_1v1.txt  2>&1 &



# ppo none
# python -u train_agents.py --model_tag 'ppo' --net 'none' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 1 --threads_num 1 --root_url '/mnt/znn/data/magent_data/data/' --capacity 256 --e_coef 0.02 --episodes_per_epoch 5 --episodes_per_test 20 --epoch_num 6000 --lr 1e-3 --k_epoch 3 --lmbda 0.95 --eps_clip 0.1 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/ppo_none_seed_0v1.txt  2>&1 &

# 测试一下noisy attn(oan, )
# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 10 --episodes_per_epoch 10 --episodes_per_test 5 --epoch_num 50 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_oan_seed_0_nosiy_v0.txt  2>&1 &
# 试跑1个seed的所有架构实验
# python -u train_agents.py --model_tag 'dqn' --net 'oan' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 1 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_oan_v2_seed_0_noisy.txt  2>&1 &
# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 1 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v1_seed_0_noisy.txt  2>&1 &
# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 1 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v2_seed_0_noisy.txt  2>&1 &
# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v1_seed_0_noisy.txt  2>&1 &
# python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v2_seed_0_noisy.txt  2>&1 &

# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'oan' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_oan_v2_$seed_noisy.txt  2>&1 &
#     sleep 5
# done
# python -u train_agents.py --model_tag 'dqn' --net 'oan' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 5 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_oan_noisy_test.txt  2>&1 &
# python -u train_agents.py --model_tag 'dqn' --net 'oan' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 5 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 10 --episodes_per_epoch 10 --episodes_per_test 20 --epoch_num 20 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_oan_noisy_test.txt  2>&1 &


# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v1_$seed_noisy.txt  2>&1 &
#     sleep 5
# done

# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v2_$seed_noisy.txt  2>&1 &
#     sleep 5
# done

# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v1_$seed_noisy.txt  2>&1 &
#     sleep 5
# done

# for ((seed=0;seed<=5;seed+=1))
# do
#     python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed $seed --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v2_$seed_noisy.txt  2>&1 &
#     sleep 5
# done

# 单层网络尝试(oan使用的是自己的独有网络，不需要改变)
python -u train_agents.py --model_tag 'dqn' --net 'none' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_none_new_noisy.txt  2>&1 &
python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v1_new_noisy.txt  2>&1 &
python -u train_agents.py --model_tag 'dqn' --net 'odfsn_base' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 0 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_base_v2_new_noisy.txt  2>&1 &
python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v1' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 1 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v1_new_noisy.txt  2>&1 &
python -u train_agents.py --model_tag 'dqn' --net 'odfsn_self' --agg_version 'v2' --agent_num 5 --map_size 15 --noisy_num 5 --seed 0 --gpu_id 1 --threads_num 2 --root_url '/mnt/znn/data/magent_data/data/' --batch_size 256 --capacity 100000 --update_rate 100 --episodes_per_epoch 200 --episodes_per_test 20 --epoch_num 200 --lr 1e-4 --opp_policy './resources/opp_model/opp_5v5v5.th' > /mnt/znn/data/magent_data/log/dqn_odfsn_self_v2_new_noisy.txt  2>&1 &
