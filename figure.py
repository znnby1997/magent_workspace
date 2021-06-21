from utils.data_process import get_curve
import os
from utils.settings import create_dir

# root_url = '/Users/zhangningning/研究生/科研/experiment/liu_ubuntu/5v5v5/a2c/csv'
root_url = '/Users/zhangningning/研究生/科研/experiment/liu_ubuntu/5v5v5/ppo/'
# csv_dict = {
#     'none': [
#         '/a2c_none/none_20201202134114_0.csv',
#         '/a2c_none/none_20201202134120_1.csv',
#         '/a2c_none/none_20201202134125_2.csv',
#         '/a2c_none/none_20201202134130_3.csv',
#         '/a2c_none/none_20201202134135_4.csv'
#     ],
#     'alw': [
#         '/a2c_alw_csv/alw_20201204151143_0.csv',
#         '/a2c_alw_csv/alw_20201204151148_1.csv',
#         '/a2c_alw_csv/alw_20201204151153_2.csv',
#         '/a2c_alw_csv/alw_20201204151158_3.csv',
#         '/a2c_alw_csv/alw_20201204151203_4.csv'
#     ],
#     'ian': [
#         '/a2c_ian/ian_20201204151448_0.csv',
#         '/a2c_ian/ian_20201204151714_1.csv',
#         '/a2c_ian/res.csv',
#         '/a2c_ian/ian_20201204151724_3.csv',
#         '/a2c_ian/ian_20201204151729_4.csv',
#     ]
# }

csv_dict = {
    'none': [
        '/res_none_0.csv',
        '/res_none_1.csv',
        '/res_none_2.csv',
        '/res_none_3.csv',
        '/res_none_4.csv'
    ],
    'alw': [
        '/res_alw_0.csv',
        '/res_alw_1.csv',
        '/res_alw_2.csv',
        '/res_alw_3.csv',
        '/res_alw_4.csv'
    ],
    'ian': [
        '/res_ian_0.csv',
        '/res_ian_1.csv',
        '/res_ian_2.csv',
        '/res_ian_3.csv',
        '/res_ian_4.csv',
    ]
}

figure_url = os.path.join(root_url, 'figure/')
csv_url = os.path.join(root_url, 'csv')

for label, name in zip(['total_reward', 'kill_num', 'survive_num'], ['tr', 'kn', 'sn']):
    create_dir(figure_url)
    get_curve(csv_url, csv_dict, label, figure_url + name + '.png', epoch_num=4000, w=0.85, x_label='epoch')

print('figure successfully')