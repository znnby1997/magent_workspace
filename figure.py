from utils.data_process import get_curve
import os
from utils.settings import create_dir

root_url = ''
csv_dict = {

}

figure_url = os.path.join(root_url, 'figure/')


for label, name in zip(['total_reward', 'kill_num', 'survive_num'], ['tr', 'kn', 'sn']):
    create_dir(figure_url)
    get_curve(root_url, csv_dict, label, figure_url + name + '.png', epoch_num=2000, w=0.85, x_label='epoch')

print('figure successfully')