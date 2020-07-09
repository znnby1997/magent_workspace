import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import os

path_url = '/Users/zhangningning/研究生/科研/experiment/magent_workspace/data/'
model_set = [
    'none', 'softmax', 'tanh', 'sigmoid', 'hand', 'alw'
]
x_trick = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
y_trick = [
    'self_info', 'opp1', 'opp2', 'opp3', 'opp4', 'opp5', 'opp6', 'opp7',
    'opp8', 'opp9', 'opp10', 'partner1', 'partner2', 'partner3', 'partner4', 'partner5', 'partner6',
    'partner7', 'partner8', 'partner9']

def get_csv(csv_url, data_dict):
    if os.path.exists(csv_url):
        df = pd.read_csv(csv_url)
    else:
        df = pd.DataFrame()
        df['episode'] = [i for i in range(df.shape[0])]
    for key, value in zip(data_dict.keys(), data_dict.values()):
        df[key] = value
    df.to_csv(csv_url, index=False)

def att_pack(att_weight_list, agent_num=10):
    new_weight_list = []
    for weight in att_weight_list:
        agent_info = np.zeros(agent_num * 2)
        index = 0
        for i in range(agent_num * 2 - 1):
            agent_info[i] = sum(weight[index: index + 21])
            index += 21
        new_weight_list.append(agent_info)
    return new_weight_list

def get_att_weight_csv(new_file, data_dict, agent_num):
    df = pd.DataFrame()
    df['agent_attention'] = [i for i in range(agent_num * 2)]
    for key, value in zip(data_dict.keys(), data_dict.values()):
        df[key] = value
    df.to_csv(new_file, mode='a', index=False)

def smoothing(y_vals, w=0.99):
    smoothed = []
    last_val = y_vals[0]
    for i in range(len(y_vals)):
        smooth_val = last_val * w + (1 - w) * y_vals[i]
        smoothed.append(smooth_val)
        last_val = smooth_val
    return smoothed

def get_curve(path_url, model_set, y_label='win_rate', save_url='test.png'):
    for model_name in model_set:
        data1 = pd.read_csv(path_url + 'iql_' + model_name + '_10_seed1.csv')
        seed1 = data1['seed(1)' + y_label].values
        data2 = pd.read_csv(path_url + 'iql_' + model_name + '_10_seed2.csv')
        seed2 = data2['seed(2)' + y_label].values
        data3 = pd.read_csv(path_url + 'iql_' + model_name + '_10_seed3.csv')
        seed3 = data3['seed(3)' + y_label].values
        data4 = pd.read_csv(path_url + 'iql_' + model_name + '_10_seed4.csv')
        seed4 = data4['seed(4)' + y_label].values
        max_values = []
        mean_values = []
        min_values = []
        index = data1['episode'].values

        for s1, s2, s3, s4 in zip(seed1, seed2, seed3, seed4):
            max_values.append(max(s1, s2, s3, s4))
            mean_values.append(sum([s1, s2, s3, s4]) / 4)
            min_values.append(min(s1, s2, s3, s4))

        if y_label == 'total_reward':
            max_values = smoothing(max_values)
            mean_values = smoothing(mean_values)
            min_values = smoothing(min_values)

        x_smooth = np.linspace(index.min(), index.max(), 300)
        y_smooth1 = make_interp_spline(index, max_values)(x_smooth)
        y_smooth2 = make_interp_spline(index, mean_values)(x_smooth)
        # y_smooth4 = make_interp_spline(index, smoothed)(x_smooth)
        y_smooth3 = make_interp_spline(index, min_values)(x_smooth)
        # plt.plot(x_smooth, y_smooth1)
        plt.plot(x_smooth, y_smooth2, label=model_name)
        # plt.plot(x_smooth, y_smooth3)
        plt.fill_between(x_smooth, y_smooth1, y_smooth3, alpha=0.2)

    plt.xlabel('episode')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_url)
    plt.close()

def get_matrix_fig(path_url, save_url='test.png'):
    file_names = os.listdir(path_url)
    file_names.reverse()
    index = 1
    for file_name in file_names:
        data = pd.read_csv(path_url + '/' + file_name)
        data = data.iloc[:, 1:]
        data = data.values
        fig, ax = plt.subplots()
        im = ax.matshow(data)
        ax.set_xticks(np.arange(len(x_trick)))
        ax.set_yticks(np.arange(len(y_trick)))

        ax.set_xticklabels(x_trick)
        ax.set_yticklabels(y_trick)

        plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode="anchor")

        fig.tight_layout()
        plt.savefig('test' + str(index) + '.png')
        index += 1
        plt.close()
    
if __name__ == '__main__':
    # get_curve(path_url, model_set)
    get_matrix_fig(path_url + 'iql_agent_level_weight')