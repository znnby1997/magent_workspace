import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import os


def get_csv(csv_url, data_dict):
    if os.path.exists(csv_url):
        df = pd.read_csv(csv_url)
    else:
        df = pd.DataFrame()
        df['epoch'] = [i for i in range(df.shape[0])]
    for key, value in zip(data_dict.keys(), data_dict.values()):
        df[key] = value
    df.to_csv(csv_url, index=False)

def smoothing(y_vals, w=0.99):
    smoothed = []
    last_val = y_vals[0]
    for i in range(len(y_vals)):
        smooth_val = last_val * w + (1 - w) * y_vals[i]
        smoothed.append(smooth_val)
        last_val = smooth_val
    return smoothed

def get_curve(path_url, model_dict: dict, y_label='total_reward', save_url='baseline_total_reward1.png', epoch_num=200, w=0.95,
                x_label='epoch'):
    index = np.array([j for j in range(epoch_num)])
    for model, seeds_data in zip(model_dict.keys(), model_dict.values()):
        seeds = []
        for i in range(len(seeds_data)):
            seed_url = path_url + seeds_data[i]
            if not os.path.exists(seed_url):
                continue
            seed_data = pd.read_csv(seed_url)

            seed_data = seed_data['seed(' + str(i) + ')' + y_label].values  # type: np.ndarray
            if seed_data.shape[0] > epoch_num:
                seed_data = seed_data[0:epoch_num]
            seeds.append(seed_data)
        seeds_vals = np.stack(seeds)
        mean = np.mean(seeds_vals, axis=0)
        std = np.std(seeds_vals, axis=0)
        
        low_bound = smoothing(mean - std / 2, w=w)
        up_bound = smoothing(mean + std / 2, w=w)
        mean = smoothing(mean, w=w)

        x_smooth = np.linspace(index.min(), index.max(), 300)
        y_smooth1 = make_interp_spline(index, low_bound)(x_smooth)
        y_smooth2 = make_interp_spline(index, mean)(x_smooth)
        y_smooth3 = make_interp_spline(index, up_bound)(x_smooth)
        plt.plot(x_smooth, y_smooth2, label=model)
        plt.fill_between(x_smooth, y_smooth1, y_smooth3, alpha=0.2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_url)
    plt.close()


def get_att_weight_csv(new_file, data_dict, agent_num):
    df = pd.DataFrame()
    df['agent_attention'] = [i for i in range(agent_num * 2)]
    for key, value in zip(data_dict.keys(), data_dict.values()):
        df[key] = value
    df.to_csv(new_file, mode='a', index=False)


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

def save_weight_info(step_id, alive_info, obs_info, weights, res_url, col_num, agent_num):
    step_flag = np.full([1, col_num], np.nan)
    df = pd.DataFrame(step_flag, index=['step_' + str(step_id)])
    df.to_csv(res_url, header=None, mode='a')

    agent_idx = [str(k) for k in range(agent_num)]

    res = []
    obs_idx = 0
    for i in range(agent_num):
        if alive_info[i]:
            # print('obs_info shape: ', obs_info[obs_idx][36:].shape, ' weights shape: ', weights[obs_idx].shape)
            res.append(np.hstack((obs_info[obs_idx][36:], weights[obs_idx])))
            obs_idx += 1
        else:
            res.append(np.full((col_num, ), np.nan))
    res = np.vstack(res)
    res_df = pd.DataFrame(res, index=agent_idx)
    res_df.to_csv(res_url, header=None, mode='a')

    
if __name__ == '__main__':
    # get_curve(path_url, model_set)
    get_matrix_fig(path_url + 'iql_agent_level_weight')