import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model import MLPdynamics, Transformerdynamics
from src.dataset_loader import F1TENTH_Dataset, F1TENTH_DataLoader

import hydra
from src.logger import Logger
import src.utils as utils
from tqdm import tqdm
import pandas as pd

def metrics(pred_next_state, next_state):
    #loss = (pred_next_state - next_state) ** 2
    loss_1 = (pred_next_state[..., :2] - next_state[..., :2]) ** 2

    loss_2 = (pred_next_state[..., 2:3] - next_state[..., 2:3]) % (2 * np.pi)
    loss_2 = torch.min(loss_2, 2 * np.pi - loss_2) ** 2

    loss_3 = (pred_next_state[..., 3:] - next_state[..., 3:]) ** 2

    return torch.cat([loss_1, loss_2, loss_3], axis=1).mean(axis=0)
    #return loss.mean(axis=0)


def read_single_csv_data(data_path_file):
    """
        the State has the following format:
                    'delta_time',
                    'pose_position_x',
                    'pose_position_y',
                    'yaw',
                    'body_vx',
                    'body_vy',
                    'yaw_rate',
                    'linear_acceleration_x',
                    'linear_acceleration_y',

        the action has the following format:
                        'cmd_st_angle',
                        'cmd_speed'
    """
    from src.dataset_loader import columns_to_read
    df = pd.read_csv(data_path_file)[columns_to_read]
    df['yaw'] = (df['yaw'] + np.pi) % (2 * np.pi) - np.pi
    df = df.drop(df.index[0]).to_numpy()

    return df


def open_loop_simulation_eval(model, data_path_file):
    data = read_single_csv_data(data_path_file)
    print(data.shape)

    start_idx = 50
    end_idx = 3000

    points = []
    gts = []

    cur_states = data[start_idx - 3: start_idx + 1, :9].copy()
    actions = data[start_idx - 3: start_idx + 1, 9:].copy()

    for i in range(start_idx, end_idx):
        # pred_next state only has 5 columns
        pred_next_state = model(cur_states.reshape(1, -1, 9),
                                actions.reshape(1, -1, 2))
        points.append(pred_next_state)
        gts.append(data[i+1][1:6])

        next_state_b = data[i+1][:9].copy()
        next_state_b[1:6] = pred_next_state
        
        cur_states = np.vstack([cur_states[1:], next_state_b[:9]])
        actions = data[i-2: i+2, 9:]

    return np.array(points), np.array(gts)

def world2local(states):
    states = states.clone()
    states[:, :, 1:4] = states[:, :, 1:4] - states[:, 0:1, 1:4]
    return states

def local2world(old_state, states):
    states = states.clone()
    states[:, :, 0:3] = states[:, :, 0:3] + old_state[:, 0:1, 1:4]
    return states


def open_loop_simulation_eval_nn(model, data_path_file):
    data = read_single_csv_data(data_path_file)
    print(data.shape)

    start_idx = 50
    end_idx = 500

    points = []
    gts = []

    cur_states = data[start_idx - 3: start_idx + 1, :9].copy()
    actions = data[start_idx - 3: start_idx + 1, 9:].copy()

    cur_states = torch.tensor(cur_states, dtype=torch.float32).unsqueeze(0)
    actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)


    for i in range(start_idx, end_idx):
        # pred_next state only has 5 columns

        _cur_states = world2local(cur_states)
        pred_next_state = model(_cur_states,
                                actions)

        pred_next_state_ = local2world(cur_states, pred_next_state.unsqueeze(1)).squeeze()
        points.append(pred_next_state_.detach().numpy())
        gts.append(data[i+1][1:6])

        next_state_b = torch.tensor(data[i+1][:9], dtype=torch.float32)
        next_state_b[1:6] = pred_next_state_
        
        cur_states = torch.cat([cur_states[:, 1:], next_state_b[:9].unsqueeze(0).unsqueeze(0)], dim=1)
        actions = torch.tensor(data[i-2: i+2, 9:], dtype=torch.float32).unsqueeze(0)

    return np.array(points), np.array(gts)


def evaluate_single_benchmark(model, data_path_file):
    rootdir = '/Users/mac/Desktop/PENN/f1tenth-final-project'

    test_dataset = F1TENTH_Dataset(
                                    root_dir=rootdir,
                                    data_path_file=data_path_file,
                                    state_dim=9,
                                    act_dim=2,
                                    history_len=2,
                                    normalize=False)
        
    test_loader = F1TENTH_DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    losses = []
    for eval_batch in tqdm(test_loader):
        state, action, next_state = eval_batch

        next_state[:, 2] = (next_state[:, 2] + np.pi) % (2 * np.pi) - np.pi
        #state = state.to(model.device)
        #action = action.to(model.device)
        #next_state = next_state.to(model.device)

        with torch.no_grad():
            pred_next_state = model(state, action)
            #loss = nn.MSELoss()(pred_next_state, next_state)
            loss = (pred_next_state - next_state.squeeze()).pow(2)
            losses.append(loss.numpy())
            if np.mean(losses) > 2:
                print(f'Loss looks too high: {np.mean(losses)}')
    
    # print each column MSE
    print('MSE for each column')
    print("px, py, yaw, vx, vy")
    print(np.mean(losses, axis=0))

    print('MSE for all columns')
    print(np.mean(losses))

    return np.mean(losses, axis=0)


def evaluate_NN():
    model = MLPdynamics(
                        state_dim=9,
                        act_dim=2,
                        out_dim=5,
                        history_len=2,
                        hidden_dim=256)

    model.load_state_dict(torch.load('/Users/mac/Desktop/PENN/f1tenth-final-project/models/mlp_model.pth'))
    model.eval()

    evaluate_single_benchmark(model, 'data/test_data.txt')
    evaluate_single_benchmark(model, 'data/train_data.txt')

def evaluate_math(num='3'):
    model = lambda x, a: utils.single_track_dynamics(x, a, mode=num)

    losses = []
    for i in range(0, 4):
        loss = evaluate_single_benchmark(model, f'data/data_mu10/state_list{i}.txt')
        loss_ = list(loss)
        loss_ += [np.mean(loss)]
        losses += loss_
    return losses

def export_csv():
    import csv
    loss_per_model = []
    for i in range(1, 4):
        print(f'Evaluate Dymaics model {i}')
        loss_per_model.append(evaluate_math(str(i)))

    with open('result3.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'px', 'py', 'yaw', 'vx', 'vy', 'mean',
                                  'px', 'py', 'yaw', 'vx', 'vy', 'mean',
                                  'px', 'py', 'yaw', 'vx', 'vy', 'mean',
                                  'px', 'py', 'yaw', 'vx', 'vy', 'mean'])

        for i, loss in enumerate(loss_per_model):
            loss = [f'model_{i}'] + loss
            writer.writerow(loss)

def eval_open_loop():
    model = lambda x, a: utils.single_track_dynamics(x, a, mode='1')

    points, gts = open_loop_simulation_eval(model, 'data/data_mu10/state_list_4.csv')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], label='Predicted')
    ax.scatter(gts[:, 0], gts[:, 1], label='Ground Truth')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    eval_open_loop()