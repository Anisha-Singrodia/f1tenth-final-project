import os
import csv
import numpy as np
import torch
import pandas as pd

"""
This module contains the dataset and its loader for the F1TENTH Gym environment.
The dataset is a collection of states and actions of the car in the past T time steps.
Data look like as follows:
    state: [x, y, theta, velocity_x, velocity_y]
    action: [cmd_v_x, steer]
"""

columns_to_read = [
                        'delta_time',
                        'pose_position_x',
                        'pose_position_y',
                        'yaw',
                        'body_vx',
                        'body_vy',
                        'yaw_rate',
                        'linear_acceleration_x',
                        'linear_acceleration_y',
                        'cmd_st_angle',
                        'cmd_speed'
                    ]
# pos scale
pos_scale = 0.2247
yaw_scale = 0.0680

yaw_scale0 = np.array([0.46065483580895444, 0.6485790957547133, 0.7903214475683393, 0.908143076575603, 1.0100008083952738, 1.1007363179462473, 1.1828597010481563, 1.2581381835168957, 1.327482531107164, 1.3920601819145726, 1.4526270470966476, 1.509516114551529, 1.563342314992721, 1.6143269669009532, 1.662695162182545, 1.7088444428263538, 1.750900007051592, 1.7890680567869426, 1.8255969170019826, 1.8607803907100746])


# for body_vx to cmd_speed
state_means = np.array([1.4369481655474188,  -0.0009814214153652265,
                         -0.02315479468686368, 
                        -0.01802335693380986,  -0.13127789561111358, 
                        -0.022248513476725334, 1.6854859023885282])
state_stds = np.array([0.6540768044967312, 0.4019855271814585, 
                       24.784068767873386, 
                       0.36096510882634714, 0.3005877881112919, 
                       0.29410302373908653, 0.5996559876284685])

class F1TENTH_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                    root_dir, 
                    data_path_file, 
                    state_dim=9, 
                    act_dim=2, 
                    history_len=10):        
        self.root_dir = root_dir
        self.history_len = history_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.data = self.read_data(os.path.join(root_dir, data_path_file))

    def read_data(self, data_path_file):
        # data_path_file is the txt file that contains the path to the csv file
        # that contains the data
        # read the data from csv file in the directory
        # Note that the data is stored in the csv file in the following format:
        # x, y, yaw, velocity_x, velocity_y, yaw_rate, a_x, a_y, cmd_v_x, steer
        data = {}
        with open(data_path_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                line = os.path.join(self.root_dir, line)
                data[i] = pd.read_csv(line)[columns_to_read]
                # drop first row
                data[i] = data[i].drop(data[i].index[0]).to_numpy()

        self.data_cum_idx_map = [len(d) - self.history_len for d in data.values()]
        self.data_cum_idx_map = [0] + self.data_cum_idx_map
        self.data_cum_idx_map = np.cumsum(self.data_cum_idx_map)
        return data

    def old_read_data(self, data_path_file):
        # data_path_file is the txt file that contains the path to the csv file
        # that contains the data
        # read the data from csv file in the directory
        # Note that the data is stored in the csv file in the following format:
        # x, y, theta, velocity_x, velocity_y, cmd_v_x, steer
        data = {}
        with open(data_path_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                line = os.path.join(self.root_dir, line)
                data[i] = np.loadtxt(line, delimiter=',', skiprows=0).astype(np.float32)

        self.data_cum_idx_map = [len(d) - self.history_len for d in data.values()]
        self.data_cum_idx_map = [0] + self.data_cum_idx_map
        self.data_cum_idx_map = np.cumsum(self.data_cum_idx_map)
        return data

    def __len__(self):
        return self.data_cum_idx_map[-1]

    def __getitem__(self, _idx):
        """
        return:
            state: [Batch_size, history_len, state_dim]
            action: [Batch_size, history_len, act_dim]
            next_state: [Batch_size, state_dim] 
        """
        file_idx = np.searchsorted(self.data_cum_idx_map, _idx, side='right') - 1
        idx = _idx - self.data_cum_idx_map[file_idx]

        data = self.data[file_idx]

        # Normalize the state
        times = data[idx:idx + self.history_len+1][:, 0:1]

        ## For position and yaw, switch to the relative position
        pos_yaw = data[idx:idx + self.history_len+1][:, 1:4]
        pos_yaw = pos_yaw - pos_yaw[0]

        ## scale pos and yaw
        # pos scale increases by the index
        pos_scale_cur = np.arange(1, self.history_len+1) * pos_scale
        yaw_scale_cur = np.arange(1, self.history_len+1) * yaw_scale
        pos_yaw[1:, :2] = pos_yaw[1:, :2] / pos_scale_cur[:, None]
        pos_yaw[1:, 2] = pos_yaw[1:, 2] / yaw_scale_cur

        ## scale velocity to cmd
        left_state = data[idx:idx + self.history_len+1][:, 4:]
        left_state = (left_state - state_means) / state_stds

        df = np.concatenate([times, pos_yaw, left_state], axis=1)


        state = torch.tensor(df[:-1][:, :self.state_dim], dtype=torch.float32)
        action = torch.tensor(df[:-1][:, self.state_dim:], dtype=torch.float32)
        next_state = torch.tensor(df[-1][:self.state_dim], dtype=torch.float32)
        return state, action, next_state

class F1TENTH_DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        super(F1TENTH_DataLoader, self).__init__(dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=shuffle,
                                                 num_workers=num_workers)

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return super(F1TENTH_DataLoader, self).__iter__()


if __name__ == '__main__':
    dataset = F1TENTH_Dataset('/Users/mac/Desktop/PENN/f1tenth-final-project', 'data/test_data.txt', history_len=5)
    print(len(dataset))
    dataloader = F1TENTH_DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    for i, (state, action, next_state) in enumerate(dataloader):
        print("idx={} ".format(i), state, action, next_state)
        if i > 2:
            break