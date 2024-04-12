import os
import csv
import numpy as np
import torch

"""
This module contains the dataset and its loader for the F1TENTH Gym environment.
The dataset is a collection of states and actions of the car in the past T time steps.
Data look like as follows:
    state: [x, y, theta, velocity_x, velocity_y]
    action: [cmd_v_x, steer]
"""

class F1TENTH_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                    root_dir, 
                    data_path_file, 
                    state_dim=5, 
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

        state = torch.tensor(data[idx:idx + self.history_len][:, :self.state_dim], dtype=torch.float32)
        action = torch.tensor(data[idx:idx + self.history_len][:, self.state_dim:], dtype=torch.float32)
        next_state = torch.tensor(data[idx + self.history_len][:self.state_dim], dtype=torch.float32)
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
    dataset = F1TENTH_Dataset('data.txt', history_len=14)
    print(len(dataset))
    dataloader = F1TENTH_DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (state, action, next_state) in enumerate(dataloader):
        print("idx={} ".format(i), state, action, next_state)
        if i > 10:
            break