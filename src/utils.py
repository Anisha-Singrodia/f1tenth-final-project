import torch
import torch.nn as nn
import numpy as np
import random
import time


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def single_track_dynamics(state, action):
    """
    Single track model dynamics
    Args:
        state (torch.Tensor) : [x, y, theta, velocity_x], B*4
        action (torch.Tensor): [cmd_v_x, steer], B*2
    Returns:
        next_state (torch.Tensor): [x, y, theta, velocity_x, velocity_y]
    """
    # Constants
    L = 0.33  # length of the wheel base
    dt = 0.1  # time step

    x, y, theta, velocity_x, velocity_y = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    cmd_v_x, steer = action[:, 0], action[:, 1]

    # Update the state
    x += velocity_x * np.cos(theta) * dt
    y += velocity_y * np.sin(theta) * dt
    theta += (velocity_x / L) * np.tan(steer) * dt
    #velocity_x += cmd_v_x * np.cos(steer) * dt
    #velocity_y += cmd_v_x * np.sin(steer) * dt
    velocity_x = cmd_v_x
    velocity_y = 0.0

    next_state = torch.tensor([x, y, theta, velocity_x, velocity_y], dtype=torch.float32).to(state.device)
    return next_state


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()
        # Keep track of evaluation time so that total time only includes train time
        self._eval_start_time = 0
        self._eval_time = 0
        self._eval_flag = False

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time - self._eval_time
        return elapsed_time, total_time

    def eval(self):
        if not self._eval_flag:
            self._eval_flag = True
            self._eval_start_time = time.time()
        else:
            self._eval_time += time.time() - self._eval_start_time
            self._eval_flag = False
            self._eval_start_time = 0

    def total_time(self):
        return time.time() - self._start_time - self._eval_time
