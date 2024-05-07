import torch
import torch.nn as nn
import numpy as np
import random
import time

import warnings

def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity

def single_track_dynamics1(state, action, dt):
    """
    Single track model dynamics
    Args:
        state (torch.Tensor) : [x, y, theta, velocity_x, velocity_y], B*5
        action (torch.Tensor): [cmd_v_x, steer], B*2
    Returns:
        next_state (torch.Tensor): [x, y, theta, velocity_x, velocity_y]
    """
    # Constants
    L = 0.33  # length of the wheel base
    #dt = 0.1  # time step

    x, y, theta, velocity_x, velocity_y = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    #cmd_v_x, steer = action[:, 0], action[:, 1]
    cmd_v_x, steer = action[:, 1], action[:, 0]

    # Update the state
    x += velocity_x * np.cos(theta) * dt
    y += velocity_y * np.sin(theta) * dt
    theta += (velocity_x / L) * np.tan(steer) * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    #velocity_x += cmd_v_x * np.cos(steer) * dt
    #velocity_y += cmd_v_x * np.sin(steer) * dt
    velocity_x = cmd_v_x
    velocity_y = np.array([0.0])

    next_state = np.array([x, y, theta, velocity_x, velocity_y]).squeeze()
    #next_state = torch.tensor([x, y, theta, velocity_x, velocity_y], dtype=torch.float32).to(state.device)
    return next_state


def pid(speed, steer, current_speed, current_steer, max_sv=3.2, max_a=9.51, max_v=20.0, min_v=-5.0):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = np.array([0.0])

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if (vel_diff > 0):
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if (vel_diff > 0):
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv



def single_track_dynamics2(state, action, dt, last_steer):
    """
    Single track model dynamics
    Args:
        state (torch.Tensor) : [x, y, theta, velocity_x, velocity_y], B*5
        action (torch.Tensor): [cmd_v_x, steer], B*2

        last_steer (torch.Tensor): [steer], B*1
    Returns:
        next_state (torch.Tensor): [x, y, theta, velocity_x, velocity_y]
    """
    x, y, theta, velocity_x, velocity_y = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    #cmd_v_x, steer = action[:, 0], action[:, 1]
    cmd_v_x, steer = action[:, 1], action[:, 0]

    accl, sv = pid(cmd_v_x, steer, velocity_x, last_steer, max_sv=3.2, max_a=9.51, max_v=20.0, min_v=-5.0)
    WB = 0.33

    accl = accl_constraints(velocity_x, accl, 7.319, 9.51, -5.0, 20.0)
    sv = steering_constraint(steer, sv, -0.4189, 0.4189, -3.2, 3.2)

    if isinstance(steer, torch.Tensor):
        atan2 = torch.atan2
        sin = torch.sin
        cos = torch.cos
        tan = torch.tan
    else:
        atan2 = np.arctan2
        sin = np.sin
        cos = np.cos
        tan = np.tan


    slip_angle = atan2(velocity_y, velocity_x)
    x += velocity_x * cos(theta + slip_angle) * dt
    y += velocity_y * sin(theta + slip_angle) * dt
    theta += (velocity_x / WB) * tan(steer) * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    v = (velocity_x**2 + velocity_y**2)**0.5
    slip_angle += (v * sv + steer * accl) * WB * dt

    new_v = v + accl * dt
    velocity_x = new_v * cos(theta + slip_angle)
    velocity_y = new_v * sin(theta + slip_angle)

    next_state = np.array([x, y, theta, velocity_x, velocity_y]).squeeze()
    #next_state = torch.tensor([x, y, theta, velocity_x, velocity_y], dtype=torch.float32).to(state.device)
    return next_state

def vehicle_dynamics_ks(x, u):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (5, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = 0.33


    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f



def vehicle_dynamics_st(x, u_init, 
                        mu=1.0489, 
                        C_Sf=4.718, C_Sr=5.4562, 
                        lf=0.15875, lr=0.17145, 
                        h=0.074, m=3.74, I=0.04712, 
                        ):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (7, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    s_min, s_max, sv_min, sv_max = -0.4189, 0.4189, -3.2, 3.2
    v_switch, a_max, v_min, v_max = 7.319, 9.51, -5.0, 20.0

    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), 
                  accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        lwb = lf + lr
        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))
    else:
        f = np.array([x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f

def single_track_dynamics3(state, action, dt, last_steer):
    """
    Single track model dynamics
    Args:
        state (torch.Tensor) : [x, y, yaw, velocity_x, velocity_y, yaw_rate], B*6
        action (torch.Tensor): [cmd_v_x, steer], B*2

        last_steer (torch.Tensor): [steer], B*1
    Returns:
        next_state (torch.Tensor): [x, y, theta, velocity_x, velocity_y]
    """

    x, y, yaw, velocity_x, velocity_y, yaw_rate = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
    #steer, cmd_v_x = action[:, 0], action[:, 1]
    cmd_v_x, steer = action[:, 1], action[:, 0]

    v = (velocity_x**2 + velocity_y**2)**0.5
    slip_angle = np.arctan2(velocity_y, velocity_x) #if abs(v) > 0.1 else np.array([0.0])

    accl, sv = pid(cmd_v_x, steer, velocity_x, last_steer, max_sv=3.2, max_a=9.51, max_v=20.0, min_v=-5.0)

    x = np.concatenate([x, y, last_steer, v, yaw, yaw_rate, slip_angle], axis=-1)
    u = np.concatenate([sv, accl], axis=-1)

    delta_next_state = vehicle_dynamics_st(x.T, u.T).T

    # px, py, st, v, yaw, yaw_rate, slip_angle
    next_state = x + delta_next_state * dt
    v_x = next_state[3] * np.cos(next_state[6])
    v_y = next_state[3] * np.sin(next_state[6])

    yaw = (next_state[4] + np.pi) % (2 * np.pi) - np.pi


    next_state_ = np.array([next_state[0], next_state[1], yaw, v_x, v_y]).T
    #next_state_ = torch.tensor([next_state[:, 0], next_state[:, 1], next_state[:, 4], v_x, v_y], dtype=torch.float32).to(state.device)
    return next_state_


def single_track_dynamics(_states, _actions, mode='1'):
    """
    Input:
    state: [Batch_size, history_len, state_dim]
    action: [Batch_size, history_len, act_dim]
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
    if isinstance(_states, torch.Tensor):
        device = _states.device
        states = _states.cpu().detach().numpy()
        actions = _actions.cpu().detach().numpy()
    else:
        states = _states
        actions = _actions

    actions = actions[:, -1, -2:]
    dts = states[:, -1, 0]

    if mode == '1':
        cur_states = states[:, -1, 1:6]
        next_state = single_track_dynamics1(cur_states, actions, dts)
    elif mode == '2':
        cur_states = states[:, -1, 1:6]
        last_steer = states[:, -2, -2]
        next_state = single_track_dynamics2(cur_states, actions, dts, last_steer)
    elif mode == '3':
        cur_states = states[:, -1, 1:7]
        last_steer = states[:, -2, -2]
        next_state = single_track_dynamics3(cur_states, actions, dts, last_steer)

    if isinstance(_states, torch.Tensor):
        return torch.tensor(next_state, dtype=torch.float32, device=device)
    else:
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
