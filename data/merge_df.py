# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import os
import pathlib
import re

# %%
def read_datas(dir_path):
    # iterate over all csv files in the directory
    output_data = {}

    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            # read the csv file
            df = pd.read_csv(os.path.join(dir_path, file))

            # extract the topic name of the file, [bag_name]_[topic_name]_topic.csv
            output_data[re.search(r"_(.*)_topic.csv", file).group(1)] = df

    return output_data

def print_info_from_datas(datas):
    for k, v in datas.items():
        print(f"Topic: {k}")
        print(f"Shape: {v.shape}")
        print(f"Head: {v.head(5)}")
        print("\n")


# %%
def estimate_frequency(data):
    time = data['timestamp']
    time_diff = time.diff()
    time_diff = time_diff[1:]
    return  time_diff.dt.total_seconds().mean()

def convert_timestamps(data):
    return pd.to_datetime(data, unit='ns')

def merge_imu_pose(imu, vicon_pose):
  vicon_pose['timestamp'] = convert_timestamps(vicon_pose['timestamp'])
  imu['timestamp'] = convert_timestamps(imu['timestamp'])

  print("IMU shape: ", imu.shape)
  print("Vicon Pose shape: ", vicon_pose.shape)
  print("IMU frequency: ", estimate_frequency(imu))
  merged_df = pd.merge_asof(
                            imu.sort_values('timestamp'), 
                          vicon_pose.sort_values('timestamp'), 
                            on='timestamp', 
                            tolerance=pd.Timedelta('20ms'), 
                            direction='forward')
  
  merged_df.interpolate(method='linear', inplace=True)
  merged_df.dropna(inplace=True)
  return merged_df

# %% [markdown]
# ## Test

# %%
def add_derivative_cmd_columns(df):
    df['cmd_st_angle_rate'] = df['cmd_st_angle'].diff() / df['timestamp'].diff().dt.total_seconds()
    df['cmd_acceleration'] = df['cmd_speed'].diff() / df['timestamp'].diff().dt.total_seconds()
    df.bfill(inplace=True)
    return df

def upsample_df(df, freq='20ms'):
    df.set_index('timestamp', inplace=True)
    df_upsampled = df.resample(freq).bfill()
    return df_upsampled

# Add yaw and slip angle to the final_df
def add_yaw_and_slip_angle(df):
    r = R.from_quat(df[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].values)
    yaw = r.as_euler('zyx')[:, 0]
    df['yaw'] = yaw
    df['yaw_rate'] = df['yaw'].diff() / df['timestamp'].diff().dt.total_seconds()
    df['yaw_rate'].bfill(inplace=True)

    beta = np.arctan2(df['twist_linear_y'], df['twist_linear_x']) - yaw
    df['beta'] = beta

    return df


# %%
def add_body_v(df):
    df['body_vx'] = df['twist_linear_x'] * np.cos(df['yaw']) + df['twist_linear_y'] * np.sin(df['yaw'])
    df['body_vy'] = -df['twist_linear_x'] * np.sin(df['yaw']) + df['twist_linear_y'] * np.cos(df['yaw'])
    return df

def add_delta_time(df):
    df['delta_time'] = df['timestamp'].diff().dt.total_seconds()
    df['delta_time'].bfill(inplace=True)
    return df

# %% [markdown]
# ## Merge ever

# %%
# We want to get two files
# 1. state.csv
# contains: px, py, yaw, vx, vy, beta (which is the slip angle)
# 2. control.csv
# contains: delta, a

def merge_bags(folder_path, out_path):
    #bag1 = read_datas("rosbag_process/csv_data/2")
    bag1 = read_datas(folder_path)

    # Merge vicon pose and imu data
    merged_df = merge_imu_pose(bag1['imu'], bag1['vicon_odom'])
    merged_df.info()

    # Add derivative columns to the command data
    bag1['drive']['timestamp'] = convert_timestamps(bag1['drive']['timestamp'])
    df_cmd = add_derivative_cmd_columns(bag1['drive'])
    df_cmd_upsampled = upsample_df(df_cmd, '20ms')
    df_cmd_upsampled.info()

    # Merge the merged_df with the upsampled command data
    final_df = pd.merge_asof(
        merged_df.sort_values('timestamp'), 
        df_cmd_upsampled.sort_values('timestamp'), 
        on='timestamp', 
        tolerance=pd.Timedelta('20ms'), 
        direction='forward')

    final_df.interpolate(method='linear', inplace=True)
    final_df = final_df.dropna()
    final_df = add_yaw_and_slip_angle(final_df)
    final_df = add_body_v(final_df)
    final_df = add_delta_time(final_df)

    final_df.info()

    name = folder_path.split('/')[-1]
    final_df.to_csv(os.path.join(out_path, f"final_{name}.csv"), index=False)

def main():
    all_data_folder = "rosbag_process/csv_data"
    out_folder = "rosbag_process/csv_data/mergedf"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for folder in os.listdir(all_data_folder):
        if folder == 'mergedf':
            continue
        folder_path = os.path.join(all_data_folder, folder)
        merge_bags(folder_path, out_folder)

if __name__ == "__main__":
    main()