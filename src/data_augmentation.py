'''
Script to augment the data for increased nmber of samples using the CSv files saved in data/mergedf

'''
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_delta_time(data, interval):
    '''
    Script to create the delta time for the data where the (i+interval)th row - ith row is the time difference
    '''

    # Create a new column for delta time
    data['delta_time'] = 0

    # Convert the timestamp column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Iterate over the rows in the data
    for i in range(len(data) - interval):
        # Calculate the time difference between the (i+interval)th row and the ith row
        time_diff = data['timestamp'][i + interval] - data['timestamp'][i]

        # Convert the time difference to seconds
        time_diff_seconds = time_diff.total_seconds()

        # Assign the time difference to the delta_time column
        data.at[i, 'delta_time'] = time_diff_seconds

    # Drop the rows where the delta_time is 0
    data = data[data['delta_time'] != 0]

    return data

def main():

    # Read arguments from aurgmentation.yaml file
    config = 'cfgs/augmentation.yaml'
    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    input_file = params['input_file']
    output_file = params['output_file']
    interval = params['interval']

    # Read the CSV file
    df = pd.read_csv(input_file)


    # Create the delta time column
    create_delta_time(df, interval)

    # Save the augmented data to a new CSV file
    df.to_csv(output_file, index=False)

    print(f'Data augmented and saved to {output_file}')

if __name__ == '__main__':
    main()










