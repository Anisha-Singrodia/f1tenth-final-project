import numpy as np
import pandas as pd

# read csv file and return a pandas dataframe
def read_csv_file(file_path):
    return pd.read_csv(file_path)

# write a pandas dataframe to a csv file
def write_csv_file(df, file_path):
    df.to_csv(file_path, index=False)

# concatenate two pandas dataframes
def concat_dataframes(df1, df2):
    return pd.concat([df1, df2], axis=1)

# add a column of 0s to a pandas dataframe
def add_column_of_zeros(df, column_name):
    df[column_name] = 0.0
    return df

def main():
    # read csv file
    df1 = read_csv_file('data/wp-2024-04-12-20-02-03.csv')
    df2 = read_csv_file('data/wp-2024-04-12-20-02-03_drive.csv')
    print(df1.head(), '\n', df2.head())
    # df1 = df1[:3000]
    # df2 = df2[:len(df1)]
    df1 = df1[3000:]
    df2 = df2[3000:][:len(df1)]
    print(len(df1), len(df2))

    # add a column of 0s
    df = add_column_of_zeros(df1, 'v_y')
    print(df.head())

    # concatenate dataframes
    dfin = concat_dataframes(df, df2[:len(df)])
    print(dfin.head())

    # write dataframe to csv file
    write_csv_file(dfin, 'data/test_data_wp.csv')

if __name__ == '__main__':
    main()