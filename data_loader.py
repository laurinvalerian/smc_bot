import pandas as pd
import os
import sys
import glob


def load_histdata_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=',',
        header=None,
        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
    )
    df['timestamp'] = pd.to_datetime(
        df['date'] + ',' + df['time'], format="%Y.%m.%d,%H:%M"
    )
    df.drop(columns=['date', 'time'], inplace=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.ffill()
    return df


def load_all_pairs(folder_path: str) -> dict:
    data_dict = {}
    for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
        key = os.path.splitext(os.path.basename(file_path))[0]
        data_dict[key] = load_histdata_csv(file_path)
    return data_dict


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else 'data'
    data_dict = load_all_pairs(folder)
    print("Loaded:", len(data_dict), "Pairs")
    if 'AUDUSD_2018' in data_dict:
        print(data_dict['AUDUSD_2018'].head())
