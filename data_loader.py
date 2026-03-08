import pandas as pd
import os
import glob


def clean_pair_name(filename: str) -> str:
    """Convert e.g. 'DAT_MT_USDJPY_M1_2025.csv' to 'USDJPY_2025'.

    Assumes the currency pair is exactly 6 uppercase letters (standard forex pairs).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    pair = next((p for p in parts if p.isalpha() and len(p) == 6), '')
    year = next((p for p in reversed(parts) if p.isdigit() and len(p) == 4), '')
    if pair and year:
        return f"{pair}_{year}"
    return base


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


def load_all_pairs() -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    print("🔍 Rekursive Suche...")
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    print(f"📁 Gefundene CSV-Dateien: {len(csv_files)}")

    data_dict = {}
    for file_path in csv_files:
        pair_name = clean_pair_name(file_path)
        print(f"   ✓ Lade {pair_name}")
        data_dict[pair_name] = load_histdata_csv(file_path)
    return data_dict


if __name__ == '__main__':
    data_dict = load_all_pairs()
    print("Loaded:", len(data_dict), "Pairs")
    for key in list(data_dict.keys())[:3]:
        print(f"\n{key}:")
        print(data_dict[key].head())
