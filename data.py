import kagglehub
import pandas as pd

def download_dataset():
    return kagglehub.dataset_download("kartik2112/fraud-detection")

def load_raw_data(path='/data'):
    if path is None:
        path = download_dataset()
    return pd.read_csv(f'{path}/fraudTrain.csv')

if __name__ == '__main__':
    load_raw_data()