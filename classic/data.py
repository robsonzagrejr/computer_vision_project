import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import skimage as ski
import skimage.io as skio
import skimage.viewer as skiv

from matplotlib import pyplot as plt
import glob
import joblib
import pickle

seed = 72


def read_image(file_dir, target):
    img = skio.imread(file_dir)
    return {'image': img, 'target': target}


def create_data():
    df = pd.DataFrame(columns = ['image', 'target'])
    target = 'non-vehicle'
    non_vehicles_files = glob.glob(f'data/non-vehicles/**/*.png', recursive=True)
    non_vehicles_data = joblib.Parallel(n_jobs=-1)(joblib.delayed(read_image)(f, target) for f in non_vehicles_files)
    df = df.append(non_vehicles_data, ignore_index=True)
    target = 'vehicle'
    vehicles_files = glob.glob(f'data/vehicles/**/*.png', recursive=True)
    vehicles_data = joblib.Parallel(n_jobs=-1)(joblib.delayed(read_image)(f, target) for f in vehicles_files)
    df = df.append(vehicles_data, ignore_index=True)
    
    #shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    pickle.dump(df, open('data/data.pkl', 'wb'))
    return df


def load_data():
    return pickle.load(open('data/data.pkl', 'rb'))


def split_train_test(df):
    X = df[['image']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    y_train = (y_train == 'vehicle').astype(int)
    y_test = (y_test == 'vehicle').astype(int)

    return X_train, X_test, y_train, y_test
