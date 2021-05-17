import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle
\   
from .features import as f

class CustomLogisticRegression(LogisticRegression):

    def __init__(self, color_type='HSL', orient=9, pix_per_cell=8, cell_per_block=2,
                    block_norm='L1', transform_sqrt=True, bins=32):
        self.color_type=color_type
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt
        self.bins = bins
        LogisticRegression.__init__(self,
                l1_ratio=0.1,
                solver='saga',
                penalty='elasticnet',
                random_state=seed,
                max_iter=100
        )


    def transform(self, X, y):
        X = self.get_features(X)
        return X, y


    def fit(self, X, y):
        if 'image' in X.columns:
            X, y = self.transform(X, y)
        return LogisticRegression.fit(self, X, y)
 

    def fit_transform(self, X, y):
        X_t, y_t = self.transform(X, y)
        return self.fit(X_t,y_t)


    def predict(self, X):
        if 'image' in X.columns:
            X, _ = self.transform(X, None)
        return LogisticRegression.predict(self, X)


    def predict_proba(self, X):
        if 'image' in X.columns:
            X, _ = self.transform(X, None)
        return LogisticRegression.predict_proba(self, X)


    def get_features(self, X):
        features=joblib.Parallel(n_jobs=-1)(
            joblib.delayed(f.img_feature)(
                image,
                self.color_type,
                self.orient,
                self.pix_per_cell,
                self.cell_per_block,
                self.block_norm,
                self.transform_sqrt,
                self.bins                     
            ) for image in X['image'].values
        )
        X_scaler = StandardScaler().fit(features)
        scaled_X = X_scaler.transform(features)
        return scaled_X


def search_for_good_pipe_params(X_train, y_train):
    # Use a custom model to define best params used in pipeline of feature generate for model 
    parameters = dict(
        color_type=['HSL','HSV'],
        orient=[7,8,9],
        pix_per_cell=[4,8,12],
        cell_per_block=[2,4,6],
        block_norm=['L1','L1-sqrt','L2'],
        transform_sqrt=[False,True],
        bins=[16,32]
    )
    log_reg = CustomLogisticRegression()
    log_reg_gscv = GridSearchCV(log_reg, parameters, n_jobs=-1, refit=True, return_train_score=True)
    #Look just for a part because of computer complexit and RAM
    log_reg_gscv.fit(X_train.iloc[0:50], y_train.iloc[0:50])
    return log_reg_gscv.best_params_


def define_best_params(X_train, y_train, search=False):
    if search:
        return search_for_good_pipe_params(X_train, y_train)
    # Previus best result
    return {
        'bins': 16,
        'block_norm': 'L2',
        'cell_per_block': 4,
        'color_type': 'HSL',
        'orient': 8,
        'pix_per_cell': 4,
        'transform_sqrt': False
    }


def define_model():
    return LogisticRegression(
        l1_ratio=0.1,
        solver='saga',
        penalty='elasticnet',
        random_state=seed,
        max_iter=100
    )


def train_model(X_train, y_train, search=False):
    best_params = define_best_params(X_train, y_train, search)
    model = define_model()
    train_params = {
        'X': X_train,
        **best_params
    }
    X_train_n = f.pipeline(**train_params)
    X_train_n.to_csv('data/feature/x_train_n.csv')
    model.fit(X_train_n, y_train)
    pickle.dump(model, open('data/model/hog_model.pkl', 'wb'))

    return model



