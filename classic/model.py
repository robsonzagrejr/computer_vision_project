import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import json

import classic.features as f

seed = 72
class CustomSGDClassifier(SGDClassifier):

    def __init__(self, color_type='HSL', orient=9, pix_per_cell=8, cell_per_block=2,
                    block_norm='L1', transform_sqrt=True, bins=32):
        self.color_type=color_type
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt
        self.bins = bins
        SGDClassifier.__init__(self)


    def transform(self, X, y):
        X = self.get_features(X)
        return X, y


    def fit(self, X, y):
        if 'image' in X.columns:
            X, y = self.transform(X, y)
        return SGDClassifier.fit(self, X, y)
 

    def fit_transform(self, X, y):
        X_t, y_t = self.transform(X, y)
        return self.fit(X_t,y_t)


    def predict(self, X):
        if 'image' in X.columns:
            X, _ = self.transform(X, None)
        return SGDClassifier.predict(self, X)


    def predict_proba(self, X):
        if 'image' in X.columns:
            X, _ = self.transform(X, None)
        return SGDClassifier.predict_proba(self, X)


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
    print("Search for Best Params")
    # Use a custom model to define best params used in pipeline of feature generate for model 
    parameters = dict(
        color_type=['HSL','HSV'],
        orient=[7,8],
        pix_per_cell=[4,8],
        cell_per_block=[2,4,6],
        block_norm=['L1','L2'],
        transform_sqrt=[False,True],
        bins=[16,32]
    )
    model = CustomSGDClassifier()
    modelgscv = GridSearchCV(model, parameters, n_jobs=-1, refit=True, return_train_score=True)
    #Look just for a part because of computer complexit and RAM
    modelgscv.fit(X_train.iloc[0:50], y_train.iloc[0:50])
    with open('data/feature/best_param.json', 'w', encoding='utf-8') as f:
        json.dump(modelgscv.best_params_, f, ensure_ascii=False, indent=4)
    return modelgscv.best_params_


def define_best_params(X_train, y_train, search=False):
    if search:
        return search_for_good_pipe_params(X_train, y_train)
    # Previus best result
    return {
        "bins": 16,
        "block_norm": "L2",
        "cell_per_block": 2,
        "color_type": "HSL",
        "orient": 7,
        "pix_per_cell": 8,
        "transform_sqrt": False
    }


def define_model():
    return SGDClassifier()


def batch(iterable_X, iterable_y, n=1):
    l = len(iterable_X)
    for ndx in range(0, l, n):
        yield iterable_X[ndx:min(ndx + n, l)], iterable_y[ndx:min(ndx + n, l)]


def train_model(X_train, y_train, search=False):
    print("Model ...")
    print("Define Best Params")
    best_params = define_best_params(X_train, y_train, search)
    model = define_model()
    
    #base_index = 0
    #index = 1000
    #bach = 0
    ROUNDS = 20
    for bach in range(ROUNDS):
        print(f"Training Epoch {bach}")
        batcherator = batch(X_train, y_train, 10)
        for index, (chunk_X, chunk_y) in enumerate(batcherator):
            train_params = {
                'X': chunk_X,
                **best_params
            }
            chunk_X_n = f.pipeline(**train_params)
            pd.DataFrame(chunk_X_n).to_csv(f'data/feature/x_train_n_{bach}.csv')
            model.partial_fit(chunk_X_n, chunk_y, classes=[0, 1])
    # while index < len(X_train):
    #     print(f"Training Epoch {bach}")
    #     train_params = {
    #         'X': X_train.iloc[base_index:index],
    #         **best_params
    #     }
    #     X_train_n = f.pipeline(**train_params)
    #     pd.DataFrame(X_train_n).to_csv(f'data/feature/x_train_n_{bach}.csv')
    #     model.partial_fit(X_train_n, y_train.iloc[base_index:index])

    #     base_index = index
    #     index += 1000
    #     bach += 1
    # if base_index < len(X_train):
    #     print(f"Training Epoch {bach}")
    #     train_params = {
    #         'X': X_train.iloc[base_index:],
    #         **best_params
    #     }
    #     X_train_n = f.pipeline(**train_params)
    #     pd.DataFrame(X_train_n).to_csv(f'data/feature/x_train_n_{bach}.csv')
    #     model.partial_fit(X_train_n, y_train.iloc[base_index:])

    pickle.dump(model, open('data/model/hog_sgd_model.pkl', 'wb'))
    return model



