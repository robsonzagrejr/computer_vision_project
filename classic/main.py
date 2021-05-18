from numpy import dtype, float32
import classic.data as cdata
import classic.features as cfeatures
import classic.model as cmodel
import classic.predictor as cpredictor
import skimage.io as skio

import matplotlib.pyplot as plt

import time


def create_classic_model():
    print("===========TRAIN CLASSIC MODEL===========")
    print("Loading data...")
    df = cdata.load_data()
    print(df.head())
    print("Split data...")
    X_train, X_test, y_train, y_test = cdata.split_train_test(df)
    cmodel.train_model(X_train, y_train, search=False)


def test_simple_classic_model():
    print("===========TEST CLASSIC MODEL===========")
    print("Loading data...")
    df = cdata.load_data()
    print("Split data...")
    X_train, X_test, y_train, y_test = cdata.split_train_test(df)
    model = cmodel.load_model()
    cmodel.test_model(model, X_test, y_test)


def test_game_classic_model():
    print("===========TEST GAME CLASSIC MODEL===========")
    img = skio.imread('data/game/image/img2.png')
    predictor = cpredictor.ClassicPredictor()
    
    start = time.time()
    img_p = predictor.predict(img)
    end = time.time()
    print(f"Time: {end - start}")
    skio.imshow(img_p)
    plt.show()