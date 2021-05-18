from numpy import dtype, float32
import classic.data as cdata
import classic.features as cfeatures
import classic.model as cmodel
import classic.predictor as cpredictor
import skimage.io as skio

import joblib
import cv2

import matplotlib.pyplot as plt

import time


def create_classic_model():
    print("===========TRAIN CLASSIC MODEL===========")
    print("Loading data...")
    df = cdata.load_data()
    print(df.head())
    print("Split data...")
    X_train, X_test, y_train, y_test = cdata.split_train_test(df)
    cmodel.train_model(X_train, y_train, search=False, features=True)


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
    img = []
    img.append(skio.imread('data/game/image/img1.png'))
    # img.append(skio.imread('data/game/image/img2.png'))
    # img.append(skio.imread('data/game/image/img3.png'))
    # img.append(skio.imread('data/game/image/img4.png'))
    # img.append(skio.imread('data/game/image/img5.png'))
    
    predictor = cpredictor.ClassicPredictor()

    #start = time.time()
    #joblib.Parallel(n_jobs=4)(joblib.delayed(predictor.predict)(i) for i in img)
    car_chunks = []
    images = []
    for i in img:
        car_chunk, image, image_s = predictor.predict(i, show_find=True)
        car_chunks.append(car_chunk)
        images.append(image_s)
    #end = time.time()
    
    #print(f"Class Time: {end - start}")
    for i in images:
        skio.imshow(i)  
        skio.show()

