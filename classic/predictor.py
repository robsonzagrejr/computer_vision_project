from sklearn.linear_model import SGDClassifier
import skimage.transform as skit
import skimage.filters as skif
import skimage.color as skic
import skimage as ski
import cv2
import joblib

import pickle
import json
import skimage.io as skio

import matplotlib.pyplot as plt

import classic.features as f
from classic.config import (
    img_size,
    params_path,
    model_path,
    windows_chunks_definition
)

class ClassicPredictor():

    def __init__(self):
        self.model = pickle.load(open(model_path,'rb'))
        self.best_params = json.load(open(params_path,'r'))
        windows_chunks = []
        for val in windows_chunks_definition.values():
            windows_chunks += self.define_windows(**val)
        self.windows = windows_chunks


    def transform_img(self, img):
        img = skic.rgba2rgb(img)
        img = skit.resize(img, img_size, anti_aliasing=True)
        return skif.gaussian(img, sigma=1.2, cval=0.4, multichannel=True)


    def predict_chunk(self, img_c):
        img_c = skit.resize(img_c, (64,64), anti_aliasing=True)
        params = {
            'X': [img_c],
            'train': False,
            **self.best_params
        }
        features = f.pipeline(**params)
        y_pred = self.model.predict(features)
        return y_pred


    def predict(self, image):
        image_t = self.transform_img(image)
        
        def _predict_chunk(window):
            print(window)
            start, end = window
            y_pred = self.predict_chunk(image_t[start[0]:end[0], start[1]:end[1],:].copy())
            if y_pred:
                print(f"CAR IN -> {start}, {end}")
                #skio.imshow(image_t[start[0]:end[0], start[1]:end[1],:])
                #plt.show()
                #cv2.rectangle(image, (start[1], start[0]), (end[1], end[0]), (255,0,0), 2)
                
        features=joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_predict_chunk)(
                window                
            ) for window in self.windows
        )

        return image


    def define_windows(self, scale, bottom, draw=False, color=(255,0,0), line_size=5, img=None):
        bottom = int(bottom)
        size = int(64*scale)
        n_top = int(bottom - size)
        step = int(64*scale/4)
        stop = int(img_size[1] - size)
        windows = []
        for d in range(0, stop, step):
            start = (n_top, d)
            end = (bottom, d+size)
            windows.append((start,end))
            if draw:
                cv2.rectangle(img, (d, n_top), (d+size, bottom), color, line_size)
        return windows
