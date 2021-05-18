from sklearn.linear_model import SGDClassifier
import skimage.transform as skit
import skimage.filters as skif
import skimage.color as skic
import skimage as ski
import cv2
import joblib
import numpy as np
import time
import numba as nb


from skimage.feature import hog

import pickle
import json
import skimage.io as skio

import matplotlib.pyplot as plt

import classic.features as f
from classic.config import (
    img_size,
    x_scalar_path,
    params_path,
    model_path,
    windows_chunks_definition
)

def gpu_prediction(img, window):
    model = pickle.load(open(model_path,'rb'))
    params = {
        'X': [img_c],
        'train': False,
        **self.best_params
    }
    features = f.pipeline(**params)
    y_pred = self.model.predict(features)

class ClassicPredictor():

    def __init__(self):
        
        start_t = time.time()
        self.model = pickle.load(open(model_path,'rb'))
        end_t = time.time()
        print(f'Model-> {round(end_t - start_t, 40)}')
        self.best_params = json.load(open(params_path,'r'))
        windows_chunks = []
        for val in windows_chunks_definition.values():
            windows_chunks += self.define_windows(**val)
        self.windows = windows_chunks



    def transform_img(self, img):
        #img = skic.rgba2rgb(img)
        img = ski.img_as_float32(img)
        img = skit.resize(img, img_size, anti_aliasing=True)
        img = skif.gaussian(img, sigma=1.2, cval=0.4, multichannel=True)
        return 


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
        start_t = time.time()
        #image_t = self.transform_img(image)
        #img = skic.rgba2rgb(image)
        img = ski.img_as_float32(image)
        img = skit.resize(img, img_size, anti_aliasing=False)
        end_t = time.time()
        print(f'Transform time-> {end_t-start_t}')
        image_t = skif.gaussian(img, sigma=1.2, cval=0.4, multichannel=True)
        # def _predict_chunk(window):
        #     start_t = time.time()
        #     start, end = window
        #     y_pred = self.predict_chunk(image_t[start[0]:end[0], start[1]:end[1],:].copy())
        #     if y_pred:
        #         print(f"CAR IN -> {start}, {end}")
        #         #skio.imshow(image_t[start[0]:end[0], start[1]:end[1],:])
        #         #plt.show()
        #         #cv2.rectangle(image_c, (start[1], start[0]), (end[1], end[0]), (255,0,0), 2)
        #         print('ok')
            
        #     end_t = time.time()
        #     #print(f'{window} time-> {end_t-start_t}')
                
        # features=joblib.Parallel(n_jobs=-1)(
        #     joblib.delayed(_predict_chunk)(
        #         window                
        #     ) for window in self.windows
        # )
        # @nb.vectorize(nopython=False)
        # def _get_chunk(window, img):
        #     img = img[window[0][0]:window[1][0], window[0][1]:window[1][1],:]
        #     return img
        # start_t = time.time()
        # imgs=joblib.Parallel(n_jobs=-1)(
        #     joblib.delayed(_get_chunk)(
        #         window,
        #         img            
        #     ) for window in self.windows
        # )
        # @nb.njit(parallel=True,nopython=False)
        # def chunk(windows):
        #     imgs = np.empty((len(windows),img_size))
        #     for i in nb.prange(len(windows)):
        #         imgs[i] = _get_chunk(windows[i], image_t)

        #     return imgs
        # imgs = chunk(self.windows)
        start_t = time.time()
        features = []
        color_type = self.best_params['color_type']
        bins = self.best_params['bins']
        orient = self.best_params['orient']
        block_norm = self.best_params['block_norm']
        transform_sqrt = self.best_params['transform_sqrt']
        cell_per_block = self.best_params['cell_per_block']
        pix_per_cell = self.best_params['pix_per_cell']
        
        image_c = f.ski_color_change[color_type](image_t)
        for window in self.windows:
            img = image_c[window[0][0]:window[1][0], window[0][1]:window[1][1],:]
            img = skit.resize(img, (64,64), anti_aliasing=True)
            start_t = time.time()
            f_histogram_0 = np.histogram(img[:,:,0], bins=bins, range=(0,256))
            f_histogram_1 = np.histogram(img[:,:,1], bins=bins, range=(0,256))
            f_histogram_2 = np.histogram(img[:,:,2], bins=bins, range=(0,256))
            #f_histogram = np.concatenate((f_histogram_0[0],f_histogram_1[0],f_histogram_2[0]))
            end_t = time.time()
            #print(f'Hist time-> {end_t-start_t}')
            start_t = time.time()
            f_hog_0 = hog(
                img[:,:,0],
                orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                block_norm=block_norm,
                transform_sqrt=transform_sqrt,
                feature_vector=True,
                visualize=False,
            )
            f_hog_1 = hog(
                img[:,:,1],
                orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                block_norm=block_norm,
                transform_sqrt=transform_sqrt,
                feature_vector=True,
                visualize=False,
            )
            f_hog_2 = hog(
                img[:,:,2],
                orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                block_norm=block_norm,
                transform_sqrt=transform_sqrt,
                feature_vector=True,
                visualize=False,
            )
            features.append(np.concatenate((f_histogram_0[0], f_histogram_1[0], f_histogram_2[0], f_hog_0, f_hog_1, f_hog_2)))
        X_scaler = pickle.load(open(x_scalar_path, 'rb'))
        scaled_X = X_scaler.transform(features)
        end_t = time.time()
        print(f'Chunk time-> {end_t-start_t}')
        start_t = time.time()
        # params = {
        #     'X': imgs,
        #     **self.best_params
        # }
        #features = f.pipeline(**params)
        end_t = time.time()
        print(f'Features time-> {end_t-start_t}')
        
        start_t = time.time()
        y_pred = self.model.predict(scaled_X)
        end_t = time.time()
        print(f'Model time-> {end_t-start_t}')
        return image_t


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
