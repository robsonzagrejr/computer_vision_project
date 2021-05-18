import pandas as pd
import numpy as np
import skimage as ski
import skimage.color as skic
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import cv2
import joblib
import pickle
import numba as nb
import skimage.transform as skit
import time
import skimage.io as skio
import multiprocessing
import matplotlib.pyplot as plt
from classic.config import (
    x_scalar_path
)

import itertools as it

ski_color_change = {
    "RGB": lambda img: img,
    'HSV': skic.rgb2hsv,
    'YCbCr': skic.rgb2ycbcr,
    'GREY': skic.rgb2grey,
    'HSL': lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
}
import classic.test as t


def get_hog_fatures(img, orient, pix_per_cell, cell_per_block, channel=0,
                    block_norm='L1', transform_sqrt=True, visualize=False):
     return hog(
        img[:,:,channel],
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        block_norm=block_norm,
        transform_sqrt=transform_sqrt,
        feature_vector=True,
        visualize=visualize,
    )


#def img_feature(img, color_type='HSL', orient=9, pix_per_cell=8, cell_per_block=2,
#                    block_norm='L1', transform_sqrt=True, bins=32):
def img_feature(term):
    img, color_type, orient, pix_per_cell, cell_per_block,block_norm, transform_sqrt, bins = term

    img = skit.resize(img, (64,64), anti_aliasing=True)
    end_t = time.time()
    #print(f'Color time-> {end_t-start_t}')
    img_color = ski_color_change[color_type](img)
    start_t = time.time()
    f_histogram_0 = np.histogram(img_color[:,:,0], bins=bins, range=(0,256))
    f_histogram_1 = np.histogram(img_color[:,:,1], bins=bins, range=(0,256))
    f_histogram_2 = np.histogram(img_color[:,:,2], bins=bins, range=(0,256))
    f_histogram = np.concatenate((f_histogram_0[0],f_histogram_1[0],f_histogram_2[0]))
    end_t = time.time()
    #print(f'Hist time-> {end_t-start_t}')
    start_t = time.time()
    f_hog_0 = get_hog_fatures(img_color, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,channel=0,
                            block_norm=block_norm,
                            transform_sqrt=transform_sqrt, visualize=False)
    f_hog_1 = get_hog_fatures(img_color, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,channel=1,
                            block_norm=block_norm,
                            transform_sqrt=transform_sqrt, visualize=False)
    f_hog_2 = get_hog_fatures(img_color, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,channel=2,
                            block_norm=block_norm,
                            transform_sqrt=transform_sqrt, visualize=False)
    end_t = time.time()
    #print(f'HOG time-> {end_t-start_t}')
    features =  np.concatenate((f_histogram, f_hog_0, f_hog_1, f_hog_2))
    return features


def pipeline(X, color_type='HSL', orient=9, pix_per_cell=8, cell_per_block=2,
                block_norm='L1', transform_sqrt=True, bins=32, train=True):
    # features = []
    # for image in X:
    #     features.append(img_feature(
    #             image,
    #             color_type,
    #             orient,
    #             pix_per_cell,
    #             cell_per_block,
    #             block_norm,
    #             transform_sqrt,
    #             bins                     
    #         ))
    features = []
    for image in X:
        img = skit.resize(image, (64,64), anti_aliasing=True)
        img_color = ski_color_change[color_type](img)
        start_t = time.time()
        f_histogram_0 = np.histogram(img_color[:,:,0], bins=bins, range=(0,256))
        f_histogram_1 = np.histogram(img_color[:,:,1], bins=bins, range=(0,256))
        f_histogram_2 = np.histogram(img_color[:,:,2], bins=bins, range=(0,256))
        #f_histogram = np.concatenate((f_histogram_0[0],f_histogram_1[0],f_histogram_2[0]))
        end_t = time.time()
        #print(f'Hist time-> {end_t-start_t}')
        start_t = time.time()
        f_hog_0 = get_hog_fatures(img_color, orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,channel=0,
                                block_norm=block_norm,
                                transform_sqrt=transform_sqrt, visualize=False)
        f_hog_1 = get_hog_fatures(img_color, orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,channel=1,
                                block_norm=block_norm,
                                transform_sqrt=transform_sqrt, visualize=False)
        f_hog_2 = get_hog_fatures(img_color, orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,channel=2,
                                block_norm=block_norm,
                                transform_sqrt=transform_sqrt, visualize=False)
        features.append(np.concatenate((f_histogram_0[0], f_histogram_1[0], f_histogram_2[0], f_hog_0, f_hog_1, f_hog_2)))
    if train:
        X_scaler = StandardScaler().fit(features)
        pickle.dump(X_scaler,  open(x_scalar_path, 'wb'))
    else:
        X_scaler = pickle.load(open(x_scalar_path, 'rb'))

    scaled_X = X_scaler.transform(features)
    return scaled_X

# target='gpu',
#@nb.vectorize("[float32]([[float32],[float32],[float32]], uint8)",nopython=True)
def gpu_hist_features(img, bins):
    f_histogram_0 = np.histogram(img[:,:,0], bins=bins, range=(0,256))[0]
    f_histogram_1 = np.histogram(img[:,:,1], bins=bins, range=(0,256))[0]
    f_histogram_2 = np.histogram(img[:,:,2], bins=bins, range=(0,256))[0]
    f_histogram = np.concatenate((f_histogram_0,f_histogram_1,f_histogram_2))
    return f_histogram


def fast_gpu_img_hh_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                    block_norm='L1', transform_sqrt=True, bins=32):
    img = skit.resize(img, (64,64), anti_aliasing=True)
    f_histogram = gpu_hist_features(img, bins)
    f_hog_0 = get_hog_fatures(img, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,channel=0,
                            block_norm=block_norm,
                            transform_sqrt=transform_sqrt, visualize=False)
    f_hog_1 = get_hog_fatures(img, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,channel=1,
                            block_norm=block_norm,
                            transform_sqrt=transform_sqrt, visualize=False)
    f_hog_2 = get_hog_fatures(img, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,channel=2,
                            block_norm=block_norm,
                            transform_sqrt=transform_sqrt, visualize=False)
    features =  np.concatenate((f_histogram, f_hog_0, f_hog_1, f_hog_2))
    return features


@nb.njit(parallel=True)
def fast_gpu_pipeline(X, orient=9, pix_per_cell=8, cell_per_block=2,
                    block_norm='L1', transform_sqrt=True, bins=32, color_type=None):

    features = np.empty((len(X),4164))
    for i in nb.prange(len(X)):
        features[i] = fast_gpu_img_hh_features(
            X[i],
            orient,
            pix_per_cell,
            cell_per_block,
            block_norm,
            transform_sqrt,
            bins                     
        )

    X_scaler = pickle.load(open(x_scalar_path, 'rb'))

    scaled_X = X_scaler.transform(features)

    return scaled_X