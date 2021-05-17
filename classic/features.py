import pandas as pd
import numpy as np
import skimage as ski
import skimage.color as skic
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import cv2
import joblib

ski_color_change = {
    "RGB": lambda img: img,
    'HSV': skic.rgb2hsv,
    'YCbCr': skic.rgb2ycbcr,
    'GREY': skic.rgb2grey,
    'HSL': lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
}


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


def img_feature(img, color_type='HSL', orient=9, pix_per_cell=8, cell_per_block=2,
                    block_norm='L1', transform_sqrt=True, bins=32):
    img_color = ski_color_change[color_type](img)

    f_histogram_0 = np.histogram(img_color[:,:,0], bins=bins, range=(0,256))
    f_histogram_1 = np.histogram(img_color[:,:,1], bins=bins, range=(0,256))
    f_histogram_2 = np.histogram(img_color[:,:,2], bins=bins, range=(0,256))
    f_histogram = np.concatenate((f_histogram_0[0],f_histogram_1[0],f_histogram_2[0]))

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
    
    features =  np.concatenate((f_histogram, f_hog_0, f_hog_1, f_hog_2))
    return features

X_scaler = StandardScaler()
def pipeline(X, color_type='HSL', orient=9, pix_per_cell=8, cell_per_block=2,
                block_norm='L1', transform_sqrt=True, bins=32, train=True): 
    features=joblib.Parallel(n_jobs=-1)(
            joblib.delayed(img_feature)(
                image,
                color_type,
                orient,
                pix_per_cell,
                cell_per_block,
                block_norm,
                transform_sqrt,
                bins                     
            ) for image in X['image'].values
        )
    if train:
        X_scaler = StandardScaler().fit(features)
    scaled_X = X_scaler.transform(features)
    return scaled_X