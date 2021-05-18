import numpy as np
import skimage.color as skic
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import cv2
import pickle
import skimage.transform as skit

import skimage as ski

from classic.config import (
    x_scalar_path
)


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


def pipeline(X, color_type='HSL', orient=9, pix_per_cell=8, cell_per_block=2,
                block_norm='L1', transform_sqrt=True, bins=32, train=True):
    features = []
    for image in X:
        # Resize to 64x64
        #img = skit.resize(image, (64,64), anti_aliasing=True)
        img = ski.img_as_float32(image)
        # Change color
        img_color = ski_color_change[color_type](img)

        # Get Histogram
        f_histogram_0 = np.histogram(img_color[:,:,0], bins=bins, range=(0,256))
        f_histogram_1 = np.histogram(img_color[:,:,1], bins=bins, range=(0,256))
        f_histogram_2 = np.histogram(img_color[:,:,2], bins=bins, range=(0,256))

        # Get Hog
        f_hog_0 = hog(
            img_color[:,:,0],
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            block_norm=block_norm,
            transform_sqrt=transform_sqrt,
            feature_vector=True,
            visualize=False,
        )
        f_hog_1 = hog(
            img_color[:,:,1],
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            block_norm=block_norm,
            transform_sqrt=transform_sqrt,
            feature_vector=True,
            visualize=False,
        )
        f_hog_2 = hog(
            img_color[:,:,2],
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            block_norm=block_norm,
            transform_sqrt=transform_sqrt,
            feature_vector=True,
            visualize=False,
        )
        # Concatenate
        features.append(np.concatenate((f_histogram_0[0], f_histogram_1[0], f_histogram_2[0], f_hog_0, f_hog_1, f_hog_2)))

    if train:
        X_scaler = StandardScaler().fit(features)
        pickle.dump(X_scaler,  open(x_scalar_path, 'wb'))
    else:
        X_scaler = pickle.load(open(x_scalar_path, 'rb'))

    scaled_X = X_scaler.transform(features)
    return scaled_X
