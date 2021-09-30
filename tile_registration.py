import gc
import numpy as np
import cv2 as cv

from slicer import split_image_into_tiles_of_size
from feature_detection import find_features_parallelized, match_features


def split_image_into_tiles(img, tile_size: int):
    x_size = tile_size
    y_size = tile_size
    img_tiles = split_image_into_tiles_of_size(img, x_size, y_size, overlap=51)
    return img_tiles


def combine_features(features, x_ntiles, y_ntiles, tile_size_x, tile_size_y):
    keypoints_combined = []
    descriptors_list = []
    for tile_id, feature in enumerate(features):
        if feature != ([], []):
            kp_list, des_arr = feature
            descriptors_list.append(des_arr)
            for i, kp in enumerate(kp_list):
                tile_coord_x = tile_id % x_ntiles * tile_size_x
                tile_coord_y = tile_id // x_ntiles * tile_size_y
                new_coords = (tile_coord_x + kp[0][0], tile_coord_y + kp[0][1])
                new_kp = cv.KeyPoint(x=new_coords[0], y=new_coords[1], size=kp[1],
                                     angle=kp[2], response=kp[3], octave=kp[4], class_id=kp[5])
                keypoints_combined.append(new_kp)

    descriptors_combined = np.concatenate(descriptors_list, axis=0)
    return keypoints_combined, descriptors_combined


def get_features(img, tile_size):
    img_tiles, img_tile_info = split_image_into_tiles(img, tile_size)

    x_ntiles = img_tile_info['ntiles']['x']
    y_ntiles = img_tile_info['ntiles']['y']
    tile_size_y, tile_size_x = img_tile_info['tile_shape']

    tiles_features = find_features_parallelized(img_tiles)
    del img_tiles
    combined_features = combine_features(tiles_features, x_ntiles, y_ntiles, tile_size_x, tile_size_y)
    del tiles_features
    gc.collect()
    return combined_features


def register_img_pair(ref_combined_features, mov_combined_features):
    transform_matrix = match_features(ref_combined_features, mov_combined_features)
    return transform_matrix
