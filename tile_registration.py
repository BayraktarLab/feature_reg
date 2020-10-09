import numpy as np
import cv2 as cv
import tifffile as tif
from skimage.transform import AffineTransform, warp, warp_polar

from slicer import split_image_into_number_of_blocks
from feature_detection import find_features_parallelized, register_pair


def split_image_into_tiles(img):
    ntiles = 4
    x_ntiles = ntiles
    y_ntiles = ntiles

    img_tiles = split_image_into_number_of_blocks(img, x_ntiles, y_ntiles, overlap=0)
    return img_tiles


def combine_features(features, x_ntiles, y_ntiles, tile_size_x, tile_size_y):
    keypoints_combined = []
    descriptors_list = []
    for tile_id, feature in enumerate(features):
        print(tile_id)
        if feature != ([], []):
            kp_list, des_arr = feature
            descriptors_list.append(des_arr)
            for i, kp in enumerate(kp_list):
                tile_coord_x = tile_id % x_ntiles * tile_size_x
                tile_coord_y = tile_id // x_ntiles * tile_size_y
                kp.pt = (tile_coord_x + kp.pt[0], tile_coord_y + kp.pt[1])
                keypoints_combined.append(kp)

    descriptors_combined = np.concatenate(descriptors_list, axis=0)
    return keypoints_combined, descriptors_combined


def register_tiles(ref_img_tiles, mov_img_tiles, img_tile_info):
    ref_tiles_features = find_features_parallelized(ref_img_tiles)
    mov_tiles_features = find_features_parallelized(mov_img_tiles)

    x_ntiles = img_tile_info['nblocks']['x']
    y_ntiles = img_tile_info['nblocks']['y']
    tile_size_y, tile_size_x = img_tile_info['block_shape']

    ref_combined_features = combine_features(ref_tiles_features, x_ntiles, y_ntiles, tile_size_x, tile_size_y)
    mov_combined_features = combine_features(mov_tiles_features, x_ntiles, y_ntiles, tile_size_x, tile_size_y)
    registration_results = register_pair(ref_combined_features, mov_combined_features)
    return registration_results


def split_into_tiles_and_register(img1, img2):
    img1_tiles, img1_tile_info = split_image_into_tiles(img1)
    img2_tiles, img2_tile_info = split_image_into_tiles(img2)

    result = register_tiles(img1_tiles, img2_tiles, img2_tile_info)  # will delete tiles inside function
    estimated_transformation = result['reg_transform']
    return estimated_transformation
