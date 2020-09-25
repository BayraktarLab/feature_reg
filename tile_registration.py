import numpy as np
import cv2 as cv
import tifffile as tif
from skimage.transform import AffineTransform, warp, warp_polar

from slicer import split_image_into_number_of_blocks
from feature_detection import find_features_parallelized, register_pairs_parallelized
from filtering import filter_outliers


def split_image_into_tiles(img):
    ntiles = 4
    x_ntiles = ntiles
    y_ntiles = ntiles

    img_tiles = split_image_into_number_of_blocks(img, x_ntiles, y_ntiles, overlap=0)
    return img_tiles


def register_tiles(ref_img_tiles, mov_img_tiles):
    ref_tiles_features = find_features_parallelized(ref_img_tiles)
    mov_tiles_features = find_features_parallelized(mov_img_tiles)
    registration_results = register_pairs_parallelized(ref_tiles_features, mov_tiles_features, ref_img_tiles, mov_img_tiles)
    return registration_results


def find_best_matched_tiles(reg_results: dict):
    """ returns dictionary of a kind
        {mov_tile_id: {ref_tile_id: {'reg_transform': np.array, 'matches': int, 'good_matches': int}}}
        That contains matched tiles from moving image and reference image.
    """
    best_match_pairs = dict()

    for mov_tile in reg_results:
        ref_tile_ids = list(reg_results[mov_tile].keys())
        reg_info = list(reg_results[mov_tile].values())
        this_tile_good_matches = [i['good_matches'] if i is not None else None for i in reg_info]
        best_match = max([m for m in this_tile_good_matches if m is not None] + [-1])
        if best_match == -1:
            continue
        else:
            best_match_id = this_tile_good_matches.index(best_match)
            best_match_pairs[mov_tile] = {ref_tile_ids[best_match_id]: reg_info[best_match_id]}
    return best_match_pairs


def add_tile_coordinates(best_matches, x_ntiles, y_ntiles, tile_size_x, tile_size_y):
    coord_matrix = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=np.float32)
    best_matches_with_coords = dict()
    for mov_img_id, match_info in best_matches.items():
        ref_img_id = list(match_info.keys())[0]

        mov_img_coord_x = mov_img_id % x_ntiles * tile_size_x
        mov_img_coord_y = mov_img_id // x_ntiles * tile_size_y
        mov_coord_matrix = coord_matrix.copy()
        mov_coord_matrix[:, 2] = (mov_img_coord_x, mov_img_coord_y)

        ref_img_coord_x = ref_img_id % x_ntiles * tile_size_x
        ref_img_coord_y = ref_img_id // x_ntiles * tile_size_y
        ref_coord_matrix = coord_matrix.copy()
        ref_coord_matrix[:, 2] = (ref_img_coord_x, ref_img_coord_y)

        global_tile_translation = coord_matrix.copy()
        global_tile_translation[:, 2] = mov_coord_matrix[:, 2] - ref_coord_matrix[:, 2]
        match_info[ref_img_id].update({'mov_tile_coord': mov_coord_matrix,
                                       'ref_tile_coord': ref_coord_matrix,
                                       'global_tile_translation': global_tile_translation})
        best_matches_with_coords[mov_img_id] = match_info

    return best_matches_with_coords


def convert_transform_matrix_local_to_global(reg_transform, global_tile_translation, mov_tile_coord):
    """ This function uses global translation """
    h_mov_tile_coord = np.append(mov_tile_coord, [[0, 0, 1]], axis=0)
    translate_to_origin = np.linalg.pinv(h_mov_tile_coord)

    h_global_tile_translation = np.append(global_tile_translation, [[0, 0, 1]], axis=0)
    h_inv_global_tile_translation = np.linalg.pinv(h_global_tile_translation)

    h_reg_transform = np.append(reg_transform, [[0, 0, 1]], axis=0)

    global_transform_matrix = h_inv_global_tile_translation @ (h_mov_tile_coord @ h_reg_transform @ translate_to_origin)

    return global_transform_matrix[:-1, :]


def convert_transform_matrices(best_matches):
    global_transform_matrices = []
    for match_info in best_matches.values():
        match_info = list(match_info.values())[0]
        reg_transform = match_info['reg_transform']
        global_tile_translation = match_info['global_tile_translation']
        mov_tile_coord = match_info['mov_tile_coord']

        global_transform_matrix = convert_transform_matrix_local_to_global(reg_transform, global_tile_translation, mov_tile_coord)
        global_transform_matrices.append(global_transform_matrix)
    return global_transform_matrices


def split_into_tiles_and_register(img1, img2):
    img1_tiles, img1_tile_info = split_image_into_tiles(img1)
    img2_tiles, img2_tile_info = split_image_into_tiles(img2)
    
    x_ntiles = img2_tile_info['nblocks']['x']
    y_ntiles = img2_tile_info['nblocks']['y']
    tile_size_y, tile_size_x = img2_tile_info['block_shape']

    result = register_tiles(img1_tiles, img2_tiles)  # will delete tiles inside function
    best_matches = find_best_matched_tiles(result)
    best_matches = add_tile_coordinates(best_matches, x_ntiles, y_ntiles, tile_size_x, tile_size_y)
    #global_tile_translation_matrices = get_transform_matrices(best_matches)

    global_matrices = convert_transform_matrices(best_matches)
    final_transformation_matrix = filter_outliers(global_matrices)

    return final_transformation_matrix
