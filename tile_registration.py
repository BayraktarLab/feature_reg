import numpy as np
import dask

from slicer import split_image_into_number_of_blocks
from feature_detection import find_features, register_pair


def split_image_into_tiles(img, x_ntiles, y_ntiles):
    img_tiles = split_image_into_number_of_blocks(img, x_ntiles, y_ntiles, overlap=0)
    return img_tiles


def find_features_parallelized(tile_list):
    task = []
    for tile in tile_list:
        task.append(dask.delayed(find_features)(tile))
    tiles_features = dask.compute(*task)
    return tiles_features


def register_pairs_parallelized(ref_tiles_features, mov_tiles_features):
    """ Run feature matching algorithm for reference and moving tiles """
    results = {i: {} for i in range(0, len(mov_tiles_features))}

    # if pair is valid add its information to the dask tasks else delete key
    task = []
    task_id = 0
    for m, mov_tile in enumerate(mov_tiles_features):
        if mov_tile != ([], []):
            delayed_mov_tile = dask.delayed(mov_tile)
            for r, ref_tile in enumerate(ref_tiles_features):
                if ref_tile != ([], []):
                    results[m].update({r: task_id})
                    task.append(dask.delayed(register_pair)(ref_tile, delayed_mov_tile))
                    task_id += 1
        else:
            del results[m]

    if task != []:
        matching_results = dask.compute(*task)
    else:
        raise ValueError('No tile matches were found')

    # find which pairs had been processed by checking if they have task id
    for mov_tile, ref_tiles in results.items():
        for ref_tile in list(ref_tiles):
            this_ref_tile_task_id = ref_tiles[ref_tile]
            results[mov_tile][ref_tile] = matching_results[this_ref_tile_task_id]

    return results


def register_tiles(ref_img_tiles, mov_img_tiles):
    ref_tiles_features = find_features_parallelized(ref_img_tiles)
    mov_tiles_features = find_features_parallelized(mov_img_tiles)
    del ref_img_tiles, mov_img_tiles
    registration_results = register_pairs_parallelized(ref_tiles_features, mov_tiles_features)
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


def zscore_filter_stack(arr):
    z_scores = np.abs((arr - np.mean(arr, axis=0)) / np.std(arr, axis=0))
    mask = z_scores < 1
    inliers = np.argwhere(np.all(mask, axis=1) == True)
    inlier_ids = [i[0] for i in inliers.tolist()]
    if len(inlier_ids) > 0:
        result = np.median(arr[inlier_ids, :], axis=0)
    else:
        result = np.array([])
    return result


def iqr_filter_stack(arr):
    q1 = np.quantile(arr, 0.25, axis=0)
    q3 = np.quantile(arr, 0.75, axis=0)

    mask = np.bitwise_and(arr >= q1,  arr <= q3)
    inliers = np.argwhere(np.all(mask, axis=1) == True)
    inlier_ids = [i[0] for i in inliers.tolist()]
    if len(inlier_ids) > 0:
        result = np.median(arr[inlier_ids, :], axis=0)
    else:
       result = np.array([])
    return result


def filter_outliers(matrix_list):
    matrices_raveled = [m.ravel() for m in matrix_list]
    matrix_stack = np.stack(matrices_raveled)

    iqr_res = iqr_filter_stack(matrix_stack)
    zscore_res = zscore_filter_stack(matrix_stack)
    print('IQR', iqr_res)
    print('z-score', zscore_res)
    if iqr_res.size != 0 and zscore_res.size != 0:
        filtered_transformation_matrix = np.mean([iqr_res, zscore_res], axis=0).reshape((2, 3))
    elif iqr_res.size == 0 and zscore_res.size != 0:
        filtered_transformation_matrix = zscore_res.reshape((2, 3))
    elif iqr_res.size != 0 and zscore_res.size == 0:
        filtered_transformation_matrix = iqr_res.reshape((2, 3))
    else:
        raise ValueError('No proper alignment was found')
    print(filtered_transformation_matrix)
    return filtered_transformation_matrix


def split_into_tiles_and_register(img1, img2):
    ntiles = max(max(img1.shape) // 1000 // 2, 2)
    x_ntiles = ntiles
    y_ntiles = ntiles

    img1_tiles, info_img1 = split_image_into_tiles(img1, x_ntiles, y_ntiles)
    img2_tiles, info_img2 = split_image_into_tiles(img2, x_ntiles, y_ntiles)
    del img2
    print(info_img2)

    x_ntiles = info_img2['nblocks']['x']
    y_ntiles = info_img2['nblocks']['y']
    tile_size_y, tile_size_x = info_img2['block_shape']

    result = register_tiles(img1_tiles, img2_tiles)  # will delete tiles inside function
    best_matches = find_best_matched_tiles(result)
    best_matches = add_tile_coordinates(best_matches, x_ntiles, y_ntiles, tile_size_x, tile_size_y)
    #global_tile_translation_matrices = get_transform_matrices(best_matches)

    global_matrices = convert_transform_matrices(best_matches)
    final_transformation_matrix = filter_outliers(global_matrices)

    return final_transformation_matrix

