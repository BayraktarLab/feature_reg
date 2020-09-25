import numpy as np


def filter_outliers(matrix_list):
    matrices_raveled = [m.ravel() for m in matrix_list]
    matrix_stack = np.stack(matrices_raveled)

    filtering_result = iqr_filter_stack(matrix_stack)

    if filtering_result.size != 0:
        filtered_transformation_matrix = filtering_result.reshape((2, 3))
        print(filtered_transformation_matrix)
    else:
        raise ValueError('Failed to find valid transformation matrix')

    return filtered_transformation_matrix


def mad_filter_stack(arr):
    """ Median absolute deviation filtering """
    mad = np.median(np.abs(arr - np.median(arr, axis=0)), axis=0)
    scores = np.abs((arr - np.median(arr, axis=0)) / mad)
    mask = scores <= 1
    return get_inliers(arr, mask)


def iqr_filter_stack(arr):
    q1 = np.quantile(arr, 0.25, axis=0)
    q3 = np.quantile(arr, 0.75, axis=0)

    mask = np.bitwise_and(arr >= q1,  arr <= q3)
    return get_inliers(arr, mask)


def get_inliers(arr, mask):
    """ Return only those matrices that have all values inside it passed filtering """
    inlier_ids = []
    for i in range(0, mask.shape[0]):
        if np.all(mask[i, :]):
            inlier_ids.append(i)
    if len(inlier_ids) > 0:
        result = np.median(arr[inlier_ids, :], axis=0)
    else:
        result = np.array([])
    return result
