import copy

import numpy as np
import dask
import cv2 as cv
Image = np.ndarray


def img_to_float(img):
    return cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)


def float_to_img(fimg, original_numpy_dtype):
    cv_np_map = {np.dtype('uint8'): cv.CV_8U,
                 np.dtype('uint16'): cv.CV_16U,
                 np.dtype('int8'): cv.CV_8S,
                 np.dtype('int16'): cv.CV_16S
                 }
    cv_dtype = cv_np_map[original_numpy_dtype]
    alpha = np.iinfo(original_numpy_dtype).min
    beta = np.iinfo(original_numpy_dtype).max
    return cv.normalize(fimg, None, alpha, beta, cv.NORM_MINMAX, cv_dtype)


def diff_of_gaus(img: Image, low_sigma: int = 5, high_sigma: int = 9):
    #TODO replace with difference of kernels
    original_dtype = copy.copy(img.dtype)
    if original_dtype != np.float32:
        fimg = img_to_float(img)
    else:
        fimg = img
    low_sigma = cv.GaussianBlur(fimg, (0, 0), sigmaX=low_sigma, dst=None, sigmaY=low_sigma)
    high_sigma = cv.GaussianBlur(fimg, (0, 0), sigmaX=high_sigma, dst=None, sigmaY=high_sigma)
    diff = low_sigma - high_sigma
    del low_sigma, high_sigma
    if original_dtype == np.float32:
        # do not need to convert back already in float32
        return diff
    else:
        return float_to_img(diff, original_dtype)


def preprocess_image(img: Image):
    # TODO try to use opencv retina module
    processed_img = diff_of_gaus(img, 3, 5)
    if processed_img.dtype != np.uint8:
        processed_img = cv.normalize(processed_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    return processed_img


def find_features(img):

    processed_img = preprocess_image(img)
    if processed_img.max() == 0:
        return [], []
    #detector = cv.MSER_create()
    detector = cv.FastFeatureDetector_create(1, True)
    descriptor = cv.xfeatures2d.DAISY_create()
    kp = detector.detect(processed_img)

    # get 1000 best points based on feature detector response
    if len(kp) <= 3000:
        pass
    else:
        kp = sorted(kp, key=lambda x: x.response, reverse=True)[:3000]

    kp, des = descriptor.compute(processed_img, kp)

    if kp is None or len(kp) < 3:
        return [], []
    if des is None or len(des) < 3:
        return [], []

    return kp, des


def register_pair(img1_kp_des, img2_kp_des):
    kp1, des1 = img1_kp_des
    kp2, des2 = img2_kp_des

    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.knnMatch(des2, des1, k=2)

    print(len(matches))
    # Filter out unreliable points
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    print('good matches', len(good), '/', len(matches))
    if len(good) < 3:
        return None
    # convert keypoints to format acceptable for estimator
    src_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # find out how images shifted (compute affine transformation)
    affine_transform_matrix, mask = cv.estimateAffinePartial2D(dst_pts, src_pts, method=cv.RANSAC, confidence=0.99)
    return {'reg_transform': affine_transform_matrix, 'matches': len(matches), 'good_matches': len(good)}


def find_features_parallelized(tile_list):
    task = []
    for tile in tile_list:
        task.append(dask.delayed(find_features)(tile))
    tiles_features = dask.compute(*task)
    return tiles_features
