import numpy as np
import dask
import cv2 as cv
Image = np.ndarray


def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


def find_features(img):
    processed_img = img
    if processed_img.dtype != np.uint8:
        processed_img = cv.normalize(processed_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    if processed_img.max() == 0:
        return [], []

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

    # fix problem with pickle
    temp_kp_storage = []
    for point in kp:
        temp_kp_storage.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id))

    return temp_kp_storage, des


def match_features(img1_kp_des, img2_kp_des):
    kp1, des1 = img1_kp_des
    kp2, des2 = img2_kp_des

    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.knnMatch(des2, des1, k=2)

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
    return affine_transform_matrix


def find_features_parallelized(tile_list):
    task = []
    for tile in tile_list:
        task.append(dask.delayed(find_features)(tile))
    tiles_features = dask.compute(*task)
    return tiles_features
