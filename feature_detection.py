import numpy as np
import dask
import cv2 as cv
Image = np.ndarray


def diff_of_gaus(img: Image, low_sigma: int = 5, high_sigma: int = 9):
    if img.max() == 0:
        return img
    else:
        fimg = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        kernel = (low_sigma * 4 * 2 + 1, low_sigma * 4 * 2 + 1)  # as in opencv
        ls = cv.GaussianBlur(fimg, kernel, sigmaX=low_sigma, dst=None, sigmaY=low_sigma)
        hs = cv.GaussianBlur(fimg, kernel, sigmaX=high_sigma, dst=None, sigmaY=high_sigma)
        dog = hs - ls
        del hs, ls
        return cv.normalize(dog, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def store_kp(kp):
    # fix problem with pickle
    temp_kp_storage = []
    for point in kp:
        temp_kp_storage.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id))
    return temp_kp_storage


def view_tile_without_overlap(img, overlap):
    return img[overlap:-overlap, overlap:-overlap]


def find_features(img, nfeatures_limit=5000):
    processed_img = diff_of_gaus(img)

    if processed_img.max() == 0:
        return [], []
    # default values except for threshold - discard points that have 0 response
    detector = cv.FastFeatureDetector_create(threshold=1, nonmaxSuppression=True,
                                             type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
    # default values
    descriptor = cv.xfeatures2d.DAISY_create(radius=21, q_radius=3, q_theta=8, q_hist=8,
                                             norm=cv.xfeatures2d.DAISY_NRM_NONE,
                                             interpolation=True, use_orientation=False)
    overlap = 51
    kp = detector.detect(view_tile_without_overlap(processed_img, overlap))

    kp = sorted(kp, key=lambda x: x.response, reverse=True)[:nfeatures_limit]

    kp, des = descriptor.compute(processed_img, kp)
    
    if kp is None or len(kp) < 3:
        return [], []
    if des is None or len(des) < 3:
        return [], []

    stored_kp = store_kp(kp)

    return stored_kp, des


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
    src_pts = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32).reshape((-1, 1, 2))
    dst_pts = np.array([kp2[m.queryIdx].pt for m in good], dtype=np.float32).reshape((-1, 1, 2))

    # find out how images shifted (compute affine transformation)
    affine_transform_matrix, mask = cv.estimateAffinePartial2D(dst_pts, src_pts, method=cv.RANSAC, confidence=0.99)
    return affine_transform_matrix


def find_features_parallelized(tile_list):
    n_tiles = len(tile_list)
    nfeatures_limit_per_tile = min(1000000 // n_tiles, 5000)
    task = []
    for tile in tile_list:
        task.append(dask.delayed(find_features)(tile, nfeatures_limit_per_tile))
    tiles_features = dask.compute(*task)
    return tiles_features
