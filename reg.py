import numpy as np
import cv2 as cv
import tifffile as tif
from tifffile import TiffWriter
from tifffile import TiffFile
import re
import os

np.set_printoptions(suppress=True)


def alphaNumOrder(string):
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
   Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
   """
    return ''.join([format(int(x), '05d') if x.isdigit()
                    else x for x in re.split(r'(\d+)', string)])


def calculate_padding_size(bigger_shape, smaller_shape):
    """ find difference between shapes of bigger and smaller image"""
    diff = bigger_shape - smaller_shape

    if diff == 1:
        dim1 = 1
        dim2 = 0
    elif diff % 2 != 0:
        dim1 = int(diff // 2)
        dim2 = int((diff // 2) + 1)
    else:
        dim1 = dim2 = int(diff / 2)

    return dim1, dim2


def pad_to_size(target_shape, img):
    left, right = calculate_padding_size(target_shape[1], img.shape[1])
    top, bottom = calculate_padding_size(target_shape[0], img.shape[0])
    return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0)


def rescale_translation_coordinates(trans_matrix, scale):
    """ Does rescaling of translation coordinates x and y """
    x_coord = trans_matrix[0][2] / scale
    y_coord = trans_matrix[1][2] / scale
    # new_translation_matrix = np.array([[1.0, 0.0, x_coord], [0.0, 1.0, y_coord]], dtype=np.float32)
    new_translation_matrix = np.array(
        [[trans_matrix[0][0], trans_matrix[0][1], x_coord], [trans_matrix[1][0], trans_matrix[1][1], y_coord]],
        dtype=np.float32)
    return new_translation_matrix


def reg_feratures(left, right, scale):
    # convert images to uint8, so detector can use them
    img1_8b = cv.normalize(left, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    img2_8b = cv.normalize(right, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    img1_decr_shape = int(img1_8b.shape[1] * scale), int(img1_8b.shape[0] * scale)
    img2_decr_shape = int(img2_8b.shape[1] * scale), int(img2_8b.shape[0] * scale)

    img1 = cv.resize(img1_8b, img1_decr_shape, interpolation=cv.INTER_CUBIC)
    img2 = cv.resize(img2_8b, img2_decr_shape, interpolation=cv.INTER_CUBIC)

    # create feature detector and keypoint descriptors

    detector = cv.FastFeatureDetector_create()
    descriptor = cv.xfeatures2d.DAISY_create()
    kp1 = detector.detect(img1)
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2 = detector.detect(img2)
    kp2, des2 = descriptor.compute(img2, kp2)
    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.knnMatch(des2, des1, k=2)

    # Filter out unreliable points
    good = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good.append(m)

    # convert keypoints to format acceptable for estimator
    src_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # find out how images shifted (compute affine transformation)
    A, mask = cv.estimateAffinePartial2D(dst_pts, src_pts)
    M = rescale_translation_coordinates(A, 0.5)

    # warp image based on translation matrix
    # dst = cv.warpAffine(right, M, (right.shape[1], right.shape[0]), None)
    return M


def read_images(path, is_dir):
    """ Read images in natural order (with respect to numbers) """
    allowed_extensions = ('tif', 'tiff')
    if is_dir == True:
        file_list = [fn for fn in os.listdir(path) if fn.endswith(allowed_extensions)]
        file_list.sort(key=alphaNumOrder)
        img_list = list(map(tif.imread, [path + fn for fn in file_list]))
    else:
        if type(path) == list:
            img_list = list(map(tif.imread, path))
        else:
            img_list = tif.imread(path)
    return img_list


def estimate_registration_parameters(image_paths, ref_image_path, scale):
    transform_matrices = []
    images = read_images(image_paths, is_dir=False)

    ref_img_id = image_paths.index(ref_image_path)
    max_size_x = max([img.shape[1] for img in images])
    max_size_y = max([img.shape[0] for img in images])
    target_shape = (max_size_y, max_size_x)

    for i in range(0, len(images)):
        images[i] = pad_to_size((max_size_y, max_size_x), images[i])

    reference_image = images[ref_img_id]

    for i in range(0, len(images)):
        if i == ref_img_id:
            transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            transform_matrices.append(transform_matrix)
        else:
            transform_matrices.append(reg_feratures(reference_image, images[i], scale))
    return reference_image, transform_matrices, target_shape


def transform_by_plane(input_file_paths, output_path, target_shape, transform_matrices, append, image_id, new_meta, max_planes):
    TW = TiffWriter(output_path + 'out.tif', bigtiff=True, append=append)

    for i, path in enumerate(input_file_paths):
        transform_matrix = transform_matrices[i]
        with TiffFile(path, is_ome=True) as TF:
            npages = len(TF.pages)
            image_axes = (TF.series[0].axes)
            image_shape = TF.series[0].shape
            plane_shape = TF.pages[0].shape

            if 'C' in image_axes:
                idx = image_axes.index('C')
                nchannels = image_shape[idx]
            else:
                nchannels = 1
            if 'Z' in image_axes:
                idx = image_axes.index('Z')
                nplanes = image_shape[idx]
            else:
                nplanes = 1
            if 'T' in image_axes:
                idx = image_axes.index('T')
                nplanes = image_shape[idx]
            else:
                ntimes = 1

            ome_meta = TF.ome_metadata

            ome_meta = re.sub('SizeX="(\d+)"', 'SizeX="' + str(target_shape[-1]) + '"', ome_meta)
            ome_meta = re.sub('SizeY="(\d+)"', 'SizeY="' + str(target_shape[-2]) + '"', ome_meta)
            ome_meta.replace('Image ID="Image:0"', 'Image ID="Image:' + str(image_id) + '"')

            if plane_shape == target_shape:
                    page = 0
                    for t in range(0, ntimes):
                        for z in range(0, nplanes):
                            for c in range(0, nchannels):
                                image = TF.asarray(key=page)
                                TW.save(image, photometric='minisblack', description=new_meta)
                                page += 1
                            if nplanes < max_planes:
                                diff = max_planes - nplanes
                                empty_page = np.zeros_like(image)
                                for a in range(0, diff):
                                    TW.save(empty_page, photometric='minisblack', description=new_meta)
            else:
                page = 0
                for t in range(0, ntimes):
                    for z in range(0, nplanes):
                        for c in range(0, nchannels):
                            image = TF.asarray(key=page)
                            image = pad_to_size(target_shape, image)
                            image = cv.warpAffine(image, transform_matrix, (image.shape[1], image.shape[0]), None)
                            TW.save(image, photometric='minisblack', description=new_meta)
                            page += 1
                        if nplanes < max_planes:
                            diff = max_planes - nplanes
                            empty_page = np.zeros_like(image)
                            for a in range(0, diff):
                                TW.save(empty_page, photometric='minisblack', description=new_meta)
    TW.close()


def get_image_data(image_paths):
    for i in range(0, len(image_paths)):
        with TiffFile(image_paths[i]) as TF:
            image_axes = list(TF.series[0].axes)
            image_shape = TF.series[0].shape
            ome_meta = TF.ome_metadata
            len_axes = len(image_axes)

            if 'C' in image_axes:
                idx = image_axes.index('C')
                nchannels = image_shape[idx]
            else:
                nchannels = 1
            if 'Z' in image_axes:
                idx = image_axes.index('Z')
                nplanes = image_shape[idx]
            else:
                nplanes = 1
            if 'T' in image_axes:
                idx = image_axes.index('T')
                nplanes = image_shape[idx]
            else:
                ntimes = 1


input_dir = '/home/ubuntu/test/1_2/'
image_path_list = os.listdir(input_dir)
output_path = '/home/ubuntu/test/modified/'
with open('/home/ubuntu/Desktop/new_meta.txt','r',encoding='utf-8') as f:
    new_meta = f.read()

with open('/home/ubuntu/Desktop/new_meta_max.txt','r',encoding='utf-8') as f:
    new_meta_max = f.read()

transform_by_plane(['/home/ubuntu/test/1_1/maxz_HiPlexCtxLayers_1.tif', '/home/ubuntu/test/2_1/maxz_HiPlexCtxLayers_1.tif'],
                    output_path, reference_image, transformation_matrices, False, 0, new_meta_max, 1)




def main():

    maxz_input = '/home/ubuntu/test/previews/'

    imgs = read_images(maxz_input, is_dir=True)
    reference_image, transform_matrices, target_shape = estimate_registration_parameters(maxz_input, imgs, 0.5)
    result = register_images(imgs, transformation_matrices)
    save_registered_images(result, image_info, 'stack', output_path)


if __name__ == '__main__':
    main()
