import numpy as np
import pandas as pd
import cv2 as cv
import tifffile as tif
from tifffile import TiffWriter
from tifffile import TiffFile
from skimage.feature import register_translation
from skimage.transform import warp_polar
import re
import os
import argparse
np.set_printoptions(suppress=True)


def alphaNumOrder(string):
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
 Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
 """
    return ''.join([format(int(x), '05d') if x.isdigit()
                    else x for x in re.split(r'(\d+)', string)])


def calculate_padding_size(bigger_shape, smaller_shape):
    """ Find difference between shapes of bigger and smaller image. """
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


def convert_to_transform_matrix(scale, angle, vector):
    import math
    """Return homogeneous transformation matrix from similarity parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector (of size 2).
    The order of transformations is: scale, rotate, translate.
    """
    transform_matrix = np.zeros((2,3), dtype=np.float32)
    S = np.diag([scale, scale])
    R = np.identity(2)
    angle = math.radians(angle)
    R[0, 0] = math.cos(angle)
    R[0, 1] = math.sin(angle)
    R[1, 0] = -math.sin(angle)
    R[1, 1] = math.cos(angle)
    new_vector = np.dot(vector, np.dot(R, S))
    transform_matrix[:,:2] = R
    transform_matrix[:,2] = new_vector
    return transform_matrix


def pad_to_size(target_shape, img):
    left, right = calculate_padding_size(target_shape[1], img.shape[1])
    top, bottom = calculate_padding_size(target_shape[0], img.shape[0])
    return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0)


def pad_to_size2(target_shape, img):
    left, right = calculate_padding_size(target_shape[1], img.shape[1])
    top, bottom = calculate_padding_size(target_shape[0], img.shape[0])
    return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0), (left, right, top, bottom)


def rescale_translation_coordinates(trans_matrix, scale):
    """ Does rescaling of translation coordinates x and y """
    x_coord = trans_matrix[0][2] / scale
    y_coord = trans_matrix[1][2] / scale
    # new_translation_matrix = np.array([[1.0, 0.0, x_coord], [0.0, 1.0, y_coord]], dtype=np.float32)
    new_translation_matrix = np.array(
        [[trans_matrix[0][0], trans_matrix[0][1], x_coord], [trans_matrix[1][0], trans_matrix[1][1], y_coord]],
        dtype=np.float32)
    return new_translation_matrix


def reg_features(left, right, scale):
    # convert images to uint8 so detector can use them
    img1 = cv.normalize(left, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    img2 = cv.normalize(right, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    img1_new_shape = int(img1.shape[1] * scale), int(img1.shape[0] * scale)
    img2_new_shape = int(img2.shape[1] * scale), int(img2.shape[0] * scale)

    img1 = cv.resize(img1, img1_new_shape, interpolation=cv.INTER_CUBIC)
    img2 = cv.resize(img2, img2_new_shape, interpolation=cv.INTER_CUBIC)

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
    M = rescale_translation_coordinates(A, scale)

    # warp image based on translation matrix
    # dst = cv.warpAffine(right, M, (right.shape[1], right.shape[0]), None)
    return M


def reg_features2(img1, img2):
    """ Image registration using phase shift correlation.
        Angle shift is calculated using polar transformation.
    """
    # register translation
    translation_shift, error, phasediff = register_translation(img1, img2, upsample_factor=20)

    # register rotation
    radius = max(img1.shape)
    img1 = warp_polar(img1, radius=radius, scaling='linear', multichannel=False)
    img2 = warp_polar(img2, radius=radius, scaling='linear', multichannel=False)
    shift, error, phasediff = register_translation(img1, img2, upsample_factor=100)
    rotation_shift = shift[0]
    tansform_matrix = convert_to_transform_matrix(1, rotation_shift, (translation_shift[1], translation_shift[0]))

    return tansform_matrix


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
    print('estimating registration parameters')
    transform_matrices = []
    images = read_images(image_paths, is_dir=False)

    ref_img_id = image_paths.index(ref_image_path)
    max_size_x = max([img.shape[1] for img in images])
    max_size_y = max([img.shape[0] for img in images])
    target_shape = (max_size_y, max_size_x)
    padding = []
    for i in range(0, len(images)):
        images[i], pad = pad_to_size2((max_size_y, max_size_x), images[i])
        padding.append(pad)

    reference_image = images[ref_img_id]

    for i in range(0, len(images)):
        if i == ref_img_id:
            transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            transform_matrices.append(transform_matrix)
        else:
            #transform_matrices.append(reg_features(reference_image, images[i], scale))
            transform_matrices.append(reg_features2(reference_image, images[i]))
    return transform_matrices, target_shape, padding


def generate_new_metadata(image_paths, target_shape):
    time = []
    planes = []
    channels = []
    metadata_list = []
    phys_size_x_list = []
    phys_size_y_list = []

    for i in range(0, len(image_paths)):
        with TiffFile(image_paths[i]) as TF:
            image_axes = list(TF.series[0].axes)
            image_shape = TF.series[0].shape

            if 'T' in image_axes:
                idx = image_axes.index('T')
                time.append(image_shape[idx])
            else:
                time.append(1)
            if 'Z' in image_axes:
                idx = image_axes.index('Z')
                planes.append(image_shape[idx])
            else:
                planes.append(1)
            if 'C' in image_axes:
                idx = image_axes.index('C')
                channels.append(image_shape[idx])
            else:
                channels.append(1)

            ome_meta = TF.ome_metadata.replace('\n', '').replace('\r', '')
            metadata_list.append(ome_meta)
            phys_size_x_list.extend(re.findall('PhysicalSizeX="(.*?)"', ome_meta))
            phys_size_y_list.extend(re.findall('PhysicalSizeY="(.*?)"', ome_meta))

    max_time = max(time)
    max_planes = max(planes)
    total_channels = sum(channels)
    max_phys_size_x = max(phys_size_x_list)
    max_phys_size_y = max(phys_size_y_list)

    sizes = {' SizeX=".*?"': ' SizeX="' + str(target_shape[1]) + '"',
             ' SizeY=".*?"': ' SizeY="' + str(target_shape[0]) + '"',
             ' SizeC=".*?"': ' SizeC="' + str(total_channels) + '"',
             ' SizeZ=".*?"': ' SizeZ="' + str(max_planes) + '"',
             ' SizeT=".*?"': ' SizeT="' + str(max_time) + '"',
             ' PhysicalSizeX=".*?"': ' PhysicalSizeX="' + str(max_phys_size_x) + '"',
             ' PhysicalSizeY=".*?"': ' PhysicalSizeY="' + str(max_phys_size_y) + '"'
             }

    header_limit = metadata_list[0].find('<Channel ID')
    header = metadata_list[0][:header_limit]

    for key, value in sizes.items():
        header = re.sub(key, value, header)

    ncycles = len(image_paths)

    total_channel_meta = ''
    write_format = '0' + str(len(str(ncycles)) + 1) + 'd'  # e.g. for number 5 format = 02d, result = 05
    channel_id = 0
    for i in range(0, ncycles):
        cycle_name = 'Cycle' + format(i+1, write_format) + ' '
        channel_names = re.findall('(?<=<Channel).*?Name="(.*?)"', metadata_list[i])
        channel_ids = re.findall('Channel ID="(.*?)"', metadata_list[i])
        new_channel_names = [cycle_name + ch for ch in channel_names]
        channel_meta = re.findall('<Channel.*?<TiffData', metadata_list[i])[0].replace('<TiffData', '')

        for n in range(0, len(new_channel_names)):
            new_channel_id = 'Channel:0:' + str(channel_id)
            channel_meta = channel_meta.replace(channel_names[n], new_channel_names[n]).replace(channel_ids[n],
                                                                                                new_channel_id)
            channel_id += 1
        total_channel_meta += channel_meta

    plane_meta = ''
    IFD = 0
    for t in range(0, max_time):
        for c in range(0, total_channels):
            for z in range(0, max_planes):
                plane_meta += '<TiffData FirstC="{0}" FirstT="{1}" FirstZ="{2}" IFD="{3}" PlaneCount="1"></TiffData>'.format(
                    c, t, z, IFD)
                IFD += 1

    footer = '</Pixels></Image></OME>'
    result_ome_meta = header + total_channel_meta + plane_meta + footer

    return max_time, max_planes, total_channels, result_ome_meta


def transform_by_plane(input_file_paths, output_path, target_shape, transform_matrices):
    print('transforming images')
    max_time, max_planes, max_channels, new_meta = generate_new_metadata(input_file_paths, target_shape)
    no_transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with TiffWriter(output_path + 'out.tif', bigtiff=True) as TW:

        for i, path in enumerate(input_file_paths):
            transform_matrix = transform_matrices[i]

            with TiffFile(path, is_ome=True) as TF:
                image_axes = list(TF.series[0].axes)
                image_shape = TF.series[0].shape

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
                    ntime = image_shape[idx]
                else:
                    ntime = 1

                page = 0
                for t in range(0, ntime):
                    for z in range(0, nplanes):
                        for c in range(0, nchannels):
                            image = TF.asarray(key=page)

                            if image.shape != target_shape:
                                image = pad_to_size(target_shape, image)
                            if not np.array_equal(transform_matrix, no_transform_matrix):
                                image = cv.warpAffine(image, transform_matrix, (image.shape[1], image.shape[0]), None)

                            TW.save(image, photometric='minisblack', description=new_meta)
                            page += 1
                        if nplanes < max_planes:
                            diff = max_planes - nplanes
                            empty_page = np.zeros_like(image)
                            for a in range(0, diff):
                                TW.save(empty_page, photometric='minisblack', description=new_meta)


def main():

    parser = argparse.ArgumentParser(description='Image registration')

    parser.add_argument('--maxz_images', type=str, nargs='+', required=True,
                        help='specify, separated by space, paths to maxz images of anchor channels\n'
                             ' you want to use for estimating registration parameters.\n'
                             ' They should also include reference image.')
    parser.add_argument('--maxz_ref_image', type=str, required=True,
                        help='specify path to reference maxz image, the one that will be used as reference for aligning all other images.')
    parser.add_argument('--register_images', type=str, nargs='+', default='none',
                        help='specify, separated by space, paths to z-stacked images you want to register.\n'
                             'They should be in the same order as images specified in --maxz_images argument.'
                             'If not specified, --maxz_images will be used.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='directory to output registered image.')
    parser.add_argument('--estimate_only', action='store_true',
                        help='add this flag if you want to get only registration parameters and do not want to process images.')
    parser.add_argument('--load_params', type=str, default='none',
                        help='specify path to csv file with registration parameters')
    parser.add_argument('--scale', type=str, default=0.5,
                        help='scale of the images during registration. Default value is 0.5. '
                             'The lower the value the smaller the scale.')

    args = parser.parse_args()
    maxz_images = args.maxz_images
    maxz_ref_image = args.maxz_ref_image
    imgs_to_register = args.register_images
    out_dir = args.out_dir
    estimate_only = args.estimate_only
    load_params = args.load_params
    scale = float(args.scale)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not out_dir.endswith('/'):
        out_dir = out_dir + '/'

    if load_params == 'none':
        transform_matrices, target_shape, padding = estimate_registration_parameters(maxz_images, maxz_ref_image, scale)
    else:
        reg_param = pd.read_csv(load_params)
        target_shape = (reg_param.loc[0, 'height'], reg_param.loc[0, 'width'])

        transform_matrices = []
        padding = []
        for i in reg_param.index:
            matrix = reg_param.loc[i, ['0', '1', '2', '3', '4', '5']].to_numpy().reshape(2, 3).astype(np.float32)
            pad = reg_param.loc[i, ['left', 'right', 'top', 'bottom']].to_list()
            transform_matrices.append(matrix)
            padding.append(pad)

    if imgs_to_register == 'none':
        imgs_to_register = maxz_images

    if not estimate_only:
        transform_by_plane(imgs_to_register, out_dir, target_shape, transform_matrices)

    transform_matrices_flat = [M.flatten() for M in transform_matrices]
    transform_table = pd.DataFrame(transform_matrices_flat)
    for i in transform_table.index:
        transform_table.loc[i, 'path'] = imgs_to_register[i]
    cols = transform_table.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    transform_table = transform_table[cols]
    for i in range(0, len(padding)):
        transform_table.loc[i, 'left'] = padding[i][0]
        transform_table.loc[i, 'right'] = padding[i][1]
        transform_table.loc[i, 'top'] = padding[i][2]
        transform_table.loc[i, 'bottom'] = padding[i][3]
        transform_table.loc[i, 'width'] = target_shape[1]
        transform_table.loc[i, 'height'] = target_shape[0]
    try:
        transform_table.to_csv(out_dir + 'registration_parameters.csv', index=False)
    except PermissionError:
        transform_table.to_csv(out_dir + 'registration_parameters_1.csv', index=False)


if __name__ == '__main__':
    main()
