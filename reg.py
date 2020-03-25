import re
import os
import gc
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import cv2 as cv
import tifffile as tif
from tifffile import TiffWriter
from tifffile import TiffFile


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


def reg_features(reference_img, moving_img, scale):
    """ Perform feature based image registration """
    # convert images to uint8 so detector can use them
    if reference_img.dtype != np.uint8:
        img1 = cv.normalize(reference_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    else:
        img1 = reference_img
    if moving_img.dtype != np.uint8:
        img2 = cv.normalize(moving_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    else:
        img2 = moving_img

    # create feature detector and keypoint descriptors
    detector = cv.FastFeatureDetector_create()
    descriptor = cv.xfeatures2d.DAISY_create()
    kp1 = detector.detect(img1)
    kp1, des1 = descriptor.compute(img1, kp1)
    del img1
    gc.collect()
    kp2 = detector.detect(img2)
    kp2, des2 = descriptor.compute(img2, kp2)
    del img2
    gc.collect()
    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.knnMatch(des2, des1, k=2)

    # Filter out unreliable points
    good = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good.append(m)

    if good == []:
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)

    assert good != [], 'Not enough good features to calculate image alignment'

    # convert keypoints to format acceptable for estimator
    src_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # find out how images shifted (compute affine transformation)
    A, mask = cv.estimateAffinePartial2D(dst_pts, src_pts)
    M = rescale_translation_coordinates(A, scale)

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


def find_ref_ch(ome_meta, ref_channel):
    # find selected reference channels in ome metadata
    if ome_meta is None:
        print('No OME XML detected. Using first channel')
        return 0
    matches = re.findall(r'Fluor=".*?"', ome_meta)
    if matches != []:
        matches = [m.replace('Fluor=', '').replace('"', '') for m in matches]
        matches = [re.sub(r'c\d+\s+', '', m) for m in matches]  # remove cycle name
    else:
        matches = re.findall(r'ID="Channel:.*?" Name=".*?"', ome_meta)
        matches = [re.sub(r'ID="Channel:.*?" Name="[c\d\s]*', '', m) for m in matches]
        matches = [m.replace('"', '') for m in matches]
    # encode reference channels as 1 other 0

    for i, channel in enumerate(matches):
        if channel == ref_channel:
            ref_channel_id = i
    # check if reference channel is available
    if ref_channel not in matches:
        raise ValueError('Incorrect reference channel. Available reference channels ' + ', '.join(set(matches)))

    return ref_channel_id


def read_and_max_project(path, ref_channel):
    """ Iteratively load z planes to create max intensity projection"""
    with TiffFile(path) as TF:
        img_axes = list(TF.series[0].axes)
        img_shape = TF.series[0].shape
        ome_meta = TF.ome_metadata
    if len(img_shape) == 2 and ome_meta is None:
        ref_channel_id = 0
    else:
        ref_channel_id = find_ref_ch(ome_meta, ref_channel)

    if 'Z' in img_axes:
        idx = img_axes.index('Z')
        nzplanes = img_shape[idx]
    else:
        nzplanes = 1
    if 'C' in img_axes:
        idx = img_axes.index('C')
        nchannels = img_shape[idx]
    else:
        nchannels = 1

    start_reading_from = ref_channel_id * nzplanes
    end_reading_at = start_reading_from + nzplanes

    if nzplanes == 1:
        # handle maxz projection where 1 z plane per channel
        return cv.normalize(tif.imread(path, key=start_reading_from), None, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
    else:
        # initialize with first z plane
        max_intensity = cv.normalize(tif.imread(path, key=start_reading_from), None, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
        for i in range(start_reading_from + 1, end_reading_at + 1):
            max_intensity = np.maximum(max_intensity, tif.imread(path, key=i))

    return max_intensity


def estimate_registration_parameters(img_paths, ref_img_id, ref_channel, scale):
    print('estimating registration parameters')
    nimgs = len(img_paths)
    padding = []
    transform_matrices = []
    img_shapes = []

    for i in range(0, len(img_paths)):
        with TiffFile(img_paths[i]) as TF:
            img_shapes.append(TF.series[0].shape[-2:])

    ref_img_path = img_paths[ref_img_id]
    max_size_x = max([s[1] for s in img_shapes])
    max_size_y = max([s[0] for s in img_shapes])
    target_shape = (max_size_y, max_size_x)
    target_shape_resized = int(target_shape[1] * scale), int(target_shape[0] * scale)

    reference_img = read_and_max_project(ref_img_path, ref_channel)
    reference_img, pad = pad_to_size2(target_shape, reference_img)
    padding.append(pad)
    reference_img = cv.resize(reference_img, target_shape_resized, interpolation=cv.INTER_CUBIC)

    gc.collect()

    for i in range(0, nimgs):
        print('image {0}/{1}'.format(i + 1, nimgs))
        if i == ref_img_id:
            transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            transform_matrices.append(transform_matrix)
        else:
            moving_img, pad = pad_to_size2(target_shape, read_and_max_project(img_paths[i], ref_channel))
            padding.append(pad)
            moving_img = cv.resize(moving_img, target_shape_resized, interpolation=cv.INTER_CUBIC)

            transform_matrices.append(reg_features(reference_img, moving_img, scale))
        gc.collect()
    return transform_matrices, target_shape, padding


def generate_new_metadata(img_paths, target_shape):
    time = []
    planes = []
    channels = []
    metadata_list = []
    phys_size_x_list = []
    phys_size_y_list = []

    for i in range(0, len(img_paths)):
        with TiffFile(img_paths[i]) as TF:
            img_axes = list(TF.series[0].axes)
            img_shape = TF.series[0].shape

            if 'T' in img_axes:
                idx = img_axes.index('T')
                time.append(img_shape[idx])
            else:
                time.append(1)
            if 'Z' in img_axes:
                idx = img_axes.index('Z')
                planes.append(img_shape[idx])
            else:
                planes.append(1)
            if 'C' in img_axes:
                idx = img_axes.index('C')
                channels.append(img_shape[idx])
            else:
                channels.append(1)

            ome_meta = TF.ome_metadata.replace('\n', '').replace('\r', '')
            metadata_list.append(ome_meta)
            phys_size_x_list.extend(re.findall(r'PhysicalSizeX="(.*?)"', ome_meta))
            phys_size_y_list.extend(re.findall(r'PhysicalSizeY="(.*?)"', ome_meta))

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

    ncycles = len(img_paths)

    total_channel_meta = ''
    write_format = '0' + str(len(str(ncycles)) + 1) + 'd'  # e.g. for number 5 format = 02d, result = 05
    channel_id = 0
    for i in range(0, ncycles):
        cycle_name = 'c' + format(i+1, write_format) + ' '
        channel_names = re.findall(r'(?<=<Channel).*?Name="(.*?)"', metadata_list[i])
        channel_ids = re.findall(r'Channel ID="(.*?)"', metadata_list[i])
        new_channel_names = [cycle_name + ch for ch in channel_names]
        channel_meta = re.findall(r'<Channel.*?<TiffData', metadata_list[i])[0].replace('<TiffData', '')

        for n in range(0, len(new_channel_names)):
            new_channel_id = 'Channel:0:' + str(channel_id)
            channel_meta = channel_meta.replace(channel_names[n], new_channel_names[n]).replace(channel_ids[n], new_channel_id)
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
    # try allows to work with images without omexml. Need to remove later
    try:
        max_time, max_planes, max_channels, new_meta = generate_new_metadata(input_file_paths, target_shape)
    except AttributeError:
        max_time, max_planes, max_channels = (1,1,1)
        new_meta = ''
    no_transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    with TiffWriter(output_path + 'out.tif', bigtiff=True) as TW:

        for i, path in enumerate(input_file_paths):
            print('image {0}/{1}'.format(i + 1, len(input_file_paths)))

            transform_matrix = transform_matrices[i]

            with TiffFile(path, is_ome=True) as TF:
                img_axes = list(TF.series[0].axes)
                img_shape = TF.series[0].shape

                if 'C' in img_axes:
                    idx = img_axes.index('C')
                    nchannels = img_shape[idx]
                else:
                    nchannels = 1
                if 'Z' in img_axes:
                    idx = img_axes.index('Z')
                    nplanes = img_shape[idx]
                else:
                    nplanes = 1
                if 'T' in img_axes:
                    idx = img_axes.index('T')
                    ntime = img_shape[idx]
                else:
                    ntime = 1

                page = 0
                for t in range(0, ntime):
                    for z in range(0, nplanes):
                        for c in range(0, nchannels):
                            img = TF.asarray(key=page)

                            if img.shape != target_shape:
                                img = pad_to_size(target_shape, img)
                            if not np.array_equal(transform_matrix, no_transform_matrix):
                                img = cv.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]), None)

                            TW.save(img, photometric='minisblack', description=new_meta)
                            page += 1

                        if nplanes < max_planes:
                            diff = max_planes - nplanes
                            empty_page = np.zeros_like(img)
                            for a in range(0, diff):
                                TW.save(empty_page, photometric='minisblack', description=new_meta)


def main():

    parser = argparse.ArgumentParser(description='Image registration')

    parser.add_argument('-i', type=str, nargs='+', required=True,
                        help='paths to images you want to register separated by space.')
    parser.add_argument('-r', type=int, required=True,
                        help='reference image id, e.g. if -i 1.tif 2.tif 3.tif, and you ref image is 1.tif, then -r 0 (starting from 0)')
    parser.add_argument('-c', type=str, required=True,
                        help='reference channel name, e.g. DAPI. Enclose in double quotes if name consist of several words e.g. "Atto 490LS".')
    parser.add_argument('-o', type=str, required=True,
                        help='directory to output registered image.')
    parser.add_argument('-s', type=float, default=0.5,
                        help='scale of the images during registration in fractions of 1. 1-full scale, 0.5 - half scale. '
                             'Default value is 0.5.')
    parser.add_argument('--estimate_only', action='store_true',
                        help='add this flag if you want to get only registration parameters and do not want to process images.')
    parser.add_argument('--load_param', type=str, default='none',
                        help='specify path to csv file with registration parameters')

    args = parser.parse_args()

    img_paths = args.i
    ref_img_id = args.r
    ref_channel = args.c
    out_dir = args.o
    scale = args.s
    estimate_only = args.estimate_only
    load_param = args.load_param

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not out_dir.endswith('/'):
        out_dir = out_dir + '/'

    st = datetime.now()
    print('\nstarted', st)

    if load_param == 'none':
        transform_matrices, target_shape, padding = estimate_registration_parameters(img_paths, ref_img_id, ref_channel, scale)
    else:
        reg_param = pd.read_csv(load_param)
        target_shape = (reg_param.loc[0, 'height'], reg_param.loc[0, 'width'])

        transform_matrices = []
        padding = []
        for i in reg_param.index:
            matrix = reg_param.loc[i, ['0', '1', '2', '3', '4', '5']].to_numpy().reshape(2, 3).astype(np.float32)
            pad = reg_param.loc[i, ['left', 'right', 'top', 'bottom']].to_list()
            transform_matrices.append(matrix)
            padding.append(pad)

    if not estimate_only:
        transform_by_plane(img_paths, out_dir, target_shape, transform_matrices)

    transform_matrices_flat = [M.flatten() for M in transform_matrices]
    transform_table = pd.DataFrame(transform_matrices_flat)
    for i in transform_table.index:
        dataset_name = 'dataset_{0}_{1}'.format(i + 1, re.sub('\..*', '', os.path.basename(img_paths[i])) )
        transform_table.loc[i, 'name'] = dataset_name
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

    fin = datetime.now()
    print('\nelapsed time', fin - st)


if __name__ == '__main__':
    main()
