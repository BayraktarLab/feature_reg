import xml.etree.ElementTree as ET
from io import StringIO
import re

from tifffile import TiffFile


def str_to_xml(xmlstr: str):
    """ Converts str to xml and strips namespaces """
    it = ET.iterparse(StringIO(xmlstr))
    for _, el in it:
        _, _, el.tag = el.tag.rpartition('}')
    root = it.root
    return root


def extract_pixels_info(xml):
    dims = ['SizeX', 'SizeY', 'SizeC', 'SizeZ', 'SizeT']
    sizes = ['PhysicalSizeX', 'PhysicalSizeY']
    pixels = xml.find('Image').find('Pixels')
    pixels_info = dict()
    for d in dims:
        pixels_info[d] = int(pixels.get(d, default=1))
    for s in sizes:
        pixels_info[s] = float(pixels.get(s, default=1))
    return pixels_info


def extract_channel_info(xml):
    channels = xml.find('Image').find('Pixels').findall('Channel')
    channel_names = [ch.get('Name') for ch in channels]
    channel_ids = [ch.get('ID') for ch in channels]
    channel_fluors = []
    for ch in channels:
        if 'Fluor' in ch.attrib:
            channel_fluors.append(ch.get('Fluor'))
    return channels, channel_names, channel_ids, channel_fluors


def get_dimension_size(img_axes, img_shape):
    dims = {'T': 1, 'Z': 1, 'C': 1}
    for d in dims:
        if d in img_axes:
            idx = img_axes.index(d)
            dim_size = img_shape[idx]
            dims[d] = dim_size

    return dims


def generate_new_metadata(img_paths, target_shape):
    ncycles = len(img_paths)
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
            ome_meta = TF.ome_metadata
            metadata_list.append(ome_meta)

    for meta in metadata_list:
        pixels_info = extract_pixels_info(str_to_xml(meta))
        time.append(pixels_info['SizeT'])
        planes.append(pixels_info['SizeZ'])
        channels.append(pixels_info['SizeC'])
        phys_size_x_list.append(pixels_info['PhysicalSizeX'])
        phys_size_y_list.append(pixels_info['PhysicalSizeY'])

    max_time = max(time)
    max_planes = max(planes)
    total_channels = sum(channels)
    max_phys_size_x = max(phys_size_x_list)
    max_phys_size_y = max(phys_size_y_list)

    sizes = {'SizeX': str(target_shape[1]),
             'SizeY': str(target_shape[0]),
             'SizeC': str(total_channels),
             'SizeZ': str(max_planes),
             'SizeT': str(max_time),
             'PhysicalSizeX': str(max_phys_size_x),
             'PhysicalSizeY': str(max_phys_size_y)
             }

    # use metadata from first image as reference metadata
    ref_xml = str_to_xml(metadata_list[0])

    # set proper ome attributes tags
    proper_ome_attribs = {'xmlns': 'http://www.openmicroscopy.org/Schemas/OME/2016-06',
                          'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                          'xsi:schemaLocation': 'http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd'
                          }
    ref_xml.attrib.clear()

    for attr, val in proper_ome_attribs.items():
        ref_xml.set(attr, val)

    # set new dimension sizes
    for attr, size in sizes.items():
        ref_xml.find('Image').find('Pixels').set(attr, size)


    # remove old channels and tiffdata
    old_channels = ref_xml.find('Image').find('Pixels').findall('Channel')
    for ch in old_channels:
        ref_xml.find('Image').find('Pixels').remove(ch)

    tiffdata = ref_xml.find('Image').find('Pixels').findall('TiffData')
    if tiffdata is not None or tiffdata != []:
        for td in tiffdata:
            ref_xml.find('Image').find('Pixels').remove(td)


    # add new channels
    write_format = '0' + str(len(str(ncycles)) + 1) + 'd'  # e.g. for number 5 format = 02d, result = 05
    channel_id = 0
    for i in range(0, ncycles):
        channels, channel_names, channel_ids, channel_fluors = extract_channel_info(str_to_xml(metadata_list[i]))
        cycle_name = 'c' + format(i+1, write_format) + ' '
        new_channel_names = [cycle_name + ch for ch in channel_names]

        for ch in range(0, len(channels)):
            new_channel_id = 'Channel:0:' + str(channel_id)
            new_channel_name = new_channel_names[ch]
            channels[ch].set('Name', new_channel_name)
            channels[ch].set('ID', new_channel_id)
            ref_xml.find('Image').find('Pixels').append(channels[ch])
            channel_id += 1


    # add new tiffdata
    ifd = 0
    for t in range(0, max_time):
        for c in range(0, total_channels):
            for z in range(0, max_planes):
                ET.SubElement(ref_xml.find('Image').find('Pixels'), "TiffData", dict(FirstC=str(c), FirstT=str(t), FirstZ=str(z), IFD=str(ifd), PlaneCount=str(1)))
                ifd += 1

    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
    result_ome_meta = xml_declaration + ET.tostring(ref_xml, method='xml', encoding='utf-8').decode('ascii', errors='ignore')

    return max_time, max_planes, total_channels, result_ome_meta


def find_ref_channel(ome_meta, ref_channel):
    # find selected reference channels in ome metadata
    if ome_meta is None:
        print('No OME XML detected. Using first channel')
        return 0

    channels, channel_names, channel_ids, channel_fluors = extract_channel_info(str_to_xml(ome_meta))

    # strip cycle id from channel name and fluor name
    if channel_fluors != []:
        fluors = [re.sub(r'^(c|cyc|cycle)\d+(\s+|_)', '', fluor) for fluor in channel_fluors]  # remove cycle name
    else:
        fluors = None
    names = [re.sub(r'^(c|cyc|cycle)\d+(\s+|_)', '', name) for name in channel_names]

    # check if reference channel is present somewhere
    if ref_channel in names:
        matches = names
    elif fluors is not None and ref_channel in fluors:
        matches = fluors
    else:
        if fluors is not None:
            message = 'Incorrect reference channel. Available channel names: {names}, fluors: {fluors}'
            raise ValueError(message.format(names=', '.join(set(names)), fluors=', '.join(set(fluors))))
        else:
            message = 'Incorrect reference channel. Available channel names: {names}'
            raise ValueError(message.format(names=', '.join(set(names))))

    # get position of reference channel in cycle
    for i, channel in enumerate(matches):
        if channel == ref_channel:
            ref_channel_id = i

    return ref_channel_id
