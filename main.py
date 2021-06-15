import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import diplib as dip
from PIL import Image
from PIL.TiffTags import TAGS

from skimage import io, measure, morphology, segmentation, transform, filters


def raw_parameters(rawFilePath):
    '''Obtain raw image parameters: zSpacing, xResolution and yResolution from TIFF'''
    
    rawImg = tiff.TiffFile(rawFilePath);
    try:
        zSpacing = rawImg.imagej_metadata['spacing'];
    except Exception as e:
        zSpacing = 1;
    
    if rawImg.imagej_metadata['unit'] == 'micron':
        measurementsInMicrons = True;
        
    rawImg = Image.open(rawFilePath)
    
    if TAGS[282] == 'XResolution':
        xResolution = 1/rawImg.tag_v2[282];
        
    if TAGS[283] == 'YResolution':
        yResolution = 1/rawImg.tag_v2[283];
        
    if measurementsInMicrons:
        #To nanometers
        zSpacing=zSpacing*1000;
        xResolution=xResolution*1000;
        yResolution=yResolution*1000;
        
    return zSpacing, xResolution, yResolution

chosen_cells = {}

raw_folder = 'Data/';

for file_name in os.listdir(raw_folder + 'Images/'):
        
    stack_ID = file_name.rstrip('.tif');
    print(stack_ID)
    chosen_cells[stack_ID] = set()

    file_path = raw_folder + 'Cells/' + stack_ID + '_GOOD-CELLS.txt'

    if not os.path.exists(file_path):
        print('%s does not exists, skipping...', file_path)
        continue


    with open(file_path, 'r') as file:

        for cell in file:

            cell = (int(cell),)

            chosen_cells[stack_ID].add(cell)

    file_path = raw_folder + 'Cells/' + stack_ID + '_SPLIT-CELLS.txt'

    if not os.path.exists(file_path):
        print('%s does not exists, skipping...', file_path)
        continue

    with open(file_path, 'r') as file:

        for cells in file:

            cells = tuple([int(cell) for cell in cells.split()])

            chosen_cells[stack_ID].add(cells)


    raw_img = io.imread(raw_folder + 'Images/' + file_name);

    seg_file = h5py.File(raw_folder + 'Segmentations/' + stack_ID + '_predictions_gasp_average.h5', 'r')
    seg_data = seg_file['/segmentation'][()];

    if file_name.endswith('tif') or file_name.endswith('tiff'):
        zSpacing, xResolution, yResolution = raw_parameters(raw_folder + 'Images/' + file_name)
        print(round(raw_img.shape[0]*(zSpacing/xResolution)))
        # raw_img = transform.resize(raw_img, (round(raw_img.shape[0]*(zSpacing/xResolution)), raw_img.shape[1], raw_img.shape[2]),
        #                    order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
        # seg_data = transform.resize(seg_data, (round(raw_img.shape[0]*(zSpacing/xResolution)), raw_img.shape[1], raw_img.shape[2]),
        #                    order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)

    try:
        good_seg_data = np.zeros_like(seg_data);
    except Exception as e:
        seg_data = np.array(seg_file.get('segmentation'));
        good_seg_data = np.zeros_like(seg_data);


    for cell in chosen_cells:
        for num_cell in chosen_cells[cell]:
            print(num_cell)
            good_seg_data[seg_data == num_cell] = num_cell[0]

    #propertyTable = ('area', 'bbox', 'bbox_area', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity')

    shapeProps = ('label', 'area', 'convex_area', 'equivalent_diameter', 
                  'extent', 'feret_diameter_max', 'filled_area', 'major_axis_length', 
                  'minor_axis_length', 'solidity', 'mean_intensity', 'weighted_centroid')

    #'eccentricity', 'perimeter' and 'perimeter_crofton' is not implemented for 3D images

    props = measure.regionprops_table(good_seg_data, intensity_image=raw_img, properties=[shapeProps, intensityProps])


    np.savetxt(raw_folder + stack_ID + '_cell-analysis.csv', props, delimiter=", ", fmt="% s")