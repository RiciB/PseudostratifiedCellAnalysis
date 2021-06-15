import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import diplib as dip
from PIL import Image
from PIL.TiffTags import TAGS
import pandas as pd

from skimage import io, measure, morphology, segmentation, transform, filters


def raw_parameters(rawFilePath):
    '''Obtain raw image parameters: zSpacing, xResolution and yResolution from TIFF'''
    
    rawImg = tiff.TiffFile(rawFilePath);
    try:
        zSpacing = rawImg.imagej_metadata['spacing'];
    except Exception as e:
        zSpacing = 1;
        
    rawImg = Image.open(rawFilePath)
    
    if TAGS[282] == 'XResolution':
        xResolution = 1/rawImg.tag_v2[282];
        
    if TAGS[283] == 'YResolution':
        yResolution = 1/rawImg.tag_v2[283];
        
    return zSpacing, xResolution, yResolution

def neighbours(segmentedImg, threshold_height_cells):
    '''Return array of pairs of neighbouring cells from list of thresholded cells in whole image'''
    
    #If cells aren't thresholded
    if threshold_height_cells == []:
        cellIds = np.sort(np.unique(segmentedImg))
        cellIds = cellIds[1:]
    else:
        cellIds = threshold_height_cells

    
    neighbours=np.empty((0,2))
    
    for cel in cellIds:
        BW = segmentation.find_boundaries(segmentedImg==cel)
        BW_dilated = morphology.binary_dilation(BW)
        neighs = np.unique(segmentedImg[BW_dilated==1])
        indices = np.where(neighs==0.0)
        indices = np.append(indices, np.where(neighs==cel))
        neighs = np.delete(neighs, indices)
        for n in neighs:
            neighbours = np.append(neighbours, [(cel, n)], axis=0).astype(np.uint16)
            
    return neighbours

def surfacesArea(segmentedImg, neighbours):
    print('surfaces')
    #dilation of boundaries to obtain the lateral surfaces

    #Remove that regions from the surface area and we got two different surfaces: apical and basal


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

    if file_name.endswith('tif') or file_name.endswith('tiff'):
        zSpacing, xResolution, yResolution = raw_parameters(raw_folder + 'Images/' + file_name)
        raw_img = transform.resize(raw_img, (round(raw_img.shape[0]*(zSpacing/xResolution)), raw_img.shape[1], raw_img.shape[2]),
                           order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)

    seg_file = h5py.File(raw_folder + 'Segmentations/' + stack_ID + '_predictions_gasp_average.h5', 'r')
    seg_data = seg_file['/segmentation'][()];

    try:
        good_seg_data = np.zeros_like(seg_data);
    except Exception as e:
        seg_data = np.array(seg_file.get('segmentation'));
        good_seg_data = np.zeros_like(seg_data);


    for cell in chosen_cells:
        for num_cell in chosen_cells[cell]:
            print(num_cell)
            good_seg_data[seg_data == num_cell] = num_cell[0]

    # Splitting the 'resize' fuctions for the RAM to recover
    if file_name.endswith('tif') or file_name.endswith('tiff'):
        good_seg_data = transform.resize(good_seg_data, (round(good_seg_data.shape[0]*(zSpacing/xResolution)), good_seg_data.shape[1], good_seg_data.shape[2]),
                           order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)

    #propertyTable = ('area', 'bbox', 'bbox_area', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity')
    #'eccentricity', 'perimeter' and 'perimeter_crofton' is not implemented for 3D images

    shapeProps = ('label', 'area', 'major_axis_length', 'minor_axis_length', 'mean_intensity', 'weighted_centroid')

    
    print('Calculating cell features')
    props = measure.regionprops_table(good_seg_data, intensity_image=raw_img, properties=shapeProps)

    props['area'] = props['area'] * (xResolution * xResolution * xResolution);
    props['major_axis_length'] = props['major_axis_length'] * (xResolution * xResolution);
    props['minor_axis_length'] = props['minor_axis_length'] * (xResolution * xResolution);

    
    np.savetxt(raw_folder + stack_ID + '_cell-analysis.csv', pd.DataFrame(props), delimiter=", ", fmt="% s")