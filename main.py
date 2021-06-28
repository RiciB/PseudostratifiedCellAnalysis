import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import diplib as dip
import pandas as pd

from PIL import Image
from PIL.TiffTags import TAGS
from operator import xor
from skimage import io, measure, morphology, segmentation, transform, filters

import napari


def ismember(indices, matrix):
    return [ np.sum(index == matrix) for index in indices ]

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
    
    print('Calculating neighbours...')

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

def fillEmptyCells(segmentedImg, backgroundIndices):
    print('Filling empty cells...')
    backgroundImg = np.zeros_like(segmentedImg);
    segmentedImgFilled = np.zeros_like(segmentedImg)
    for numBackgroundIds in backgroundIndices:
        backgroundImg[segmentedImg == numBackgroundIds] = 1
    
    #3D option
    #morphology.binary_dilation(backgroundImg == 0, ball(2));

    #2D Alternative
    newIndex = np.max(segmentedImg) + 1;
    for numZ in range(0, segmentedImg.shape[0]):
        segmentedImgFilled[numZ, : , :] = morphology.remove_small_holes(backgroundImg[numZ, :, :]==0, area_threshold = 15000) * newIndex;

    return segmentedImgFilled

def surfacesArea(segmentedImg, backgroundIndices, chosen_cells):
    print('Calculating surface areas...')
    if chosen_cells == []:
        cellIds = np.sort(np.unique(segmentedImg))
        cellIds = cellIds[1:]
    else:
        cellIds = chosen_cells;

    ## In case we need more, we can store each image
    #lateralSurface = np.zeros_like(segmentedImg);
    #topSurface = np.zeros_like(segmentedImg);
    #bottomSurface = np.zeros_like(segmentedImg);

    lateralSurfaceAreas = np.zeros(shape=cellIds.shape[0])
    topSurfaceAreas = np.zeros(shape=cellIds.shape[0])
    bottomSurfaceAreas = np.zeros(shape=cellIds.shape[0])
    cellHeightZs = np.zeros(shape=cellIds.shape[0])

    num_cell = 0;

    for cel in cellIds:
        #print(cel)
        if cel in backgroundIndices or cel == 0:
            continue
        
        cellHeightZs[num_cell] = np.sum((segmentedImg == cel).any(2).any(1));

        #Boundary of cell and boundary of cell and background
        boundaryCell = segmentation.find_boundaries(segmentedImg==cel)

        if backgroundIndices.shape[0] == 2:
            boundaryCellAndTop = segmentation.find_boundaries((segmentedImg==cel) | (segmentedImg == backgroundIndices[0]))
            boundaryCellAndBottom = segmentation.find_boundaries((segmentedImg==cel) | (segmentedImg == backgroundIndices[1]))

            lateralSurfaceAreas[num_cell] = np.sum(boundaryCellAndTop & boundaryCellAndBottom & boundaryCell)
            topSurfaceAreas[num_cell] = np.sum(~boundaryCellAndTop & boundaryCell)
            bottomSurfaceAreas[num_cell] = np.sum(~boundaryCellAndBottom & boundaryCell)

            #lateralSurface[boundaryCellAndTop & boundaryCellAndBottom & boundaryCell] = cel;
            #topSurface[~boundaryCellAndTop & boundaryCell] = cel;
            #bottomSurface[~boundaryCellAndBottom & boundaryCell] = cel;
        else:
            boundaryCellAndBackground = segmentation.find_boundaries(ismember((cell, backgroundIndices), segmentedImg))
            lateralSurface[boundaryCellAndBackground & boundaryCell] = cel;
            topAndBottomSurface[~boundaryCellAndBackground & boundaryCell] = cel;

            #Do something here like splitting the two possible regions
            #topSurface = topAndBottomSurface;
            #bottomSurface = topAndBottomSurface;
            print('CAREEEEEEE!!')

        num_cell = num_cell + 1;
    #Export average zs of basal and apical layer
    return pd.DataFrame({'lateralSurfaceArea': lateralSurfaceAreas[0:num_cell]}), pd.DataFrame({'topSurfaceArea': topSurfaceAreas[0:num_cell]}), pd.DataFrame({'bottomSurfaceArea': bottomSurfaceAreas[0:num_cell]}), pd.DataFrame({'cellHeightZs' : cellHeightZs[0:num_cell]});

chosen_cells = {}

raw_folder = 'Data/';

all_files = os.listdir(raw_folder + 'Images/');

for file_name in all_files:
        
    stack_ID = file_name.rstrip('.tif');
    print(stack_ID)
    chosen_cells = set()

    file_path = raw_folder + 'Cells/' + stack_ID + '_GOOD-CELLS.txt'

    if not os.path.exists(file_path):
        print('%s does not exists, skipping...', file_path)
        continue


    with open(file_path, 'r') as file:
        for cell in file:
            cell = (int(cell),)
            chosen_cells.add(cell)

    file_path = raw_folder + 'Cells/' + stack_ID + '_SPLIT-CELLS.txt'

    if not os.path.exists(file_path):
        print('%s does not exists, skipping...', file_path)
        continue

    with open(file_path, 'r') as file:
        for cells in file:
            cells = tuple([int(cell) for cell in cells.split()])
            chosen_cells.add(cells)


    raw_img = io.imread(raw_folder + 'Images/' + file_name);

    if file_name.endswith('tif') or file_name.endswith('tiff'):
        zSpacing, xResolution, yResolution = raw_parameters(raw_folder + 'Images/' + file_name)
        raw_img = transform.resize(raw_img, (round(raw_img.shape[0]*(zSpacing/xResolution)), raw_img.shape[1], raw_img.shape[2]),
                           order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)

    seg_file = h5py.File(raw_folder + 'Segmentations/' + stack_ID + '_predictions_gasp_average.h5', 'r')
    seg_data = seg_file['/segmentation'][()];

    backgroundIndices = np.unique((seg_data[0, 0, 0], seg_data[seg_data.shape[0]-1, seg_data.shape[1]-1, seg_data.shape[2]-1]))
    # Splitting the 'resize' fuctions for the RAM to recover
    if file_name.endswith('tif') or file_name.endswith('tiff'):
        seg_data = transform.resize(seg_data, (round(seg_data.shape[0]*(zSpacing/xResolution)), seg_data.shape[1], seg_data.shape[2]),
                           order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
    
    #seg_neighbours = neighbours(seg_data, []);
    segmentedImg_filledHoles = fillEmptyCells(seg_data, backgroundIndices);
    # with napari.gui_qt():
    #     print('Waiting on napari')
    #     viewer = napari.view_image(seg_data, rgb=False)
    #     viewer.add_labels(segmentedImg_filledHoles, name='removedHoles')
    #     #viewer.add_labels(good_seg_data, name='selectedCells')


    try:
        good_seg_data = np.zeros_like(seg_data);
    except Exception as e:
        seg_data = np.array(seg_file.get('segmentation'));
        good_seg_data = np.zeros_like(seg_data);

    for cell in chosen_cells:
        #print(cell)
        for num_cell in cell:
            #print(num_cell)
            good_seg_data[seg_data == num_cell] = cell[0];

    seg_data[good_seg_data>0] = good_seg_data[good_seg_data>0];

    # with napari.gui_qt():
    #     print('Waiting on napari')
    #     viewer = napari.view_image(seg_data, rgb=False)
    #     viewer.add_labels(segmentedImg_filledHoles, name='removedHoles')
    #     viewer.add_labels(good_seg_data, name='selectedCells')

 
    lateralArea, top_area, bottom_area, cell_height_Zs = surfacesArea(seg_data, backgroundIndices, np.unique(good_seg_data))

    #propertyTable = ('area', 'bbox', 'bbox_area', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity')
    #'eccentricity', 'perimeter' and 'perimeter_crofton' is not implemented for 3D images

    shapeProps = ('label', 'area', 'major_axis_length', 'mean_intensity')

    print('Calculating cell features')
    props = measure.regionprops_table(good_seg_data, intensity_image=raw_img, properties=shapeProps)

    props['volume'] = np.array(props['area'] * (xResolution * xResolution * xResolution), dtype=float);
    props['cellHeight'] = np.array(props['major_axis_length'] * (xResolution), dtype=float);
    #props['minor_axis_length'] = np.array(props['minor_axis_length'] * (xResolution), dtype=float);

    featureCells = pd.DataFrame(props)
    
    featureCells['lateral_area'] = np.array(lateralArea['lateralSurfaceArea'] * (xResolution * xResolution), dtype=float)
    featureCells['top_area'] = np.array(top_area['topSurfaceArea'] * (xResolution * xResolution), dtype=float)
    featureCells['bottom_area'] = np.array(bottom_area['bottomSurfaceArea'] * (xResolution * xResolution), dtype=float);
    featureCells['cellHeightZs'] = np.array(cell_height_Zs['cellHeightZs'] * (xResolution), dtype=float);

    featureCells.to_csv(raw_folder + stack_ID + '_cell-analysis.csv', index=False, header=True, float_format="%.8f")