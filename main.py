#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, segmentation, transform, filters


raw_folder = '';

for file_name in os.listdir(raw_folder):
        
    stack_ID = file_name.rstrip('.tif');

    chosen_cells[stack_ID] = set()

    with open(raw_folder + file_name + '_GOOD-CELLS.txt', 'r') as file:
        
        for cell in file:
            
            cell = (int(cell),)
            
            chosen_cells[stack_ID].add(cell)
 
    
    with open(raw_folder + file_name + '_SPLIT-CELLS.txt', 'r') as file:
        
        for cells in file:
            
            cells = tuple([int(cell) for cell in cells.split()])
            
            chosen_cells[stack_ID].add(cells)



    raw_img = io.imread(raw_folder + file_name);

    seg_file = h5py.File(raw_folder + file_name, 'r')
    seg_data = seg_file['/segmentation'][()]


    try:
        good_seg_data = np.zeros_like(seg_data);
    except Exception as e:
        seg_data = np.array(segmentedImgh5.get('segmentation'));
        good_seg_data = np.zeros_like(seg_data);
    

    for cell in chosen_cells:
        for num_cell in cell:
            good_seg_data[seg_data == num_cell] = num_cell[0]


    propertyTable = ('area', 'bbox', 'bbox_area', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity')

    intensityProps = ('label', 'mean_intensity', 'weighted_centroid')

    shapeProps = ('label', 'area', 'convex_area', 'equivalent_diameter', 
                  'extent', 'feret_diameter_max', 'filled_area', 'major_axis_length', 
                  'minor_axis_length', 'solidity')

    #'eccentricity', 'perimeter' and 'perimeter_crofton' is not implemented for 3D images

    props = measure.regionprops_table(good_seg_data, intensity_image=raw_img, properties=propertyTable)

    np.savetxt(stack_ID + '_cell-analysis.csv', props, delimiter=", ", fmt="% s")

