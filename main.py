import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, segmentation, transform, filters

chosen_cells = {}

raw_folder = '/Users/ricardo/Documents/Temporary/Segmentations/';

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
    seg_data = seg_file['/segmentation'][()]


    try:
        good_seg_data = np.zeros_like(seg_data);
    except Exception as e:
        seg_data = np.array(segmentedImgh5.get('segmentation'));
        good_seg_data = np.zeros_like(seg_data);
    

    for cell in chosen_cells:
        for num_cell in chosen_cells[cell]:
            print(num_cell)
            good_seg_data[seg_data == num_cell] = num_cell[0]


    propertyTable = ('area', 'bbox', 'bbox_area', 'centroid', 'convex_area', 'convex_image', 'coords', 'eccentricity')

    intensityProps = ('label', 'mean_intensity', 'weighted_centroid')

    shapeProps = ('label', 'area', 'convex_area', 'equivalent_diameter', 
                  'extent', 'feret_diameter_max', 'filled_area', 'major_axis_length', 
                  'minor_axis_length', 'solidity')

    #'eccentricity', 'perimeter' and 'perimeter_crofton' is not implemented for 3D images

    props = measure.regionprops_table(good_seg_data, intensity_image=raw_img, properties=propertyTable)

    np.savetxt(raw_folder + stack_ID + '_cell-analysis.csv', props, delimiter=", ", fmt="% s")