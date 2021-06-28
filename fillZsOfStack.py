import os
import h5py
import numpy as np

raw_folder = 'Data/';

all_files = os.listdir(raw_folder + 'Images/');

txtFileWithZs = raw_folder + 'apical_substacks_to_rerun.txt'

with open(txtFileWithZs, 'r') as file:
	for line in file:
		lineSplitted = line.split()

		file_name = lineSplitted[0]
		firstZ = lineSplitted[1]
		lastZ = lineSplitted[2]

		raw_img = io.imread(raw_folder + 'Images/' + file_name + '.tif');
		seg_file = h5py.File(raw_folder + 'Segmentations/' + file_name + '_predictions_gasp_average.h5', 'r')
		seg_img = seg_file['/segmentation'][()];