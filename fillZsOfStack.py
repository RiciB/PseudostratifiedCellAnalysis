import os
import h5py
import numpy as np
from skimage import io, measure, morphology, segmentation, transform, filters
from scipy import ndimage as ndi
import napari

raw_folder = 'Data/';

all_files = os.listdir(raw_folder + 'Images/');

txtFileWithZs = raw_folder + 'apical_substacks_to_rerun.txt'

with open(txtFileWithZs, 'r') as file:
	for line in file:
		lineSplitted = line.split()

		file_name = lineSplitted[0]
		firstZ = lineSplitted[1] - 1
		lastZ = lineSplitted[2] - 1

		raw_img = io.imread(raw_folder + 'Images/' + file_name + '.tif');
		seg_file = h5py.File(raw_folder + 'Segmentations/' + file_name + '_predictions_gasp_average.h5', 'r')
		seg_img = seg_file['/segmentation'][()];

		if firstZ > lastZ: # We are at basal layer
			#Need to exchange first and last
			aux = firstZ
			firstZ = lastZ
			lastZ = aux

		for numZ in range(lastZ, firstZ-1, -1):
			# Use previous layer as seeds
			erodedPreviousLayer = morphology.erosion(seg_img[numZ+1, :, :], morphology.disk(10))
			denoised = morphology.rank.median(raw_img[numZ, :, :], disk(5))
			#gradient = filters.rank.gradient(denoised, morphology.disk(1))
			watershedImg = segmentation.watershed(filters.scharr(denoised), markers = erodedPreviousLayer)
			with napari.gui_qt():
				print('Waiting on napari')
				viewer = napari.view_image(raw_img[numZ, :, :], rgb=False)
				viewer.add_labels(seg_img[numZ, :, :], name='previousSegmentation')
				viewer.add_labels(filters.scharr(raw_img[numZ, :, :]), name='distance')
				viewer.add_labels(watershedImg, name='watershed')