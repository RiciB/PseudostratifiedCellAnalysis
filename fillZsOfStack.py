import os
import h5py
import numpy as np
from skimage import io, measure, morphology, segmentation, transform, filters
from scipy import ndimage as ndi
import napari


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
    	closedBackground = morphology.binary_closing(backgroundImg[numZ, :, :] == 0, morphology.disk(10))
    	filledZ = morphology.remove_small_holes(backgroundImg[numZ, :, :] == 0, area_threshold = 15000)
    	segmentedImgFilled[numZ, : , :] = (filledZ & (closedBackground == 0) & (backgroundImg[numZ, :, :] == 1)) * newIndex

    newLabelsImg = morphology.label(segmentedImgFilled == newIndex);

    newLabelsImg = morphology.remove_small_objects(newLabelsImg, min_size=50000)
    with napari.gui_qt():
        print('Waiting on napari')
        viewer = napari.view_image(segmentedImg, rgb=False)
        viewer.add_labels(newLabelsImg, name='newLabelsOnly')
        viewer.add_labels(segmentedImgFilled, name='segmentedImgFilled')

    #print(np.sort(np.unique(newLabelsImg)))
    for newLabel in np.sort(np.unique(newLabelsImg)):
        if newLabel != 0:
            segmentedImgFilled[newLabelsImg == newLabel] = newIndex
            newIndex = newIndex + 1

    segmentedImgFilled[segmentedImgFilled == 0] = segmentedImg[segmentedImgFilled == 0]

    for numBackgroundIds in backgroundIndices:
        segmentedImgFilled[segmentedImgFilled == numBackgroundIds] = 0

    #Divide into two regions: bottom and top
    backgroundLabelled = morphology.label(segmentedImgFilled == 0)

    segmentedImgFilled[backgroundLabelled == 1] = 0
    segmentedImgFilled[backgroundLabelled == 2] = newIndex;

    # with napari.gui_qt():
    #     print('Waiting on napari')
    #     viewer = napari.view_image(segmentedImg, rgb=False)
    #     viewer.add_labels(newLabelsImg, name='newLabelsOnly')
    #     viewer.add_labels(backgroundLabelled, name='backgroundLabelled')
    #     viewer.add_labels(segmentedImgFilled, name='segmentedImgFilled')


    return segmentedImgFilled

raw_folder = 'Data/';

all_files = os.listdir(raw_folder + 'Images/');

txtFileWithZs = raw_folder + 'apical_substacks_to_rerun.txt'

with open(txtFileWithZs, 'r') as file:
	file.readline()
	for line in file:
		lineSplitted = line.split()

		file_name = lineSplitted[0]
		firstZ = int(lineSplitted[1]) - 1
		lastZ = int(lineSplitted[2]) - 1

		print(file_name)

		raw_img = io.imread(raw_folder + 'Images/' + file_name + '.tif');
		seg_file = h5py.File(raw_folder + 'Segmentations/' + file_name + '_predictions_gasp_average.h5', 'r')
		seg_img = seg_file['/segmentation'][()];


		backgroundIndices = np.unique((seg_img[0, 0, 0], seg_img[seg_img.shape[0]-1, seg_img.shape[1]-1, seg_img.shape[2]-1]))

		fillEmptyCells(seg_img, backgroundIndices)

		if firstZ > lastZ: # We are at basal layer
			#Need to exchange first and last
			aux = firstZ
			firstZ = lastZ
			lastZ = aux

		for numZ in range(lastZ, firstZ-1, -1):
			print(numZ)
			# Use previous layer as seeds
			previousLayer = seg_img[numZ+1, :, :];
			previousLayer[previousLayer == backgroundIndices[0]] = 0
			previousLayer[previousLayer == backgroundIndices[1]] = 0

			idCells = np.sort(np.unique(previousLayer));
			idCells = idCells[1:]

			erodedPreviousLayer = np.zeros_like(previousLayer);
			#Erode each cell
			for numCell in idCells:
				erodedPreviousLayer[morphology.binary_erosion(previousLayer == numCell, morphology.disk(8))] = numCell

			denoised = filters.gaussian(raw_img[numZ, :, :], sigma=1)
			#gradient = filters.rank.gradient(denoised, morphology.disk(1))
			watershedImg = segmentation.watershed(denoised, markers = erodedPreviousLayer)
			# with napari.gui_qt():
			# 	print('Waiting on napari')
			# 	viewer = napari.view_image(denoised, rgb=False)
			# 	viewer.add_labels(seg_img[numZ, :, :], name='previousSegmentation')
			# 	viewer.add_labels(erodedPreviousLayer, name='erodedPreviousLayer')
			# 	viewer.add_labels(watershedImg, name='watershed')
			seg_img[numZ, :, :] = watershedImg



		outputFile = raw_folder + 'Segmentations_reviewed/' + file_name + '_watersheded.h5'
		postProcessFile = h5py.File(outputFile, 'w')
		segmentationFile = postProcessFile.create_dataset("segmentation", data=seg_img, dtype='uint16')
		postProcessFile.close()