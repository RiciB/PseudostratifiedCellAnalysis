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
    
    # Fill cells with a newIndex generic
    newIndex = np.max(segmentedImg) + 1;
    for numZ in range(0, segmentedImg.shape[0]):
    	closedBackground = morphology.binary_closing(backgroundImg[numZ, :, :] == 0, morphology.disk(10))
    	filledZ = morphology.remove_small_holes(backgroundImg[numZ, :, :] == 0, area_threshold = 15000)
    	segmentedImgFilled[numZ, : , :] = (filledZ & (closedBackground == 0) & (backgroundImg[numZ, :, :] == 1)) * newIndex

    # Divide cells per ID
    newLabelsImg = morphology.label(segmentedImgFilled == newIndex);
    newLabelsImg = morphology.remove_small_objects(newLabelsImg, min_size=50000)
    with napari.gui_qt():
        print('Waiting on napari')
        viewer = napari.view_image(segmentedImg, rgb=False)
        viewer.add_labels(newLabelsImg, name='newLabelsOnly')
        viewer.add_labels(segmentedImgFilled, name='segmentedImgFilled')

    # Add new cells to image with a new ID
    for newLabel in np.sort(np.unique(newLabelsImg)):
        if newLabel != 0:
            segmentedImgFilled[newLabelsImg == newLabel] = newIndex
            newIndex = newIndex + 1


    # Add the original IDs
    segmentedImgFilled[segmentedImgFilled == 0] = segmentedImg[segmentedImgFilled == 0]

	#Divide background into two regions: bottom and top with different IDs
    for numBackgroundIds in backgroundIndices:
        segmentedImgFilled[segmentedImgFilled == numBackgroundIds] = 0

    backgroundLabelled = morphology.label(segmentedImgFilled == 0)

    segmentedImgFilled[backgroundLabelled == 1] = 0
    segmentedImgFilled[backgroundLabelled == 2] = newIndex;

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

		# Fill empty spaces by new cells within the tissue
		fillEmptyCells(seg_img, backgroundIndices)

		if firstZ > lastZ: # We are at basal layer
			#Need to exchange first and last
			aux = firstZ
			firstZ = lastZ
			lastZ = aux

		# Watershed by existing cells using previous cells as seeds
		print('Watershed ongoing...')
		for numZ in range(lastZ, firstZ-1, -1):
			print(numZ)

			# Use previous layer as seeds
			previousLayer = seg_img[numZ+1, :, :];
			previousLayer[previousLayer == backgroundIndices[0]] = 0
			previousLayer[previousLayer == backgroundIndices[1]] = 0

			if numZ == lastZ:
				newLabelsImg = morphology.label(previousLayer == 0);
				newLabelsImg[newLabelsImg > 0] = newLabelsImg[newLabelsImg > 0] + (np.max(previousLayer) + 1);
				previousLayer[newLabelsImg > 0] = newLabelsImg[newLabelsImg > 0] 

			erodedPreviousLayer = np.zeros_like(previousLayer);
			
			idCells = np.sort(np.unique(previousLayer));
			idCells = idCells[1:]

			#Erode each cell
			for numCell in idCells:
				newErodedCell = morphology.binary_erosion(previousLayer == numCell, morphology.disk(10))
				if np.sum(newErodedCell == 1) == 0:
					newErodedCell = morphology.binary_erosion(previousLayer == numCell, morphology.disk(3))
					if np.sum(newErodedCell == 1) == 0:
						newErodedCell = previousLayer == numCell
				
				erodedPreviousLayer[newErodedCell] = numCell

			denoised = filters.gaussian(raw_img[numZ, :, :], sigma=1)

			watershedImg = segmentation.watershed(denoised, markers = erodedPreviousLayer)

			#CARE: We can try different methods. Also beware of bottom intensities usually drop down 
			background_threshold = np.quantile(raw_img[numZ, :, :], 0.7)
			raw_img_onlyTissue = morphology.binary_closing(raw_img > background_threshold, morphology.disk(10));
			watershedImg[raw_img_onlyTissue == 0] = 0

			if numZ == 15:
				with napari.gui_qt():
					print('Waiting on napari')
					viewer = napari.view_image(denoised, rgb=False)
					viewer.add_labels(erodedPreviousLayer, name='erodedPreviousLayer')
					viewer.add_labels(raw_img_onlyTissue, name='raw_img_onlyTissue')
					viewer.add_labels(watershedImg, name='watershed')

			
			seg_img[numZ, :, :] = watershedImg



		outputFile = raw_folder + 'Segmentations_reviewed/' + file_name + '_watersheded.h5'
		postProcessFile = h5py.File(outputFile, 'w')
		segmentationFile = postProcessFile.create_dataset("segmentation", data=seg_img, dtype='uint16')
		postProcessFile.close()