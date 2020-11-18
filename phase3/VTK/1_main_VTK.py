# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file is part of a program that is used to develop an objective way to
# segment the fetus from ultrasound images, and to analyse the effectiveness of
# using the resulting mask to produce an unobstructed visualisation of the fetus.
# The research is organised in three phases: (1) noise reduction filters,
# (2a) heuristic segmentation models, (2b) deep learning segmentation
# approach (U-net), and (3) the volume visualisation. The program is developed
# for the master Computational Science at the UvA from February to November 2020.
#
# This file contains code for the volume visualisation in VTK.
# You can run this main file to render the visualisation.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import os
import sys

from classes.reader import Reader
from classes.volume import Volume
from classes.viewports import Viewports
from classes.mask import Mask
from helpers.loadsave import *
from helpers.queries import *
from modules.convert_nptovtk import *


# Constants
DATA_PATH = '../../datasets/'
RESULTS_PATH = 'results_VTK'
RESULTS_IMG_PATH = os.path.join(RESULTS_PATH, 'results_VTK_img')
RESULTS_META_PATH =  os.path.join(RESULTS_PATH, 'results_VTK_metadata')

DATA_SMOOTHED_PATH = '../results_volumes/results_volumes_convert/convert_smoothed'
DATA_HEURISTICS_PATH = '../../phase2a/results_heuristic_models/results_heuristics_img'
DATA_UNET_PATH = '../results_volumes/results_volumes_convert/convert_unet'


def main():
    # ask two questions to the user
    # what do you want to visualise? (volume, mask, tool)
    visualise = query_function_visualisation(question = "What do you want to visualise?",  default='volume')
    # which dataset do you want to visualise? (dataset1, .., dataset7)
    from_dataset = query_function_dataset(question= "Which dataset do you want to visualise?", default='dataset1')

    # show what is going to be processed
    if visualise == 'tool':
        print('process:', visualise, '--', from_dataset)
    elif visualise == 'volume':
        # Which image do you want to visualise? (real_org, crop_org, crop_smoothed)
        from_image = query_function_volume_image(question="Which image do you want to visualise?", default='crop_org')
        print('process:',  visualise, '--', from_dataset, from_image)
    elif visualise == 'mask':
        # Of which input image do you want to apply the mask? (original, smoothed)
        from_image = query_function_image(question="Of which input image do you want to apply the mask?", default='org')
        # from the model: ground truth, heuristic model, u-net
        from_model = query_function_model(question="Of which model do you want to apply the mask?", default='ground_truth')
        if from_model == 'ground_truth':
            print('process:',  visualise, '--', from_dataset, from_model, from_image)
        elif from_model == 'heuristic':
            # Which heuristic model do you want? (Semi-automatic, fully automatic segmentation model)
            from_heuristic = query_function_heuristic(question="Which heuristic segmentation model do you want?", default='ws_semiauto')
            print('process:',  visualise, '--', from_dataset, from_model, from_heuristic, from_image)
        elif from_model == 'unet':
            # Which activation function do you want? (ReLU, ELU)
            from_activ = query_function_activation(question="Which activation function do you want?", default='relu')
            # Which simulation do you want? (number)
            from_simulation = query_function_simulation(question="Which simulation do you want?", default='1')
            print('process:',  visualise, '--', from_dataset, from_model, from_activ, from_image, from_simulation)

    # question whether the image needs to be saved
    saving_bool = query_function_saving(question="Do you want to save the visualisation?", default='no')

    # import the dataset paths
    dataset = import_paths(DATA_PATH, from_dataset)
    if dataset == {}:
        print(from_dataset, 'does not exist yet.')
        sys.exit(1)

    # create results folder
    create_folder(RESULTS_PATH)

    # create results folder for images
    create_folder(RESULTS_IMG_PATH)

    # initialise the visualisation
    read = Reader()

    # visualise the volume, mask, or tool
    if visualise == "volume":
        # visualise the volume of the real original image,
        # the cropped original image or the cropped smoothed image
        if from_image == 'real_org':
            # except for dataset1, all the real original images can be visualised
            read.input_image(dataset['realorg'])
            datatype = 'realorg'
        if from_image == 'crop_org':
            # read the cropped original dataset
            read.input_image(dataset['org'])
            datatype = 'croporg'
        elif from_image == 'crop_smoothed':
            # read the cropped smoothed dataset
            # load the metadata of the corresponding cropped original image
            metadata = load_metadata(PATH=RESULTS_META_PATH, filename=from_dataset, datatype ='croporg')

            # load the numpy image and convert to vtk
            all_images = load_smoothed_images(DATA_SMOOTHED_PATH, 'smoothed_images')
            np_smoothed_img = all_images[from_dataset]
            vtk_model = numpy_array_as_vtk_image_data(np_smoothed_img, metadata)
            smoothed_image =  read.cast_image(vtk_model)
            read.reader = smoothed_image

        # for other use, generate and save the metadata of the volume
        # of the real original image, or the croppped original image
        if from_image ==  'real_org' or from_image == 'crop_org':
            create_folder(RESULTS_META_PATH)
            metadata = read.generate_metadata()
            save_metadata(PATH = RESULTS_META_PATH, dataset=metadata, filename=from_dataset, datatype=datatype)

        # visualise the dataset as a volume
        volume = Volume()

        # whether or not to save the image
        if saving_bool == True:
            volume.pipeline_volume(read.reader, PATH=RESULTS_IMG_PATH, save_img=True)
        else:
            volume.pipeline_volume(read.reader)


    elif visualise == 'mask':
        # visualise the mask over the original/smoothed image
        if from_image == 'org' and not from_model == 'unet':
            # the origial image
            read.input_image(dataset['org'])
            input_image = read.cast_image(read.reader)
            # generate the metadata of the volume
            metadata = read.generate_metadata()
        elif from_image == 'smoothed' and not from_model == 'unet':
            # the smoothed image
            # load the metadata of the corresponding cropped original image
            metadata = load_metadata(PATH=RESULTS_META_PATH, filename=from_dataset, datatype ='croporg')
            # load the numpy image and convert to vtk
            all_images = load_smoothed_images(DATA_SMOOTHED_PATH, 'smoothed_images')
            np_smoothed_img = all_images[from_dataset]
            vtk_smoothed_img = numpy_array_as_vtk_image_data(np_smoothed_img, metadata)
            smoothed_image =  read.cast_image(vtk_smoothed_img)
            read.reader = smoothed_image
            input_image = read.cast_image(read.reader)

        # the ground truth mask images (binary)
        if from_model == 'ground_truth':
            read.input_image(dataset['gt'])
            mask_image = read.cast_image(read.reader)

        # the heuristic segmentation mask images
        elif from_model == 'heuristic':
            # load the numpy image mask (binary)
            # and create the vtk image mask
            try:
                np_model = load_heuristic_model(PATH=DATA_HEURISTICS_PATH, dataset=from_dataset, model=from_heuristic, inputimg=from_image)
                vtk_model = numpy_array_as_vtk_image_data(np_model, metadata)
                mask_image = read.cast_image(vtk_model)
            except Exception:
                print(from_dataset + '_' + from_model, 'is not generated. ')
                print('visualisation cannot be rendered. ')
                sys.exit(1)

        # the U-net mask images
        elif from_model == 'unet':
            # load the numpy image mask (binary)
            # and create the vtk image
            try:
                # load all the images of the simulation
                np_all_input_images = load_unet_model(PATH=DATA_UNET_PATH, datatype='org', simulation=from_simulation)
                np_all_mask_images =  load_unet_model(PATH=DATA_UNET_PATH, datatype='pred', simulation=from_simulation)

                # get the specific image
                key = str(from_dataset + '_' + from_activ + '_' + from_image + from_simulation)
                np_input_image = np_all_input_images[key]
                np_mask_image = np_all_mask_images[key]

                # load the metadata of the corresponding cropped original image
                # and rewrite the metadata (dimensionality is different)
                metadata = load_metadata(PATH=RESULTS_META_PATH, filename=from_dataset, datatype ='croporg')
                print(metadata)

                # update the metadata (dimensionality is different than the cropped original image)
                new_pixeldims = (np_input_image.shape[2], np_input_image.shape[1], np_input_image.shape[0])
                new_extent = (0, np_input_image.shape[2]-1, 0, np_input_image.shape[1]-1, 0, np_input_image.shape[0]-1)
                metadata.update(ConstPixelDims=new_pixeldims, ConstExtent=new_extent)

                # convert the images to vtk
                vtk_input_image = numpy_array_as_vtk_image_data(np_input_image, metadata)
                vtk_mask_image = numpy_array_as_vtk_image_data(np_mask_image, metadata)

                # input and mask image
                input_image = read.cast_image(vtk_input_image)
                read.reader = input_image
                mask_image = read.cast_image(vtk_mask_image)

            except Exception:
                print('simulation', from_simulation, 'is not generated. ')
                print('visualisation cannot be rendered. ')
                sys.exit(1)

        # show result
        outputmask = Mask(input_image, mask_image).output_mask
        volume = Volume()

        # whether or not to save the image
        if saving_bool == True:
            volume.pipeline_volume(outputmask, PATH=RESULTS_IMG_PATH, save_img=True)
        else:
            volume.pipeline_volume(outputmask)


    elif visualise == "tool":
        # the tool automatically visualises the cropped original image of the given dataset
        # it is not nesecarry to implement all readers (max of 4 readers)
        # for now, 4 viewports are vissible
        read.input_image(dataset['org'])
        viewports = Viewports(read.reader)

        # define what has to be seen
        iren_list = ["Sagittal view", "Transverse view", "Volume", "Coronal view"]

        # whether or not to save the image
        if saving_bool == True:
            viewports.view(iren_list, PATH=RESULTS_IMG_PATH, save_img=True)
        else:
            viewports.view(iren_list)


if __name__ == "__main__":
    main()
