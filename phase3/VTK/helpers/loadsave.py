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
# You can run this file to load and save data.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import os
import pickle


""" Create folders. """
def create_folder(PATH):
    """ Create a folder. """
    try:
        os.mkdir(PATH)
        print('Directory', PATH, 'created' )
    except FileExistsError:
        print('Directory', PATH, 'already exists' )


""" Loading functions. """
def import_paths(PATH, from_dataset):
    """ Import all the necessary dataset paths.
        output: dataset dictionary with the real original image,
        cropped original image, and the ground truth mask image. """
    dataset = {}

    path_real_org = PATH + from_dataset + "/real_org/"
    path_crop_org = PATH + from_dataset + "/crop_org/"
    path_crop_gt = PATH + from_dataset + "/crop_gt/"

    dataset.update({'realorg': path_real_org, 'org': path_crop_org, 'gt': path_crop_gt})

    return dataset

def load_metadata(PATH, filename, datatype):
    """ Load the metadata using pickle. """
    with open(PATH + '/' + filename + '_' + datatype + ".pkl","rb") as f:
        new_data = pickle.load(f)

    return new_data

def load_smoothed_images(PATH, filename):
    """ Load data of the smoothed images using pickle. """
    with open(PATH + '/' + filename + ".pkl","rb") as f:
        new_data = pickle.load(f)

    print(filename, "opened")
    return new_data

def load_heuristic_model(PATH, dataset, model, inputimg):
    """ Load data of the heuristic model using pickle. """
    with open(PATH + '/' + dataset + '_' + model + '_' + inputimg + ".pkl","rb") as f:
        new_data = pickle.load(f)

    print(dataset + '_' + model + '_' + inputimg, "opened")
    return new_data

def load_unet_model(PATH, datatype, simulation):
    """ Load data of the U-net model using pickle. """
    with open(PATH + '/' + datatype + '_unet_images' + simulation + ".pkl","rb") as f:
        new_data = pickle.load(f)

    print(datatype + '_unet_images' + simulation, "opened")
    return new_data


""" Save functions. """
def save_metadata(PATH, dataset, filename, datatype):
    """ Save the metadata in pickle. """
    with open(PATH + '/' + filename + '_' + datatype + ".pkl","wb") as f:
        pickle.dump(dataset,f)
    print('metadata', filename, "created")
