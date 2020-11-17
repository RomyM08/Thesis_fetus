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
# This file contains code for the heuristic segmentation models.
# You can run this main file to compute the models with the given parameters.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 2a: The heuristic segmentation models:
- semi-automatic watershed segmentation
- fully automatic watershed segmentation
"""

import os

from helpers.loadsave import get_data_scans, create_dir, save_data_pickle, load_metadata
from modules.calc_heuristic_models import *


# Constants
DATA_PATH = '../datasets/'
RESULTS_PATH = 'results_heuristic_models'
RESULTS_IMG_PATH = os.path.join(RESULTS_PATH, 'results_heuristics_img')
RESULTS_META_PATH_VTK = '../phase3/VTK/results_VTK/results_VTK_metadata'

def main():
    # load the original 3D image and the smoothed filtered image
    # and show this in a dataset
    folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    filterdata = {'filtername': 'curvatureflow', 'parameters': [5, 0.125]}
    datasets = get_data_scans(DATA_PATH, folders, filterdata)

    print('Create directories')
    # create results directory
    create_dir(RESULTS_PATH)

    # create results directory for images
    create_dir(RESULTS_IMG_PATH)

    # iterate over datasets dictionary
    for key, value in datasets.items():
        print(key)

        # images to apply models on
        img_org = value['org']
        img_smoothed = value['smoothed']


        # calculate heuristic model
        models = ['ws_semiauto', 'ws_fullyauto']
        if 'ws_semiauto' in models:
            # the semi-automatic watershed segmentation model
            # compute the watershed on the original image and save the numpy image in pickle file
            img_ws_org = calc_ws_semiauto(img_org, key, sigma=1.2, level1=4, level2=1, showing=False)
            save_data_pickle(PATH = RESULTS_IMG_PATH, data = img_ws_org, dataset=key, filename= 'ws_semiauto_org')

            # compute the watershed on the smoothed image and save the numpy image in pickle file
            img_ws_smoothed = calc_ws_semiauto(img_smoothed, key, sigma=1.2, level1=4, level2=1, showing=False)
            save_data_pickle(PATH = RESULTS_IMG_PATH, data = img_ws_smoothed, dataset=key, filename= 'ws_semiauto_smoothed')

        if 'ws_fullyauto' in models:
            # the fully-automatic watershed segmentation model
            # load the metadata
            metadata = load_metadata(PATH = RESULTS_META_PATH_VTK, filename = key)

            # compute the watershed on the original image save the numpy image in pickle file
            img_ws_org = calc_ws_fullyauto(img_org, metadata, sigma=1.2, level1=4, level2=1, showing=False)
            save_data_pickle(PATH = RESULTS_IMG_PATH, data = img_ws_org, dataset=key, filename= 'ws_fullyauto_org')

            # compute the watershed on the smoothed image and save the numpy image in pickle file
            img_ws_smoothed = calc_ws_fullyauto(img_smoothed, metadata, sigma=1.2, level1=4, level2=1, showing=False)
            save_data_pickle(PATH = RESULTS_IMG_PATH, data = img_ws_smoothed, dataset=key, filename= 'ws_fullyauto_smoothed')


if __name__ == "__main__":
    main()
