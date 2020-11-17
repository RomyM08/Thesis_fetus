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
# You can run this main file to compute the parameters of the models.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 2a: The heuristic segmentation models:
- semi-automatic watershed segmentation
- fully automatic watershed segmentation
"""

import os
import sys
import numpy as np

from helpers.loadsave import *
from modules.calc_parameters import *


# Constants
DATA_PATH = '../datasets/'
RESULTS_PATH = 'results_heuristic_models'
RESULTS_PARA_PATH = os.path.join(RESULTS_PATH, 'results_heuristics_para')
RESULTS_META_PATH_VTK = '../phase3/VTK/results_VTK/results_VTK_metadata'


def main():
    # load the original 3D image, ground truth 3D image and the smoothed
    # filtered 3D image and show this in a dataset
    folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    filterdata = {'filtername': 'anisodiff', 'parameters': [10, 0.04, 4]}
    datasets = get_data_parascans(DATA_PATH, folders, filterdata)
    print(datasets.keys())

    print('Create directories')
    # create results directory
    create_dir(RESULTS_PATH)

    # create results directory for parameter results
    create_dir(RESULTS_PARA_PATH)

    # select for which dataset and which filters you want the parameters
    datasetkey = sys.argv[1] #e.g. 'dataset1'
    models = ['ws_semiauto', 'ws_fullyauto']

    # images to apply models on
    print(datasetkey)
    img_org = datasets[datasetkey]['org']
    img_gt = datasets[datasetkey]['gt']
    img_smoothed = datasets[datasetkey]['smoothed']

    # calculate heuristic models
    if 'ws_semiauto' in models:
        # the semi-automatic watershed segmentation model
        sigma = np.arange(0.2, 5.2, 0.2)
        level1 = np.arange(0.5, 5.2, 0.5)
        level2 = np.arange(0.5, 5.2, 0.5)

        # compute the watershed on the original image
        calc_params_ws_semiauto(img_org, img_gt, sigma, level1, level2, PATH= RESULTS_PARA_PATH, dataset=datasetkey, filename='ws_semiauto_org')

        # compute the watershed on the smoothed image
        calc_params_ws_semiauto(img_smoothed, img_gt, sigma, level1, level2, PATH= RESULTS_PARA_PATH, dataset=datasetkey, filename='ws_semiauto_smoothed')


    if 'ws_fullyauto' in models:
        # the fully automatic watershed segmentation model
        sigma = np.arange(0.2, 5.2, 0.2)
        level1 = np.arange(0.5, 5.2, 0.5)
        level2 = np.arange(0.5, 5.2, 0.5)

        # load the metadata
        metadata = load_metadata(PATH = RESULTS_META_PATH_VTK, filename = datasetkey)

        # compute the watershed on the original image
        calc_params_ws_fullyauto(img_org, img_gt, metadata, sigma, level1, level2, PATH= RESULTS_PARA_PATH, dataset=datasetkey, filename='ws_fullyauto_org')

        # compute the watershed on the smoothed image
        calc_params_ws_fullyauto(img_smoothed, img_gt, metadata, sigma, level1, level2, PATH= RESULTS_PARA_PATH, dataset=datasetkey, filename='ws_fullyauto_smoothed')


if __name__ == '__main__':
    main()
