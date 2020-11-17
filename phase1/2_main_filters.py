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
# This file contains code for the noise reduction filters.
# You can run this main file to compute the noise reduction filters.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 1: The noise reduction filters.
- Gaussian image filter
- Median image filter
- Curvature flow image filter
- Anisotropic diffusion image filter
"""

import os

from helpers.loadsave import get_data_scans, create_dir, save_data_pickle
from modules.add_noise import *
from modules.calc_filters import *


# Constants
DATA_PATH = '../datasets/'
RESULTS_PATH = 'results_filters'
RESULTS_IMG_PATH = os.path.join(RESULTS_PATH, 'results_filters_img')


def main():
    # load the original 3D image and show this in a dataset
    folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    datasets = get_data_scans(DATA_PATH, folders)

    print('Create directories')
    # create results directory
    create_dir(RESULTS_PATH)

    # create results directory for images
    create_dir(RESULTS_IMG_PATH)

    # iterate over datasets dictionary
    for key,value in datasets.items():
        print(key)

        # image to apply filter on
        imgOrg = value['org']
        imgSpeckle = add_specklenoise(imgOrg, std = 0.2)
        img_to_process = imgSpeckle

        # calculate filters
        filters = ['original', 'speckle', 'gaussian', 'median', 'curvatureflow', 'anisodiff']
        if 'original' in filters:
            # the original image
            img_org = sitk.GetArrayFromImage(imgOrg)
            save_data_pickle(data = img_org, PATH = RESULTS_IMG_PATH, dataset=key, filename= 'original')

        if 'speckle' in filters:
            # the original image with speckle noise
            img_speckle = sitk.GetArrayFromImage(imgSpeckle)
            save_data_pickle(data = img_speckle, PATH = RESULTS_IMG_PATH, dataset=key, filename= 'speckle')

        if 'gaussian' in filters:
            # the smoothing recursive Gaussian image filter
            sigma = [1,2,3]
            for sig in sigma:
                img_gaus = calc_gaussian(img_to_process, sigma=sig)
                img_gaus = sitk.GetArrayFromImage(img_gaus)
                save_data_pickle(data = img_gaus, PATH = RESULTS_IMG_PATH,  dataset=key, filename= 'gaussian_' + str(sig))

        if 'median' in filters:
            # the median image filter
            radius = [1,2,3]
            for rad in radius:
                img_med = calc_median(img_to_process, radius=rad)
                img_med = sitk.GetArrayFromImage(img_med)
                save_data_pickle(data = img_med, PATH = RESULTS_IMG_PATH,  dataset=key, filename= 'median_' + str(rad))

        if 'curvatureflow' in filters:
            # the curvature flow image filter
            iteration = [5, 10]
            timestep = [0.125, 0.250]
            for i in iteration:
                for t in timestep:
                    img_curv = calc_curvatureflow(img_to_process, iteration=i, step=t)
                    img_curv = sitk.GetArrayFromImage(img_curv)
                    save_data_pickle(data = img_curv, PATH = RESULTS_IMG_PATH, dataset=key, filename= 'curvatureflow_' + str(i) + '_' + str(t))

        if 'anisodiff' in filters:
            # the Gradient Anisotropic diffusion image filter
            c = 4
            iteration = [10, 15]
            timestep = [0.04, 0.06]
            for i in iteration:
                for t in timestep:
                    img_ani = calc_anisodiff(img_to_process, iteration=i, step=t, conductance=c)
                    img_ani = sitk.GetArrayFromImage(img_ani)
                    save_data_pickle(data = img_ani, PATH = RESULTS_IMG_PATH,  dataset=key, filename= 'anisodiff_' + str(c) + '_' + str(i) + '_' + str(t))


if __name__ == '__main__':
    main()
