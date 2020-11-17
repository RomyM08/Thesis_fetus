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
# You can run this main file to compute the statistics of the filters.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 1: The noise reduction filters.
- Mean squared error (MSE)
- Structural similarity index measure (SSIM)
- Peak signal-to-noise ratio (PSNR).
"""

import os

from helpers.showing import show_results
from helpers.loadsave import *
from modules.calc_statistics import *

# Constants
DATA_PATH = '../datasets/'
RESULTS_PATH = 'results_filters'
RESULTS_IMG_PATH = os.path.join(RESULTS_PATH, 'results_filters_img')
RESULTS_STATS_PATH = os.path.join(RESULTS_PATH, 'results_filters_stats')


def main():
    # load the original, ground truth, and filtered 3D images
    # and show this in a dataset
    folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    filternames = get_filternames(RESULTS_IMG_PATH)
    datasets = get_data_filters(DATA_PATH, RESULTS_IMG_PATH, folders, filternames)

    # create results directory for statistics
    print('Create directory')
    create_dir(RESULTS_STATS_PATH)

    # calculate the statistics MSE (mean squared error), SNR (signal-to-noise)
    # and PSNR (Peak signal to noise) per filter
    results_mse = {}
    results_ssim = {}
    results_psnr = {}

    for filter in filternames:
        results_mse[filter] = []
        results_ssim[filter] = []
        results_psnr[filter] = []

    # iterate over datasets dictionary
    print('Results of: filter, mse, snr, psnr')
    for dataset, values in datasets.items():
        print(dataset)

        # images to be used
        imgOrg = values['org']
        # imgGT = values['gt']
        imgFilters = values['filters']

        # calculate and save results
        for filtername, filter in imgFilters.items():
            mse = calc_mse(imgOrg, filter)
            ssim = calc_ssim(imgOrg, filter)
            psnr = calc_psnr(imgOrg, filter)
            print(filtername, mse, ssim, psnr)

            for f in filternames:
                if filtername == f:
                    results_mse[f].append(mse)
                    results_ssim[f].append(ssim)
                    results_psnr[f].append(psnr)

    # save all results in pickle dictionary
    results_pickle = {'MSE': results_mse, 'SSIM': results_ssim,'PSNR': results_psnr}
    save_dict_pickle(PATH = RESULTS_STATS_PATH, data= results_pickle, filename='results_pickle')

    # save all results with mean and standard deviation
    results = [('MSE:', results_mse), ('SSIM:', results_ssim), ('PSNR:', results_psnr)]
    show_results(results)
    save_results(PATH = RESULTS_STATS_PATH, data=results, filename='results_mean_std')


if __name__ == '__main__':
    main()
