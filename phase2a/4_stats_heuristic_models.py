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
# You can run this file to compute the statistics of the models.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 2a: The statistics of the heuristic segmentation models:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Haussdorff Distance (HD)
"""

import os

from helpers.showing import show_results
from helpers.loadsave import *
from modules.calc_statistics import *


# Constants
DATA_PATH = '../datasets/'
RESULTS_PATH = 'results_heuristic_models'
RESULTS_IMG_PATH = os.path.join(RESULTS_PATH, 'results_heuristics_img')
RESULTS_STATS_PATH = os.path.join(RESULTS_PATH, 'results_heuristics_stats')


def main():
    # load the ground truth 3D images and the heuristic model images
    # and show this in a dataset
    folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    heuristicnames = get_heuristicnames(RESULTS_IMG_PATH)
    datasets = get_data_heuristics(DATA_PATH, RESULTS_IMG_PATH, folders, heuristicnames)

    # create results directory for statistics
    print('Create directory')
    create_dir(RESULTS_STATS_PATH)

    # calculate the statistics: dice similarity coefficient (DSC), intersection over
    # union (IoU), and hausdorff distance (HD)
    results_dsc = {}
    results_iou = {}
    results_hd = {}

    for model in heuristicnames:
        results_dsc[model] = []
        results_iou[model] = []
        results_hd[model] = []


    # iterate over datasets dictionary
    print('Results of: model, dice, ioun hausdorff dist.')
    for dataset, values in datasets.items():
        print(dataset)

        # images to be used
        imgGT = values['gt']
        imgModels = values['models']

        # calculate and save results
        for modelname, model in imgModels.items():
            dice = calc_dsc(imgGT, model)
            iou = calc_iou(imgGT, model)
            hd = calc_hd(imgGT, model)
            print(modelname, dice, iou, hd)

            for m in heuristicnames:
                if modelname == m:
                    results_dsc[m].append(dice)
                    results_iou[m].append(iou)
                    results_hd[m].append(hd)

    # save all results in pickle dictionary
    results_pickle = {'DSC': results_dsc, 'IoU': results_iou, 'HD': results_hd}
    save_dict_pickle(PATH = RESULTS_STATS_PATH, data= results_pickle, filename='results_pickle')

    # save specific results with mean and standard deviation
    results = [('DSC:', results_dsc), ('IoU:', results_iou), ('HD:', results_hd)]
    show_results(results)
    save_results(PATH = RESULTS_STATS_PATH, data=results, filename='results_mean_std')


if __name__ == '__main__':
    main()
