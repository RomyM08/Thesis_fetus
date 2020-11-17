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
# You can run this file to compute the parameters of the models.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 2a: The heuristic segmentation models:
- semi-automatic watershed segmentation
- fully automatic watershed segmentation
"""

import csv
import time
from tqdm import tqdm

from modules.calc_heuristic_models import *
from modules.calc_statistics import calc_dsc


def calc_params_ws_semiauto(img_tocompute, img_gt, sigmas, levels1, levels2, PATH, dataset, filename):
    """ Grid search of semi-automatic watershed segmentation parameters.
        Create a dictionary which saves all the results based on their names.
        The names contain the values of the parameters. """

    for sigma in tqdm(sigmas):
        time.sleep(0.1)
        for level1 in levels1:
            for level2 in levels2:
                # calculate watershed image
                img_watershed = calc_ws_semiauto(img_tocompute, key=dataset, sigma=sigma, level1=level1, level2=level2)
                name = 'sigma' + str(sigma) + "_" + "levelone" + str(level1) + "_" + "leveltwo" + str(level2)

                # calculate dice coefficient
                dice = calc_dsc(img_gt, img_watershed)

                # save results in csv file
                with open(PATH + '/' + dataset + '_' + filename + ".csv","a") as file:
                    csvwriter = csv.writer(file, delimiter=',')
                    csvwriter.writerow([sigma, level1, level2, dice])

                print('result appended of '+ name + 'dice:' + str(dice))


def calc_params_ws_fullyauto(img_tocompute, img_gt, metadata, sigmas, levels1, levels2, PATH, dataset, filename):
    """ Grid search of fully automatic watershed segmentation parameters.
        Create a dictionary which saves all the results based on their names.
        The names contain the values of the parameters. """

    for sigma in tqdm(sigmas):
        time.sleep(0.1)
        for level1 in levels1:
            for level2 in levels2:
                # calculate watershed image
                img_watershed = calc_ws_fullyauto(img_tocompute, metadata, sigma=sigma, level1=level1, level2=level2)
                name = 'sigma' + str(sigma) + "_" + "levelone" + str(level1) + "_" + "leveltwo" + str(level2)

                # calculate dice coefficient
                dice = calc_dsc(img_gt, img_watershed)

                # save results in csv file
                with open(PATH + '/' + dataset + '_' + filename + ".csv","a") as file:
                    csvwriter = csv.writer(file, delimiter=',')
                    csvwriter.writerow([sigma, level1, level2, dice])

                print('result appended of '+ name)
