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
# You can run this file to show some results.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 2a: The heuristic segmentation models:
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def sitk_show(img, title=None, seeds=None, margin=0.05, dpi=40):
    """ Show the visualisations from an SITK image. """
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)

    if title:
        plt.title(title)

    if seeds:
        x_val = [x[0] for x in seeds]
        y_val = [x[1] for x in seeds]
        plt.scatter(x_val, y_val, c='red')

    plt.show()

def show_results(results):
    """ show results mean and standard deviation. """
    for result in results:
        # print name of result, define dataset
        print(result[0])
        dataset = result[1]
        for data, values in dataset.items():
            print(data, np.mean(values), np.std(values))
