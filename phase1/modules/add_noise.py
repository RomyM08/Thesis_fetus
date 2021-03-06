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
# You can run this file to add speckle noise to the image. 
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 1: The noise reduction filters.
Excecutes the following algorithms:
- Speckle noise
"""

import SimpleITK as sitk

def add_specklenoise(img, std=0.1, seed=42):
    """ Add speckle noise to the image. """
    speckleFilter = sitk.SpeckleNoiseImageFilter()
    speckleFilter.SetStandardDeviation(std)
    speckleFilter.SetSeed(seed)
    imgSpeckle = speckleFilter.Execute(img)
    return imgSpeckle
