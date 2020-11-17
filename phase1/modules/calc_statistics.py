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
# You can run this file to calculate statistics on the images, namely the
# MSE, SSIM, and PSNR.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 1: The noise reduction filters.
Excecutes the following algorithms:
- Mean squared error (MSE)
- Structural similarity index measure (SSIM)
- Peak signal-to-noise ratio (PSNR).
"""

import numpy as np
import SimpleITK as sitk
import math
from skimage.metrics import structural_similarity as ssim

def calc_mse(imgorg, imgfilter):
    """ Calculate the mean squared error (MSE). """
    image_org = imgorg.astype(np.float64) / 255.
    image_filter = imgfilter.astype(np.float64) / 255.
    return np.mean((image_org - image_filter) ** 2)

def calc_ssim(imgorg, imgfilter):
    """ Calculate the structural similarity index measure (SSIM). """
    image_org = imgorg.astype(np.float64) / 255.
    image_filter = imgfilter.astype(np.float64) / 255.
    return ssim(image_org, image_filter, data_range= image_filter.max() - image_filter.min())

def calc_psnr(imgorg, imgfilter):
    """ Calculate the peak signal-to-noise ratio (PSNR). """
    image_org = imgorg.astype(np.float64) / 255.
    image_filter = imgfilter.astype(np.float64) / 255.
    mse = np.mean((image_org - image_filter) ** 2)
    # MSE is zero means no noise is present in the signal
    # Therefore, PSNR is 100.
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
