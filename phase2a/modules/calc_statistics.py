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
# You can run this file to compute the statisical evaluation metrics on the images,
# namely the DSC, IoU, and HD.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 2a: The statistics of the heuristic segmentation models:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Haussdorff Distance (HD)
"""

import numpy as np
import SimpleITK as sitk


def calc_dsc(y_true, y_pred):
    """ Calculate the Dice Similarity Coefficient (DSC). """
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = 2. * np.sum(y_true_f * y_pred_f) + smooth
    union = np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    dice = intersection/union

    return dice

def calc_iou(y_true, y_pred):
    """ Calculate the Intersection over Union (IoU). """
    smooth = np.finfo(float).eps
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection + smooth) / np.sum(union + smooth)

    return iou_score

def calc_hd(y_true, y_pred):
    """ Calculate the Haussdorff Distance (HD). """
    img_true = sitk.GetImageFromArray(y_true)
    img_pred = sitk.GetImageFromArray(y_pred)
    hd = sitk.HausdorffDistanceImageFilter()
    hd.Execute(img_true, img_pred)
    distance = hd.GetHausdorffDistance()

    return distance
