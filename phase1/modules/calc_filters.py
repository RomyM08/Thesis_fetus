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
# You can run this file to calculate the specific filters, namely the
# Gaussian, median, curvature flow, and anisotrpic diffusion filter.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 1: The noise reduction filters.
Excecutes the following algorithms:
- Gaussian image filter
- Median image filter
- Curvature flow image filter
- Anisotropic diffusion image filter
"""

import SimpleITK as sitk

def calc_gaussian(img, sigma=3):
    """ The Gaussian image filter. """
    blurFilter = sitk.SmoothingRecursiveGaussianImageFilter()
    blurFilter.SetSigma(sigma)
    imgSmooth = blurFilter.Execute(img)
    return imgSmooth

def calc_median(img, radius=3):
    """ The median image filter. """
    blurFilter = sitk.MedianImageFilter()
    blurFilter.SetRadius(radius)
    imgSmooth = blurFilter.Execute(img)
    return imgSmooth

def calc_curvatureflow(img, iteration=5, step=0.125):
    """ The curvature flow image filter. """
    blurFilter = sitk.CurvatureFlowImageFilter()
    blurFilter.SetNumberOfIterations(iteration)
    blurFilter.SetTimeStep(step)
    imgSmooth = blurFilter.Execute(img)
    return imgSmooth

def calc_anisodiff(img, iteration=5, step=0.05, conductance=1):
    """ The Gradient Anisotropic diffusion image filter. """
    # make the input image to a float64
    img_input = sitk.Cast(img,sitk.sitkFloat64)

    # apply filter
    blurFilter = sitk.GradientAnisotropicDiffusionImageFilter()
    blurFilter.SetNumberOfIterations(iteration)
    blurFilter.SetTimeStep(step)
    blurFilter.SetConductanceParameter(conductance)
    imgSmooth = blurFilter.Execute(img_input)
    return imgSmooth
