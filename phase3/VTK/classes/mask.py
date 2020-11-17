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
# This file contains code for the volume visualisation in VTK.
# You can run this file to apply the segmented mask over the input image.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import vtk


class Mask():
    """
    This is a class that implements the segmented mask over the
    input image to only segment the fetus.
    """

    def __init__(self, input_image, mask_image):

        # The input image, mask and segmented image
        self.input_image = input_image
        self.input_mask = mask_image
        self.output_mask = 0  #segmented_image

        self.apply_mask()

    def apply_mask(self):
        """ Apply the input mask on the input image in order to get an
            output mask which only shows the fetus. """

        imageMask = vtk.vtkImageMask()

        # set the input to be masked
        imageMask.SetInputConnection(self.input_image.GetOutputPort())

        # set the mask to be used
        imageMask.SetMaskInputData(self.input_mask.GetOutput())

        # the segmented result
        self.output_mask = imageMask
