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
# You can run this file to read and cast the input image.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import vtk


class Reader():
    """
    This is a class that reads the dataset.
    """

    def __init__(self):

        # the reader
        self.reader = 0

    def input_image(self, input):
        """ Check what kind of input the program gets and guide to the
            corresponding pipeline. """

        # check the last value of the input string
        if input[-1] == "/":
            # the input is a directory
            self.read_directory(input)
        elif input[-4:] == '.vti':
            # the input is a vti file
            self.read_vti(input)
        elif input[-4:] == '.dcm':
            # the input is a dcm file
            self.read_dcm(input)
        else:
            print("Filename is not recognized, please insert a directory /, .vti, or .dcm.")

    def read_directory(self, input):
        """ Read the directory. """
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(input)
        self.reader.Update()

    def read_vti(self, input):
        """ Read the .vti file. """
        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(input)
        self.reader.Update()

    def read_dcm(self, input):
        """ Read the dicom file. """
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetFileName(input)
        self.reader.Update()

    def generate_metadata(self):
        """ Check the metadata about the image, specifically
        the image dimensions, the pixelspacing, spacing between
        slices, the orientation and position of the patient. """

        # get the metadata
        ConstPixelDims = self.reader.GetOutput().GetDimensions()
        ConstExtent = self.reader.GetDataExtent()
        ConstPixelSpacing = self.reader.GetPixelSpacing()
        ConstDataSpacing = self.reader.GetDataSpacing()
        ConstOrigin = self.reader.GetDataOrigin()
        ConstOrientation = self.reader.GetImageOrientationPatient()
        ConstPosition = self.reader.GetImagePositionPatient()

        # put everything in a dictionary
        metadata = {"ConstPixelDims": ConstPixelDims,
                    "ConstExtent": ConstExtent,
                    "ConstPixelSpacing": ConstPixelSpacing,
                    "ConstDataSpacing": ConstDataSpacing,
                    "ConstOrigin": ConstOrigin,
                    "ConstOrientation": ConstOrientation,
                    "ConstPosition": ConstPosition}

        return metadata

    def cast_image(self, image):
        """ This filter casts the input type to match the output type in the
            image processing pipeline. The filter does nothing if the input
            already has the correct type. To specify the "CastTo" type,
            use "SetOutputScalarType" method. """

        cast = vtk.vtkImageCast()
        try:
            # works for vtk.vtkImageImport
            cast.SetInputConnection(image.GetOutputPort())
        except Exception:
            pass

        try:
            # works for vtk.vtkImageData
            cast.SetInputData(image)
        except Exception:
            pass

        cast.SetOutputScalarTypeToUnsignedChar()
        cast.Update()

        return cast
