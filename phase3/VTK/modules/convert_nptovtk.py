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
# You can run this file to convert numpy arrays to VTK.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import numpy as np
import vtk
from vtk.util import numpy_support


def numpy_array_as_vtk_image_data(source_numpy_array, metadata):
    """ Convert a numpy dataset to vtk in order to visualise it in VTK.
        Note: Channels are flipped of the np.ndarray source.
        The function returns vtk.vtkImageData. """

    # check datatype and dimension
    if not isinstance(source_numpy_array, np.ndarray):
        source_numpy_array = np.asarray(source_numpy_array)
        print('Mask was wrong datatype')
    if source_numpy_array.ndim != 3:
        raise ValueError("Only works with 3 dimensional arrays")

    # the metadata being used
    dims = metadata['ConstPixelDims']
    extent = metadata['ConstExtent']
    spacing = metadata['ConstPixelSpacing']
    origin = metadata['ConstOrigin']

    # generate the vtk-compatible image
    output_vtk_image = vtk.vtkImageData()
    output_vtk_image.SetDimensions(dims[0], dims[1], dims[2])

    vtk_type_by_numpy_type = {
        np.uint8: vtk.VTK_UNSIGNED_CHAR,
        np.uint16: vtk.VTK_UNSIGNED_SHORT,
        np.uint32: vtk.VTK_UNSIGNED_INT,
        np.uint64: vtk.VTK_UNSIGNED_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_UNSIGNED_LONG_LONG,
        np.int8: vtk.VTK_CHAR,
        np.int16: vtk.VTK_SHORT,
        np.int32: vtk.VTK_INT,
        np.int64: vtk.VTK_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_LONG_LONG,
        np.float32: vtk.VTK_FLOAT,
        np.float64: vtk.VTK_DOUBLE
    }

    vtk_datatype = vtk_type_by_numpy_type[source_numpy_array.dtype.type]

    source_numpy_array = np.flipud(source_numpy_array)
    source_numpy_array = np.fliplr(source_numpy_array)

    depth_array = numpy_support.numpy_to_vtk(source_numpy_array.ravel(), deep=True, array_type=vtk_datatype)
    depth_array.SetNumberOfComponents(1)
    output_vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    output_vtk_image.SetOrigin(origin[0], origin[1], origin[2])
    output_vtk_image.SetExtent(extent[0], extent[1], extent[2], extent[3], extent[4], extent[5])
    output_vtk_image.GetPointData().SetScalars(depth_array)

    output_vtk_image.Modified()
    return output_vtk_image
