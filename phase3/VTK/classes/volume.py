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
# You can run this file to render the volume.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import vtk


class Volume():
    """
    This is a class that renders the volume with a reader.
    """

    def __init__(self):

        # the reader which will be rendered as a volume
        self.reader = 0

    def pipeline_volume(self, reader, PATH=None, save_img = False, planes = False):
        """ The start of the pipeline of several volume visualisations.
            Either render the volume (True) or render the planes (False). """

        # the reader which will be rendered as a volume
        self.reader = reader
        # Boolean False (render volume), True (render planes)
        self.path = PATH
        self.planes = planes
        self.save_img = save_img

        # render the background
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.2, 0.2, 0.2)
        self.ren.SetBackground2(0.5, 0.5, 0.5)
        self.ren.GradientBackgroundOn()

        # the renderwindow
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(600, 600)

        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        # whether to load the planes or not
        self.render_volume()
        if self.planes == True:
            self.render_planes()

        # The end of the pipeline
        self.iren.Start()

    def render_volume(self):
        """ Render the volume. """

        # volume ray cast mapper
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputConnection(self.reader.GetOutputPort())
        volumeMapper.SetBlendModeToComposite()

        # color transfer function
        volumeCTF = vtk.vtkColorTransferFunction()
        volumeCTF.AddRGBPoint(0,    0.0, 0.0, 0.0)
        volumeCTF.AddRGBPoint(50,  1.0, 0.5, 0.3)

        # opacity transfer function
        volumeOTF = vtk.vtkPiecewiseFunction()
        volumeOTF.AddPoint(0,    0.00)
        volumeOTF.AddPoint(50,  0.50)
        volumeOTF.AddPoint(1000, 0.50)

        # global opacity transfer function
        volumeGOTF = vtk.vtkPiecewiseFunction()
        volumeGOTF.AddPoint(0,   0.0)
        volumeGOTF.AddPoint(90,  0.5)
        volumeGOTF.AddPoint(100, 1.0)

        # volume property
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(volumeCTF)
        volumeProperty.SetScalarOpacity(volumeOTF)
        volumeProperty.SetGradientOpacity(volumeGOTF)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.4)
        volumeProperty.SetDiffuse(0.6)
        volumeProperty.SetSpecular(0.2)

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        # visualise volume
        self.ren.AddViewProp(volume)
        self.ren.ResetCamera()

        # change start position orientation of 3D image
        self.ren.GetActiveCamera().Elevation(-180)
        self.ren.GetActiveCamera().Roll(70)
        self.ren.GetActiveCamera().Azimuth(-25)

        # camera movements and clipping planes
        self.ren.ResetCameraClippingRange()

        self.renWin.Render()
        self.renWin.SetWindowName('RW: volume')

        # when True save the image with a widget
        if self.save_img == True:
            axes = vtk.vtkAxesActor()
            widget = vtk.vtkOrientationMarkerWidget()
            widget.SetOutlineColor(0.9300, 0.5700, 0.1300);
            widget.SetOrientationMarker(axes);
            widget.SetInteractor(self.iren);
            widget.SetViewport(0.0, 0.0, 0.4, 0.4);
            widget.SetEnabled(1);
            widget.InteractiveOn();

            self.renWin.Render()

            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(self.renWin)
            windowToImageFilter.SetInputBufferTypeToRGBA()
            windowToImageFilter.ReadFrontBufferOff()
            windowToImageFilter.Update()

            # save the file
            text = str(input("Enter the name of the file: "))
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(self.path + '/' + text + ".png")
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            writer.Write()

        self.renWin.Render()
        self.iren.Initialize()
        # self.iren.Start() #complete pipeline


    def render_planes(self):
        """ Render the volume with 2D planes in sagittal, axial,
            and coronal view. """

        # create outline which provides context around the data
        outlineData = vtk.vtkOutlineFilter()
        outlineData.SetInputConnection(self.reader.GetOutputPort())
        outlineData.Update()

        mapOutline = vtk.vtkPolyDataMapper()
        mapOutline.SetInputConnection(outlineData.GetOutputPort())

        outline = vtk.vtkActor()
        colors = vtk.vtkNamedColors()
        outline.SetMapper(mapOutline)
        outline.GetProperty().SetColor(colors.GetColor3d("Black"))

        # create black/white lookup table
        bwLut = vtk.vtkLookupTable()
        bwLut.SetTableRange(0,2000)
        bwLut.SetSaturationRange(0,0)
        bwLut.SetHueRange(0,0)
        bwLut.SetValueRange(0,1)
        bwLut.Build()

        # create lookup table of full hue circle (HSV)
        hueLut = vtk.vtkLookupTable()
        hueLut.SetTableRange(0,2000)
        hueLut.SetSaturationRange(1,1)
        hueLut.SetHueRange(0,1)
        hueLut.SetValueRange(1,1)
        hueLut.Build()

        # create lookup table of single hue (having range in saturation of hue)
        satLut = vtk.vtkLookupTable()
        satLut.SetTableRange(0,2000)
        satLut.SetSaturationRange(0,1)
        satLut.SetHueRange(0.6,0.6)
        satLut.SetValueRange(1,1)
        satLut.Build()

        # create sagittal plane (1/3)
        sagittalColors = vtk.vtkImageMapToColors()
        sagittalColors.SetInputConnection(self.reader.GetOutputPort())
        sagittalColors.SetLookupTable(bwLut)
        sagittalColors.Update()

        sagittal = vtk.vtkImageActor()
        sagittal.GetMapper().SetInputConnection(sagittalColors.GetOutputPort())
        # sagittal.SetDisplayExtent(minX, maxX, minY, maxY, minZ, maxZ)
        sagittal.SetDisplayExtent(128,128, 0,255, 0,92)
        sagittal.ForceOpaqueOn()

        # create axial plane (2/3)
        axialColors = vtk.vtkImageMapToColors()
        axialColors.SetInputConnection(self.reader.GetOutputPort())
        axialColors.SetLookupTable(hueLut)
        axialColors.Update()

        axial = vtk.vtkImageActor()
        axial.GetMapper().SetInputConnection(axialColors.GetOutputPort())
        axial.SetDisplayExtent(0,255, 0,255, 46,46)
        axial.ForceOpaqueOn()

        # create coronal plane (3/3)
        coronalColors = vtk.vtkImageMapToColors()
        coronalColors.SetInputConnection(self.reader.GetOutputPort())
        coronalColors.SetLookupTable(satLut)
        coronalColors.Update()

        coronal = vtk.vtkImageActor()
        coronal.GetMapper().SetInputConnection(coronalColors.GetOutputPort())
        coronal.SetDisplayExtent(0,255, 128,128, 0,92)
        coronal.ForceOpaqueOn()

        # render planes
        self.ren.AddActor(outline)
        self.ren.AddActor(sagittal)
        self.ren.AddActor(axial)
        self.ren.AddActor(coronal)
