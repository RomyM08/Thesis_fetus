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
# You can run this file to render multiple viewports.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import vtk


class Viewports():
    """ One render window, multiple viewports.
        This class defines how many viewport at the same time are visible
        for the user. Each viewport works independently of another. """

    def __init__(self, reader):

        # the reader which will be rendered as a volume
        self.reader = reader

        # the list of renderwindows
        self.iren_list = []

        # define viewport ranges
        self.xmins=[0,.5,0,.5]
        self.xmaxs=[0.5,1,0.5,1]
        self.ymins=[0,0,.5,.5]
        self.ymaxs=[0.5,0.5,1,1]

    def view(self, iren_list, PATH=None, save_img = False):
        """ The requirements for the visualisation. """

        self.iren_list = iren_list
        self.path = PATH
        self.save_img = save_img

        self.start_pipeline()
        self.iterate()
        self.end_pipeline()

    def start_pipeline(self):
        """ Start the pipeline. """

        # for the start of the visualisation
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

    def iterate(self):
        """ iterate over the viewports. """

        for i in range(len(self.iren_list)):
            # render the background
            self.ren = vtk.vtkRenderer()
            self.ren.SetBackground(0.2, 0.2, 0.2)
            self.ren.SetBackground2(0.5, 0.5, 0.5)
            self.ren.GradientBackgroundOn()

            # render
            self.renWin.AddRenderer(self.ren)
            self.renWin.SetSize(600, 600)
            self.ren.SetViewport(self.xmins[i],self.ymins[i],self.xmaxs[i],self.ymaxs[i])

            # show volume
            self.generate_renderer()

            # show text
            self.show_text(i)

    def generate_renderer(self):
        """ Generate the renderer. """

        # volume ray cast mapper
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputConnection(self.reader.GetOutputPort())
        volumeMapper.SetBlendModeToComposite()

        # color transfer function
        volumeCTF = vtk.vtkColorTransferFunction()
        volumeCTF.AddRGBPoint(0,   0.0, 0.0, 0.0)
        volumeCTF.AddRGBPoint(50,  1.0, 0.5, 0.3)

        # opacity transfer function
        volumeOTF = vtk.vtkPiecewiseFunction()
        volumeOTF.AddPoint(0,    0.00)
        volumeOTF.AddPoint(50,   0.50)
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

    def show_text(self, image):
        """ Show the text in the corner of the viewports. """
        textActor = vtk.vtkTextActor()
        textActor.SetInput(self.iren_list[image])
        textActor.SetPosition2(10,40)
        textActor.GetTextProperty().SetFontSize(24)
        self.ren.AddActor2D(textActor)

    def end_pipeline(self):
        """ The end of the pipeline. """

        self.renWin.Render()
        self.renWin.SetWindowName('RW: tool')

        # when True save the image with a widget
        if self.save_img == True:
            # apply widget to the volume image
            if "Volume" in self.iren_list:
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
        self.iren.Start()
