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
# You can run this file to calculate the heuristic segmentation models, namely
# the semi-automatic watershed segmentation model, and the fully automatic
# watershed segmentation model.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 2a: The heuristic segmentation models:
- semi-automatic watershed segmentation
- fully automatic watershed segmentation
"""

import numpy as np
import random
import SimpleITK as sitk
import matplotlib.pyplot as plt

from helpers.showing import *

""" Semi-automatic watershed segmentation model. """
def get_labelvalues(img, seeds):
    """ Get the label values based on the given seed points. """
    array = sitk.GetArrayFromImage(img)
    newseeds = [list(s) for s in seeds]

    results = []
    for seed in newseeds:
        r = array[seed[0], seed[1], seed[2]]
        results.append(r)

    return results

def create_mask(images, keys):
    """ Create a binary image mask based on the key labels. """
    images = sitk.GetArrayFromImage(images)
    mask = np.zeros([images.shape[0], images.shape[1], images.shape[2]])

    for key in keys:
        # boolean whether image contains the key
        compare = images[:,:,:] == key
        mask[compare] = 1

    # reshape the image in correct format
    mask.reshape([images.shape[2], images.shape[1], images.shape[0]])

    return np.array(mask, np.uint8)

def define_seedpoints(key):
    """ Define the manual seed points corresponding to the dataset key. """
    seedpoints = {}

    if key == 'dataset1':
        seedpoints['component'] = (120,90,70)
        seedpoints['labels'] = [(70,35,40), (70, 50,90), (138,75,80),(144,80,60), (120,70,120)]
    elif key == 'dataset2':
        seedpoints['component'] = (100,70,70)
        seedpoints['labels'] = [(75, 40,50), (70, 40, 95), (75,15,70), (50,55,125)]
    elif key == 'dataset3':
        seedpoints['component'] = (130,70,80)
        seedpoints['labels'] = [(80,55,40), (80,60,90), (40,120,125), (50,110,90)]
    elif key == 'dataset4':
        seedpoints['component'] = (50,80,65)
        seedpoints['labels'] = [(65,40,40), (65,60,100), (50,60,60)]
    elif key == 'dataset5':
        seedpoints['component'] = (150,20,90)
        seedpoints['labels'] = [(90,60,50), (90,80,130), (110,90,120), (130, 95,100)]
    elif key == 'dataset6':
        seedpoints['component'] = (110,100,70)
        seedpoints['labels'] = [(70,40,40), (70,60,90), (90,75,115),(110,80,80)]
    elif key == 'dataset7':
        seedpoints['component'] = (30,80,90)
        seedpoints['labels'] = [(90,80,125), (90,50,60), (90,120,80),  (90,90,90), (100,110,125)]
    else:
        print('No seed points specified for this dataset.')

    return seedpoints

def calc_ws_semiauto(img, key, sigma=1.5, level1=4, level2=1, showing=False):
    """ Semi-automatic watershed with defined seed points for each specified dataset.
        The seed point defines which labels needs to be merged for the binary mask."""

    # get the manually selected seedpoints
    seedpoints = define_seedpoints(key)
    seed_component = seedpoints['component']
    seeds_labels = seedpoints['labels']

    # calculate the semi-automatic watershed segmentation
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(img, sigma=sigma)
    ws_img = sitk.MorphologicalWatershed(feature_img, level=level1, markWatershedLine=False, fullyConnected=False)
    seg = sitk.ConnectedComponent(ws_img!=ws_img[seed_component[0],seed_component[1],seed_component[2]])
    filled = sitk.BinaryFillhole(seg!=0)
    d = sitk.SignedMaurerDistanceMap(filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
    ws_img2 = sitk.MorphologicalWatershed( d, markWatershedLine=False, level=level2)
    ws = sitk.Mask(ws_img2, sitk.Cast(seg, ws_img2.GetPixelID()))

    # create the final binary mask based on label keys
    labels = get_labelvalues(ws, seeds_labels)
    result_ws_semi = create_mask(ws, keys=labels)

    if showing == True:
        image_size = sitk.GetArrayFromImage(img)
        z = round(image_size[2] / 2.)

        sitk_show(img[:,:,z])
        sitk_show(feature_img[:,:,z])
        sitk_show(sitk.LabelToRGB(ws_img[:,:,z]))
        sitk_show(sitk.LabelOverlay(img[:,:,z], seg[:,:,z]), seeds=[(120,90,70)])
        sitk_show(filled[:,:,z])
        sitk_show(d[:,:,z])
        sitk_show(sitk.LabelOverlay(img[:,:,z], ws_img2[:,:,z]))
        sitk_show(sitk.LabelOverlay(img[:,:,z], ws[:,:,z]))

        plt.imshow(result_ws_semi[z], cmap='gray')
        plt.title('new image of keys: ' + str(labels))
        plt.show()

    return result_ws_semi


""" Fully automatic watershed segmentation model. """
def calculate_values(image, center):
    """ Return the total number of pixel intensity of the centered pixel
    and the corresponding neighbor values divided by 9 (the amount of pixels). """
    # total number of pixel intensity
    values = 0

    values += image[center[1]-1][center[0]-1] # left top
    values += image[center[1]-1][center[0]]   # left middle
    values += image[center[1]-1][center[0]+1] # left bottom
    values += image[center[1]][center[0]-1]   # middle top
    values += image[center[1]][center[0]]     # center
    values += image[center[1]][center[0]+1]   # middle bottom
    values += image[center[1]+1][center[0]-1] # right top
    values += image[center[1]+1][center[0]]   # right middle
    values += image[center[1]+1][center[0]+1] # right bottom

    # the mean value intensity
    values = round(values / 9.)
    return values

def possible_seed(image, count):
    """ Generate a random possible seed.
        Which is not at a boundary of the image. """
    searchseed = True

    while searchseed:
        # make random simulation repeatable with counter
        random.seed(count * 3)

        # define x and y
        x = random.randint(0, image.shape[1]-1)
        y = random.randint(0, image.shape[0]-1)

        # check if seed is in boundary
        if (x != 0) and (y != 0) and (x != image.shape[1]-1) and (y != image.shape[0]-1):
            searchseed = False

    print('possible seed point', (x,y))
    return (x,y)

def compute_seed1(image):
    """ Compute the first seedpoint.
        Apply method of which the idea is based on the hit-or-miss
        Monte Carlo method.
        Pick randomly an pixel in the image.
        Check its neighbours with a kernel size of 3.
        When the total values < threshold: seed point is found. (Hit)
        Otherwise, continue searching. (Miss)
    """

    seedfound = False       # true when seedpoint is found
    value_threshold = 10    # the percentage of values including neighbors which needs to be black(ish)
    count = 0               # make random simulation repeatable with counter

    print('Start searching for seed point')
    while seedfound == False:
        center = possible_seed(image, count)

        # check the values of the center and neighbors
        values = calculate_values(image, center)

        if values < value_threshold:
            # seedpoint is found
            seedfound = True
            print('seedpoint', center)
        else:
            count += 1

    return center

def generate_seed1(img, metadata):
    """ Compute the first seed point for generating the connected component.
        Return this seed point and its neighbors. """

    # check shape of image
    size = img.GetSize()
    if size != metadata['ConstPixelDims']:
        raise ValueError("The size of the image is not the same as the metadata.")

    # take in the middle of the whole 3D image an 2D slice
    # to decrease the computational time
    # Pay attention: sitk (x,y,z) makes numpy actually (z,y,x)
    images = sitk.GetArrayFromImage(img)
    z = round(size[2] / 2.)
    slice = images[z,:,:]

    # make seedpoint
    seed = compute_seed1(slice)
    real_seed = (seed[0], seed[1], z)
    return real_seed

def get_labels_auto(image):
    """ Get all the existing labels for the fully automatic segmentation. """
    img = sitk.GetArrayFromImage(image)
    img = img.flatten()
    labels = []

    # traverse for all elements
    for x in img:
        # check if exists in unique list or not
        if x not in labels:
            labels.append(x)

    return sorted(labels)

def define_labels_auto(image, all_labels):
    """ Define the labels which are not in the boundaries of the image. """
    img = sitk.GetArrayFromImage(image)

    # create the boundaries
    boundaries = []
    boundaries += img[0].tolist() + img[-1].tolist()

    for el in img:
        for small in el:
            boundaries += [small[0]] + [small[-1]]
        boundaries += el[0].tolist() + el[-1].tolist()

    # merge the lists to one list
    merged_boundaries = []
    for bound in boundaries:
        if isinstance(bound, list):
            for x in bound:
                merged_boundaries.append(x)
        else:
            number = int(bound)
            merged_boundaries.append(number)

    # get the unique boundary labels
    bound_labels = sorted(list(set(merged_boundaries)))

    # compare label lists and return the labels which are not in boundary
    result = list(set(bound_labels)^set(all_labels))

    return result

def calc_ws_fullyauto(img, metadata, sigma=1.5, level1=4, level2=1, showing=False):
    """ Fully automatic watershed segmentation to create a binary mask."""

    # The 2D slice in the middle of the image
    image_size = sitk.GetArrayFromImage(img)
    dims = metadata['ConstPixelDims']
    z = round(dims[2] / 2.)

    # calculate watershed
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(img, sigma=sigma)
    ws_img = sitk.MorphologicalWatershed(feature_img, level=level1, markWatershedLine=False, fullyConnected=False)
    seed1 = generate_seed1(img, metadata)
    seg2 = sitk.ConnectedComponent(ws_img!=ws_img[seed1[0], seed1[1], seed1[2]])
    filled = sitk.BinaryFillhole(seg2!=0)
    d = sitk.SignedMaurerDistanceMap(filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
    ws_img2 = sitk.MorphologicalWatershed( d, markWatershedLine=False, level=level2)
    ws = sitk.Mask( ws_img2, sitk.Cast(seg2, ws_img2.GetPixelID()))
    labels = get_labels_auto(ws)
    use_labels = define_labels_auto(ws, labels)
    result_ws_auto = create_mask(ws, keys=use_labels)

    if showing == True:
        # A: origial image
        sitk_show(img[:,:,z])
        # B: gradient
        sitk_show(feature_img[:,:,z])
        # C: watsershed
        sitk_show(sitk.LabelToRGB(ws_img[:,:,z]))
        # D: seed point
        sitk_show(img[:,:,z], seeds=[(seed1[0], seed1[1], seed1[2])])
        # E: connected foreground components
        sitk_show(sitk.LabelOverlay(img[:,:,seed1[2]], seg2[:,:,seed1[2]]), seeds=[(seed1[0], seed1[1], seed1[2])])
        # F: binary fill hole
        sitk_show(filled[:,:,z])
        # G: distance map
        sitk_show(d[:,:,z])
        # H: watershed
        sitk_show(sitk.LabelOverlay(img[:,:,z], ws_img2[:,:,z]))
        # I: mask
        sitk_show(sitk.LabelOverlay(img[:,:,z], ws[:,:,z]))

        plt.imshow(result_ws_auto[z], cmap='gray')
        plt.title('new image of keys: ' + str(use_labels))
        plt.show()

    return result_ws_auto
