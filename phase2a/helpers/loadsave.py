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
# You can run this file to load and save data.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

"""
Phase 2a: The heuristic segmentation models:
- semi-automatic watershed segmentation
- fully automatic watershed segmentation
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
import pickle
import time
from tqdm import tqdm

sys.path.append('../phase1/modules/..')
from modules.calc_filters import *


""" Create directory. """
def create_dir(PATH):
    """ Create a directory. """
    try:
        os.mkdir(PATH)
        print('Directory', PATH, 'created' )
    except FileExistsError:
        print('Directory', PATH, 'already exists' )


""" Loading functions. """
def load_scans(pathDicom):
    """ Load the dicom files into sitk with the image series reader.
    Input: path of the directory with the dicom files.
    Output: the 3D image.
    """
    reader = sitk.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
    reader.SetFileNames(filenamesDICOM)
    img = reader.Execute()
    return img

def load_scans_filter(img_org, filterdata):
    """ Load and apply a specific filter of phase1 on the original image. """

    # check which filter will be used and apply that one
    filter = filterdata['filtername']
    if filter == 'gaussian':
        sigma = filterdata['parameters'][0]
        smoothed_img = calc_gaussian(img_org, sigma=sigma)
    elif filter == 'median':
        radius = filterdata['parameters'][0]
        smoothed_img = calc_median(img_org, radius=radius)
    elif filter == 'curvatureflow':
        iter = filterdata['parameters'][0]
        timestep = filterdata['parameters'][1]
        smoothed_img = calc_curvatureflow(img_org, iteration=iter, step=timestep)
    elif filter == 'anisodiff':
        iter = filterdata['parameters'][0]
        timestep = filterdata['parameters'][1]
        conductance = filterdata['parameters'][2]
        smoothed_img = calc_anisodiff(img_org, iteration=iter, step=timestep, conductance=conductance)
    else:
        print('The filtername does not exist.')

    return smoothed_img

def get_data_scans(rootdir, datasetnames, filterdata):
    """ Generate the dataset which includes the cropped, original images and
    the smoothed images with the best performed filter of the datasets.
    The input is the root directory to the folders with the corresponding
    dataset names and filterdata. The output is a dictionary with all these images.
    """
    datasets = {}

    print('Loading: ' + str(len(datasetnames)) + ' datasets')
    for dataset in tqdm(datasetnames):
        time.sleep(0.1)

        # Original images (to predict)
        images_org = load_scans(rootdir + dataset + '/crop_org')

        # Smoothed images by specific filter
        images_smoothed = load_scans_filter(images_org, filterdata)

        # Save images in datasets dictionary
        datasets.update({dataset : {'org': images_org, 'smoothed': images_smoothed}})

    print("datasets created")
    return datasets

def get_data_parascans(rootdir, datasetnames, filterdata):
    """ Generate the dataset which includes the cropped, original images and
    the smoothed images with the best performed filter of the datasets.
    The input is the root directory to the folders with the corresponding
    dataset names and filterdata. The output is a dictionary with all these images.
    """
    datasets = {}

    print('Loading: ' + str(len(datasetnames)) + ' datasets')
    for dataset in tqdm(datasetnames):
        time.sleep(0.1)

        # Original images (to predict)
        images_org = load_scans(rootdir + dataset + '/crop_org')

        # Ground truth images (mask image of expert)
        images_gt = load_scans(rootdir + dataset + '/crop_gt')
        images_gt = sitk.GetArrayFromImage(images_gt)

        # Smoothed images by specific filter
        images_smoothed = load_scans_filter(images_org, filterdata)

        # Save images in datasets dictionary
        datasets.update({dataset : {'org': images_org, 'gt': images_gt, 'smoothed': images_smoothed}})

    print("datasets created")
    return datasets

def get_heuristicnames(img_path):
    """ Get the unique filenames of the filters. """
    heuristicnames = []
    allfiles = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

    for files in allfiles:
        file = files.split('_', 1)
        if file[-1] not in heuristicnames:
            heuristicnames.append(file[1])
    for item, name in enumerate(heuristicnames):
        heuristicnames[item] = name[:-4]

    return heuristicnames

def load_data_pickle(PATH, dataset, filename):
    """ Load data from file using pickle. """
    with open(PATH + '/' + dataset + "_" + filename + ".pkl","rb") as f:
        new_data = pickle.load(f)

    # print(filename, "opened")
    return new_data

def get_data_heuristics(rootdir, img_path, datasetnames, heuristicnames):
    """ Generate the dataset which includes the ground truth, and
    all the heuristic model images.
    Output: dict {'gt', 'dict with heuristic model images'}
    """
    datasets = {}

    print('Loading: ' + str(len(datasetnames)) + ' datasets')
    for dataset in tqdm(datasetnames):
        time.sleep(0.1)

        # Ground truth images (mask image of expert)
        images_gt = load_scans(rootdir + dataset + '/crop_gt')
        images_gt = sitk.GetArrayFromImage(images_gt)

        # Heuristic model images (predictions of models)
        images_models = {}
        for model in heuristicnames:
            image = load_data_pickle(img_path, dataset=dataset, filename=model)
            images_models.update({model: image})

        # Save images in datasets dictionary
        datasets.update({dataset: {'gt':images_gt, 'models':images_models}})

    print("dataset created")
    return datasets


def load_metadata(PATH, filename):
    """ Load the metadata from pickle. """
    with open(PATH + '/' + filename + '_croporg' + ".pkl","rb") as f:
        new_data = pickle.load(f)

    print(filename, "opened")
    return new_data


""" Save functions. """
def save_data_pickle(PATH, data, dataset, filename):
    """ Save data in pickle file. """
    with open(PATH + '/' + dataset + "_" + filename + ".pkl","wb") as f:
        pickle.dump(data,f)
    print(filename, "created")

def save_dict_pickle(PATH, data, filename):
    """ Save data in pickle file. """
    with open(PATH + '/' + filename + ".pkl","wb") as f:
        pickle.dump(data,f)
    print(filename, "created")

def write_result(file, result):
    """ Write results for text file. """
    name = result[0]
    dataset = result[1]
    file.write(name + "\n")
    for data, values in dataset.items():
        file.write("%s %.3f %.3f \n" %(data, np.mean(values),np.std(values)))

def save_images(PATH, show_img, datasets, from_dataset):
    """ Show and save best images. """
    dataset = datasets[from_dataset]
    imgModels = dataset['models']
    for modelname, model in imgModels.items():
        print('save', modelname)
        plt.imshow(model[70])
        plt.set_cmap("gray")
        plt.axis('off')
        plt.savefig(PATH + '/' + from_dataset + '_' + modelname + '.png', dpi=400)

        if show_img == True:
            plt.show()

def save_results(PATH, data, filename):
    """ Save results in txt file. """
    with open(PATH + '/' + filename + ".txt","w") as file:
        file.write("Results of heuristic models with mean and standard deviation.\n")
        for result in data:
            write_result(file, result)
    file.close()
    print('results saved in:'+ PATH + '/' + filename + ".txt")
