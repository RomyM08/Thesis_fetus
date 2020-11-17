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
# You can run this file to load and save data. 
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 1: The noise reduction filters.
"""

import os
import numpy as np
import SimpleITK as sitk
import pickle
import time
from tqdm import tqdm

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

def get_data_scans(rootdir, datasetnames):
    """ Generate the dataset which includes the cropped, original images of
    the datasets. The input is the root directory to the folders with the
    corresponding dataset names. The output is a dictionary with all these images.
    """
    datasets = {}

    print('Loading: ' + str(len(datasetnames)) + ' datasets')
    for dataset in tqdm(datasetnames):
        time.sleep(0.1)

        # Original images (to predict)
        images_org = load_scans(rootdir + dataset + '/crop_org')

        # Save images in datasets dictionary
        datasets.update({dataset : {'org': images_org}})

    print("datasets created")
    return datasets

def get_filternames(img_path):
    """ Get the unique filenames of the filters. """
    filternames = []
    allfiles = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

    for files in allfiles:
        file = files.split('_', 1)
        if file[-1] not in filternames:
            filternames.append(file[1])
    for item, name in enumerate(filternames):
        filternames[item] = name[:-4]

    return filternames

def load_data_pickle(PATH, dataset, filename):
    """ Load data from file using pickle. """
    with open(PATH + '/' + dataset + "_" + filename + ".pkl","rb") as f:
        new_data = pickle.load(f)
    return new_data

def get_data_filters(rootdir, img_path, datasetnames, filternames):
    """ Generate the dataset which includes the original, ground truth, and
    all the image filters.
    """
    datasets = {}

    print('Loading: ' + str(len(datasetnames)) + ' datasets')
    for dataset in tqdm(datasetnames):
        time.sleep(0.1)

        # Original images (to predict)
        images_org = load_scans(rootdir + dataset + '/crop_org')
        images_org = sitk.GetArrayFromImage(images_org)

        # Ground truth images (mask image of expert)
        images_gt = load_scans(rootdir + dataset + '/crop_gt')
        images_gt = sitk.GetArrayFromImage(images_gt)

        # Filter images
        images_filters = {}
        for filter in filternames:
            image = load_data_pickle(img_path, dataset=dataset, filename=filter)
            images_filters.update({filter: image})

        # Save images in datasets dictionary
        datasets.update({dataset: {'org': images_org, 'gt': images_gt, 'filters':images_filters}})

    print("dataset created")
    return datasets


""" Save functions. """
def save_data_pickle(data, PATH, dataset, filename):
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

def save_results(PATH, data, filename):
    """ Save results in txt file. """
    with open(PATH + '/' + filename + ".txt","w") as file:
        file.write("Results of filters with mean and standard deviation.\n")
        for result in data:
            write_result(file, result)
    file.close()
    print('results saved in:'+ PATH + '/' + filename + ".txt")
