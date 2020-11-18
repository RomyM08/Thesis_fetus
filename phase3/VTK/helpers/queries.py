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
# You can run this file to define the queries/questions and return an answer.
#
# Made by Romy Meester
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


"""
Phase 3: The volume visualisations.
"""

import sys


def query_function_visualisation(question, default='volume'):
    """ What do you want to visualise? """

    prompt = " [volume/mask/tool -- vo/ma/to]"
    volume = {'volume','vo', 'v', ''}
    mask = {'mask', 'ma', 'm'}
    tool = {'tool', 'to', 't'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in volume:
            return 'volume'
        elif choice in mask:
            return 'mask'
        elif choice in tool:
            return 'tool'
        else:
            sys.stdout.write("Please respond with 'volume', 'mask', or 'tool' ")

def query_function_dataset(question, default='dataset1'):
    """ Which dataset do you want to visualise? """

    prompt = " [dataset1/dataset2/ ... /dataset7 -- 1/2/3/4/5/6/7]"
    dataset1 = {'dataset1','1', ''}
    dataset2 = {'dataset2', '2'}
    dataset3 = {'dataset3', '3'}
    dataset4 = {'dataset4', '4'}
    dataset5 = {'dataset5', '5'}
    dataset6 = {'dataset6', '6'}
    dataset7 = {'dataset7', '7'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in dataset1:
            return 'dataset1'
        elif choice in dataset2:
            return 'dataset2'
        elif choice in dataset3:
            return 'dataset3'
        elif choice in dataset4:
            return 'dataset4'
        elif choice in dataset5:
            return 'dataset5'
        elif choice in dataset6:
            return 'dataset6'
        elif choice in dataset7:
            return 'dataset7'
        else:
            sys.stdout.write("Please respond with '1', '2', '3', ..., '7' ")

def query_function_volume_image(question, default='crop_org'):
    """ Which image do you want to visualise? """

    prompt = " [real_org/crop_org/crop_smoothed -- 1/2/3]"
    real_org = {'real_org','1', ''}
    crop_org = {'crop_org','2', ''}
    crop_smoothed = {'crop_smoothed','3'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in real_org:
            return 'real_org'
        elif choice in crop_org:
            return 'crop_org'
        elif choice in crop_smoothed:
            return 'crop_smoothed'
        else:
            sys.stdout.write("Please respond with '1', '2' or '3' ")

def query_function_image(question, default='org'):
    """ Of which input image do you want to apply the mask? """

    prompt = " [original/smoothed -- 1/2]"
    original = {'original','1', ''}
    smoothed = {'smoothed','2'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in original:
            return 'org'
        elif choice in smoothed:
            return 'smoothed'
        else:
            sys.stdout.write("Please respond with '1' or '2' ")

def query_function_model(question, default='ground_truth'):
    """ Of which model do you want to apply the mask? """

    prompt = " [ground truth/heuristic model/unet -- 1/2/3]"
    groundtruth = {'ground_truth','1', ''}
    heuristic = {'heuristic', '2'}
    unet = {'unet', '3'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in groundtruth:
            return 'ground_truth'
        elif choice in heuristic:
            return 'heuristic'
        elif choice in unet:
            return 'unet'
        else:
            sys.stdout.write("Please respond with '1', '2', or '3' ")

def query_function_heuristic(question, default='ws_semiauto'):
    """ Which heuristic segmentation model do you want? """

    prompt = " [semi-automatc/fully automatic -- 1/2]"
    semi = {'semi','1', ''}
    fully = {'fully','2'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in semi:
            return 'ws_semiauto'
        elif choice in fully:
            return 'ws_fullyauto'
        else:
            sys.stdout.write("Please respond with '1' or '2' ")

def query_function_activation(question, default='elu'):
    """ Which activation function do you want? """

    prompt = " [relu/elu -- 1/2]"
    relu = {'relu','1', ''}
    elu = {'elu','2'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in relu:
            return 'relu'
        elif choice in elu:
            return 'elu'
        else:
            sys.stdout.write("Please respond with '1' or '2' ")

def query_function_simulation(question, default='1'):
    """ Which simulation do you want? """

    prompt = " [number -- 1/2/.../6]"
    one = {'1', ''}
    two = {'2'}
    three = {'3'}
    four = {'4'}
    five = {'5'}
    six = {'6'}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in one:
            return '1'
        elif choice in two:
            return '2'
        elif choice in three:
            return '3'
        elif choice in four:
            return '4'
        elif choice in five:
            return '5'
        elif choice in six:
            return '6'
        else:
            sys.stdout.write("Please respond with '1' until '6' ")

def query_function_saving(question, default='no'):
    """ Do you want to save the visualisation? """

    prompt = " [yes/no -- y/n]"
    yes = {'yes', 'y'}
    no = {'no', 'n', ''}

    while True:
        sys.stdout.write(question + prompt)
        # input returns the empty string for "enter"
        choice = input().lower()
        if default is not None and choice == '':
            return False
        elif choice in yes:
            return True
        elif choice in no:
            return False
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' ")
