from datetime import datetime, timedelta
from configparser import ConfigParser
import os
import ast
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
from features.features import generate_features

def get_meal_array():
    """
    To convert meal data from CSVs in MealDataFolder to a numpy array and a class label array
    Returns:
        tuple of the form (numpy array of all data, numpy array of class labels)
    """
    k = ConfigParser()
    k.read('config.ini')
    directory = k['FILES']['meal_data_directory']
    meal_data_np = []
    class_labels_np = []

    for meal_data_file in os.listdir(directory):
        print("loading file - " + meal_data_file)
        class_label = 0 if 'Nomeal' in meal_data_file else 1
        meal_data = pd.read_csv(os.path.join(directory, meal_data_file), na_filter = False, header = None, sep = '\n')

        for i,_ in enumerate(meal_data.iterrows()):
            t = getFloatFromObjectForMealData(meal_data.loc[i])
            if t.size != 0: 
                meal_data_np.append(t)
                class_labels_np.append(class_label)
        
    meal_data_np = np.array(meal_data_np)
    class_labels_np = np.array(class_labels_np)

    return meal_data_np, class_labels_np


def get_meal_vectors(apply_pca=True):
    data, labels = get_meal_array()
    data = generate_features(data, apply_pca=apply_pca)
    return data, labels


def getFloatFromObjectForMealData(array):
    arrayStr = list(array)[0].split(",")
    newArray = []
    for item in arrayStr:
        if item in ['None', 'Nan', 'Nan', '']:
            newArray.append(np.nan)
        else: newArray.append(item)
    res = np.array(newArray).astype(np.float)
    return res[~np.isnan(res)]

