import pandas as pd
from tsfresh.feature_extraction.feature_calculators import standard_deviation, fft_aggregated,  longest_strike_above_mean, linear_trend,  count_above_mean, count_below_mean
import numpy as np
from configparser import ConfigParser
from ast import literal_eval

# import sys
# sys.path.append('../')
from features.get_pca import get_pca_vectors

def return_fft_array(series):
    """generates FFT aggregate metrics like centroid, variance, skew, kurtosis
    
    Arguments:
        arr {pandas.Series} -- series of cgm_data for each group of combination patient_number and meal_number  
    
    Returns:
        list -- list of values for features in the order centroid, variance, skew, kurtosis
    """
    attrs = [
        {'aggtype': 'centroid'},
        {'aggtype': 'variance'},
        {'aggtype': 'skew'},
        {'aggtype': 'kurtosis'},
    ]
    fft_zip = fft_aggregated(series, attrs)
    return [item[1] for item in list(fft_zip)]

def get_fft(df):
    """returns dataframe with fft values for per patient per meal. 

    *drop rows where cgm data is null
    *group by patient_number, meal_number and calculate fft values for each grouped data
    *after reset_index() we get column 0 which will have the list of FFT features: [centroid, variance, skew, kurtosis]
    *call series apply on the column to split the list of 4 features into 4 columns : 0, 1, 2, 3, 4 and then rename the columns
    *concat with axis = 1 joins 2 dataframes horizontally and delete the original column 0

    Arguments:
        df {pandas.DataFrame} -- dataframe shaped after ingesting and cleaning all CGM data
    
    Returns:
        pandas.DataFrame -- dataframe with FFT values for each patient number and meal number combination
    """

    df.dropna(subset=['cgm_data'],inplace=True)
    groups_fft = df.groupby(['patient_number','meal_number']).apply(lambda x: return_fft_array(x.cgm_data))
    temp = groups_fft.reset_index()[0].apply(pd.Series)
    temp.rename(columns={0:'fft_centroid', 1:'fft_variance', 2:'fft_skew', 3:'fft_curtosis'}, inplace=True)
    groups_fft = pd.concat([groups_fft.reset_index(), temp], axis=1)
    del groups_fft[0]

    return groups_fft

def get_range_in_windows(arrayOrg):
    """get ranges for dataframe to perform min max computation
    
    Arguments:
        arrayOrg {Pandas.Series} -- group chunk to get min max on
    
    Returns:
        float -- max value across windows
    """
    windowSize = int(arrayOrg.shape[0] / 5)
    arrayLength = arrayOrg.shape[0]
    lastValue = arrayOrg[arrayLength - 1]
    arrayPadded = np.pad(arrayOrg, ((0, windowSize - 1)), mode='constant', constant_values=lastValue)
    rangeArray = []
    for index in range(arrayLength):
        window = arrayPadded[index: index + windowSize]
        minMinusMax = np.max(window) - np.min(window)
        rangeArray.append(minMinusMax)

    np_max = np.max(np.array(rangeArray))
    return np_max

def get_min_max(arr):
    """gets min max of dataframe in windows
    
    Arguments:
        df {Pandas.DataFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with min max column for each patient_number, meal_number combination
    """
    mealGroups = df.dropna(subset=['cgm_data']).groupby(['patient_number', 'meal_number']).apply(lambda group: get_range_in_windows(group.cgm_data.to_numpy()))
    return mealGroups.reset_index().rename(columns={0: 'min_max'})


def get_sd(df):
    """get standard deviation of each patient meal combo
    
    Arguments:
        df {Pandas.DatFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with standard deviation column for each patient_number, meal_number combination
    """         
    df.dropna(subset=['cgm_data'],inplace=True)
    groups_sd = df.groupby(['patient_number','meal_number']).apply(lambda x: standard_deviation(x.cgm_data))
    groups_sd = groups_sd.reset_index()
    groups_sd.rename(columns={0:'sd'}, inplace=True)
    return groups_sd

def get_cam(df):
    """get count of meals above mean of each patient meal combo
    
    Arguments:
        df {Pandas.DataFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with cam column for each patient_number, meal_number combination
    """
    df.dropna(subset=['cgm_data'],inplace=True)
    groups_cam = df.groupby(['patient_number','meal_number']).apply(lambda x: count_above_mean(x.cgm_data))
    groups_cam = groups_cam.reset_index()
    groups_cam.rename(columns={0:'cam'}, inplace=True)
    return groups_cam

def get_cbm(df):
    """get count of meals below mean of each patient meal combo
    
    Arguments:
        df {Pandas.DataFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with cam column for each patient_number, meal_number combination
    """
    df.dropna(subset=['cgm_data'],inplace=True)
    groups_cbm = df.groupby(['patient_number','meal_number']).apply(lambda x: count_below_mean(x.cgm_data))
    groups_cbm = groups_cbm.reset_index()
    groups_cbm.rename(columns={0:'cbm'}, inplace=True)
    return groups_cbm


#REFACTORED FEATURE VECTOR CODE
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def get_rms(arr):
    """get root mean square of a single row vector
    
    Arguments:
        arr {np.array} -- 1D row array
    
    Returns:
        np.array -- 1 element numpy array with RMS value
    """
    return np.array([np.sqrt(np.mean((arr)**2))])


def get_lsam(arr):
    """get longest strike above mean of a single row vector
    
    Arguments:
        arr {np.array} -- 1D row array
    
    Returns:
        np.array -- 1 element numpy array with LSAM value
    """
    return np.array([longest_strike_above_mean(arr)])

def get_lt(arr):
    """get linear trend attributes of a single row vector
    
    Arguments:
        arr {np.array} -- 1D row array
    
    Returns:
        np.array -- 4 element numpy array with linear trend values
    """
    params = [
        {'attr': 'pvalue'},
        {'attr': 'rvalue'},
        {'attr': 'slope'},
        {'attr': 'stderr'}
    ] 
    return np.array([item[1] for item in linear_trend(arr, params)])


def get_feature_func(feature_name):
    """Dynamically maps feature name to functions
    
    Arguments:
        feature_name {str} -- name of feture
    
    Returns:
        function -- corresponding feature function
    """
    feature_to_function_map = {
        'linear_trend': get_lt,
        'rms': get_rms,
        'min_max': get_min_max,
        'lsam': get_lsam
    }
    return feature_to_function_map[feature_name]


def generate_features(meal_array, apply_pca=True):
    """generates features from meal array data
    
    Arguments:
        meal_array {np.array} -- 2D numpy array of meal and no meal data
    
    Keyword Arguments:
        apply_pca {bool} -- set to False if data should not be reduced (default: {True})
    
    Returns:
        np.array -- 2D numpy array of meal data features
    """
    config = ConfigParser()
    config.read('config.ini')
    k = int(config['FEATURES']['k'])
    feature_array = np.array([])

    for feature in literal_eval(config['FEATURES']['features']):
        res = map(get_feature_func(feature), meal_array)
        res = np.array(list(res))

        if not feature_array.size: feature_array = res
        else: feature_array = np.concatenate((feature_array, res), axis=1)

    if apply_pca:
        feature_array = get_pca_vectors(feature_array, k)

    return feature_array