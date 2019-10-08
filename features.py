import pandas as pd
from tsfresh.feature_extraction.feature_calculators import standard_deviation, fft_aggregated,  longest_strike_above_mean, linear_trend,  count_above_mean, count_below_mean
import numpy as np

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

def return_linear_trend_array(arr):
    """generates a list of linear trend array value
    
    Arguments:
        arr {Pandas.Series} -- Pandas group by series to run computation over
    
    Returns:
        list -- list of linear trend values [pvalue, rvalue, slope, intercept, stderr]
    """
    params = [
        {'attr': 'pvalue'},
        {'attr': 'rvalue'},
        {'attr': 'slope'},
        {'attr': 'intercept'},
        {'attr': 'stderr'}
    ] 
    return [item[1] for item in linear_trend(arr, params)]


def get_lt(df):
    """generates a dataframe with linear trend value columsn
    
    Arguments:
        df {Pandas.DataFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with linear trend values for each patient, meal combination
    """
    df.dropna(subset=['cgm_data'], inplace=True)
    groups_lt = df.groupby(['patient_number', 'meal_number']).apply(lambda x: return_linear_trend_array(x.cgm_data))
    temp = groups_lt.reset_index()[0].apply(pd.Series)
    temp.rename(columns={0: 'lt_pvalue', 1: 'lt_rvalue', 2: 'lt_slope', 3: 'lt_intercept', 4: 'lt_stderr'}, inplace=True)
    groups_lt = pd.concat([groups_lt.reset_index(), temp], axis=1)
    del groups_lt[0]
    return groups_lt

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

def get_min_max(df):
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


def get_rms(df):
    """get root mean square of each patient meal combo
    
    Arguments:
        df {Pandas.DataFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with rms column for each patient_number, meal_number combination
    """
    df.dropna(subset=['cgm_data'],inplace=True)
    groups_rms = df.groupby(['patient_number','meal_number']).apply(lambda x: np.sqrt(np.mean((x.cgm_data)**2)))
    groups_rms = groups_rms.reset_index()
    groups_rms.rename(columns={0:'rms'}, inplace=True)
    return groups_rms

def get_lsam(df):
    """get longest strike above mean of each patient meal combo
    
    Arguments:
        df {Pandas.DataFrame} -- patient dataframe
    
    Returns:
        Pandas.DataFrame -- dataframe with lsam column for each patient_number, meal_number combination
    """

    df.dropna(subset=['cgm_data'],inplace=True)
    groups_lsam = df.groupby(['patient_number','meal_number']).apply(lambda x: longest_strike_above_mean(x.cgm_data))
    groups_lsam = groups_lsam.reset_index()
    groups_lsam.rename(columns={0:'lsam'}, inplace=True)
    return groups_lsam


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
