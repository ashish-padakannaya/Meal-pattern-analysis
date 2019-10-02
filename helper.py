from datetime import datetime, timedelta
from configparser import ConfigParser
import os
import ast
import pandas as pd
import numpy as np

def get_data_frame():
    """reads patient CGM CSVs and shapes it into a pandas dataframe

    Returns:
        Pandas.DataFrame -- dataframe with columns: patient_number, meal_number, cgm_data, time_data
    """
    
    #read config file and get patient data sources
    k = ConfigParser()
    k.read('config.ini')
    file_map = ast.literal_eval(k['FILES']['CGM_files'])
    directory = k['FILES']['data_directory']
    patient_df = pd.DataFrame()

    for patient_number, files in file_map.items():
        #read dataframes and convert each row to numpy arrays
        time_series_path = os.path.join(directory, files['time_series'])
        time_frame = pd.read_csv(time_series_path, na_filter=False)
        time_frame_array = time_frame.to_numpy()

        data_path = os.path.join(directory, files['data'])
        cgm_frame = pd.read_csv(data_path, na_filter=False)
        cgm_frame_array = cgm_frame.to_numpy()

        #zip functions joins each ith element of 2 arrays together:
        #zip([a1,a2],[b1,b2]) = [(a1,b1), (a2,b2)]
        #enumerate fetches index for each element in zip list.
        for index, (cgm_data,time_data) in enumerate(zip(cgm_frame_array, time_frame_array)):
            time_data = getFloatFromObject(time_data)
            cgm_data = getFloatFromObject(cgm_data)

            meal_data_frame = pd.DataFrame({
                'patient_number': int(patient_number[-1]),
                'meal_number': index,
                'cgm_data': cgm_data,
                'time_data': time_data
            })
            patient_df = patient_df.append(meal_data_frame)

    #convert timeseries to python datetime and save to CSV *check output
    patient_df['time_data'] = patient_df['time_data'].apply(lambda cell: convert_to_epoch(cell))
    patient_df.to_csv('test.csv', index=False)

    return patient_df

def convert_to_epoch(matlab_datenum):
    """converts shitty matlab time to Python datetime object
  
    Arguments:
        matlab_datenum {float} -- Matlab time with floating point
   
    Returns:
        datetime -- Python datetime object
    """
    try:
        return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days = 366)
    except Exception:
        return 'Nan'


def getFloatFromObject(array):
    arrayStr = array.astype(np.str)
    newArray = []
    for item in arrayStr:
        if item != '' : newArray.append(item)
        elif item == 'NaN' : newArray.append(np.nan)
        elif item == 'Nan' : newArray.append(np.nan)

    return np.array(newArray).astype(np.float)


def get_patient_df(all_dfs):
    final = all_dfs[0]
    for i in range(1, len(all_dfs)):
        final = final.merge(all_dfs[i], how='left', on=['patient_number', 'meal_number'])
    return final