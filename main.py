import pandas as pd
from configparser import ConfigParser
import ast
from helper import convert_to_epoch, getFloatFromObject
from features import get_fft, get_sd, get_rms
import os
import numpy as np
from tsfresh.feature_extraction.feature_calculators import standard_deviation

def getDataFrame():
    #read config file and get patient data sources
    k = ConfigParser()
    k.read('config.ini')
    file_map = ast.literal_eval(k['FILES']['CGM_files'])
    directory = k['FILES']['data_directory']
    patientMealCountMap = {}
    patient_df = pd.DataFrame()
    for patient_number, files in file_map.items():
        #read dataframes and convert each row to numpy arrays
        time_series_path = os.path.join(directory, files['time_series'])
        time_frame = pd.read_csv(time_series_path, na_filter=False)
        data_path = os.path.join(directory, files['data'])
        cgm_frame = pd.read_csv(data_path, na_filter=False)
        time_frame_array = time_frame.to_numpy()
        cgm_frame_array = cgm_frame.to_numpy()
        patientMealCountMap[patient_number] = time_frame_array.shape[0]

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

patient_df = getDataFrame()
get_fft(patient_df).to_csv('fft.csv', index=False)

# Code to retrieve STANDARD DEVIATION (SD) per person per meal
# Check patient_df_df for SD Dataframe
patient_sd_df = get_sd(patient_df)

# Code to retrieve ROOT MEAN SQUARE (RMS) per person per meal
# Check patient_rms_df for RMS Dataframe
patient_rms_df = get_rms(patient_df)
