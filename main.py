import pandas as pd
from configparser import ConfigParser
import ast
from helper import convert_to_epoch, getFloatFromObject, get_patient_df
from features import get_fft, get_sd, get_rms
from pca import pca_analysis
import os
import numpy as np
from tsfresh.feature_extraction.feature_calculators import standard_deviation

def getDataFrame():
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

all_dfs = []

patient_df = getDataFrame()

#retrieve FFT per person per meal
all_dfs.append(get_fft(patient_df))

# Code to retrieve STANDARD DEVIATION (SD) per person per meal
# Check patient_df_df for SD Dataframe
all_dfs.append(get_sd(patient_df))

# Code to retrieve ROOT MEAN SQUARE (RMS) per person per meal
# Check patient_rms_df for RMS Dataframe
all_dfs.append(get_rms(patient_df))

patient_features_df = get_patient_df(all_dfs)

pca_features_df = pca_analysis(patient_features_df)
patient_features_df.to_csv('patient_features.csv', index=False)

np.savetxt("pca_features.csv", pca_features_df, delimiter=",")
