import pandas as pd
from helper import convert_to_epoch, getFloatFromObject, get_data_frame, get_patient_df
from features import get_fft, get_sd, get_rms, get_min_max, get_lsam
from pca import pca_analysis
import os
import numpy as np
from tsfresh.feature_extraction.feature_calculators import standard_deviation


if __name__ == "__main__":
    
    #get patient dataframe from CSV
    patient_df = get_data_frame()
    
    #generate all features and join 
    all_feature_dfs = []
    all_feature_dfs.append(get_fft(patient_df))
    all_feature_dfs.append(get_sd(patient_df))
    all_feature_dfs.append(get_rms(patient_df))
    all_feature_dfs.append(get_min_max(patient_df))
    all_feature_dfs.append(get_lsam(patient_df))
    patient_features_df = get_patient_df(all_feature_dfs)

    #ouput CSVs for original features and PCA features
    pca_features_df = pca_analysis(patient_features_df)
    patient_features_df.to_csv('patient_features.csv', index=False)
    np.savetxt("pca_features.csv", pca_features_df, delimiter=",")
