# to extract the top 5 features from the patient_features_df by applying PCA over it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from configparser import ConfigParser
import ast

def pca_analysis(patient_features_df):
    '''
    To get the top 5 features after applying PCA over the feature matrix
    '''
    k = ConfigParser()
    k.read('config.ini')
    '''
    reading the names of all the features to name the components matrix for PCA.
    If you add any new feature to the code, make sure you append the feature name in the config.ini file also
    '''
    feature_dict = ast.literal_eval(k['FEATURES']['feature_dict'])

    df = patient_features_df.dropna()
    sc = StandardScaler() 
    df = df.iloc[:, 2:].values
    df_normalized = sc.fit_transform(df)
    feature_cols = [feature_dict[i] for i in range(df_normalized.shape[1])]
    normalized_features = pd.DataFrame(df_normalized, columns = feature_cols)
    
    pca_func = PCA(n_components = 5)
    pca_features = pca_func.fit_transform(normalized_features)
    explained_variance_ratio = list(pca_func.explained_variance_ratio_)

    with open('Outputs/explained_variance_ratio.txt', 'w') as f:
        for item in explained_variance_ratio:
            f.write("%s\n" % item)

    pca_components = pd.DataFrame(pca_func.components_,columns=normalized_features.columns,index = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5'])
    pca_components.to_csv('Outputs/pca_components.csv', index=True)

    return pca_features

    