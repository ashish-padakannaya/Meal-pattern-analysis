import pandas as pd
from tsfresh.feature_extraction.feature_calculators import standard_deviation, fft_aggregated

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
    """returns dataframe with fft values for per patient per meal
    
    Arguments:
        df {pandas.DataFrame} -- dataframe shaped after ingesting and cleaning all CGM data
    
    Returns:
        pandas.DataFrame -- dataframe with FFT values for each patient number and meal number combination
    """
    #drop rows where cgm_data is null group by patient_number, meal_number.
    #after grouping, generate FFT for the data in each group
    df.dropna(subset=['cgm_data'],inplace=True)
    groups_fft = df.groupby(['patient_number','meal_number']).apply(lambda x: return_fft_array(x.cgm_data))

    #after reset_index() we get column 0 which will have the list of FFT features: [centroid, variance, skew, kurtosis]
    #call series apply on the column to split the list of 4 features into 4 columns : 0, 1, 2, 3, 4 and then rename the columns
    temp = groups_fft.reset_index()[0].apply(pd.Series)
    temp.rename(columns={0:'fft_centroid',1:'fft_variance', 2:'fft_skew',3:'fft_curtosis'},inplace=True)

    #concat with axis = 1 joins 2 dataframes horizontally and delete the original column 0
    groups_fft = pd.concat([groups_fft.reset_index(), temp], axis=1)
    del groups_fft[0]

    return groups_fft