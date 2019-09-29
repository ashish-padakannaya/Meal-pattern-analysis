from datetime import datetime, timedelta
import numpy as np

def convert_to_epoch(matlab_datenum):
    """converts shitty matlab time to Python datetime object
    
    Arguments:
        matlab_datenum {float} -- Matlab time with floating point
    
    Returns:
        datetime -- Python datetime object 
    """
    try:
        return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    except Exception:
        return 'Nan'
        # return np.nan

def getFloatFromObject(array):
    arrayStr = array.astype(np.str)
    newArray = []
    for item in arrayStr:
        if item != '': newArray.append(item)
        elif item == 'NaN': newArray.append(np.nan)
        elif item == 'Nan': newArray.append(np.nan)

    return np.array(newArray).astype(np.float)
