from DM_dummy.main import getDataFrame
import numpy as np
import pandas as pd

def getRangeInWindows(arrayOrg):
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


def getMinMax():
    df = getDataFrame()
    mealGroups = df.dropna(subset=['cgm_data']).groupby(['patient_number', 'meal_number']).apply(lambda group: getRangeInWindows(group.cgm_data.to_numpy()))
    # mealGroups = df.groupby(['patient_number', 'meal_number'])
    return mealGroups.reset_index().rename(columns={0: 'minMax'})

    # minMax_df = pd.DataFrame()
    # for mealGroupName in mealGroups.groups.keys():
    #     mealDf = mealGroups.get_group(mealGroupName)
    #     cgmData = mealDf.cgm_data.to_numpy()
    #     time_data = mealDf.time_data.to_numpy()
    #     windowSize = int(cgmData.shape[0] / 5)
    #     ranges = getRangeInWindows(cgmData)
    #
    #     if mealGroupName[0] == 2 and mealGroupName[1] == 0:
    #         print(ranges)


if __name__ == "__main__":
    getMinMax()