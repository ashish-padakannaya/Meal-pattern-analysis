import sys
import os
from helper import get_meal_vectors
from pathlib import Path
from dynaconf import settings
import pandas as pd
import numpy as np
import helper
from sklearn.externals import joblib

from features.features import generate_features 

if __name__ == "__main__":
    
    filename = sys.argv[1]
    meal_data_np = []
    print("loading file - " + filename)
    meal_data = pd.read_csv(os.path.join(filename), na_filter = False, header = None, sep = '\n')

    for i,_ in enumerate(meal_data.iterrows()):
            t = helper.getFloatFromObjectForMealData(meal_data.loc[i])
            if t.size != 0: 
                t = t[::-1]
                meal_data_np.append(t)                
    meal_data_np = np.array(meal_data_np)

    directory = Path(settings.path_for(settings.FILES.MODELS))
    directory = str(directory)
    model_dict = list(settings.CLASSIFIER.MODEL_DICT)

    classifier_preditions = pd.DataFrame()
    for classifier in model_dict:
        filename = classifier[1]
        model = joblib.load(os.path.join(directory, filename))
        meal_vectors, labels = get_meal_vectors(classifier[0],True,False,True)
        predictions = model.predict(meal_vectors)

        classifier_preditions[classifier[0]] = predictions
    
    # output_folder = Path(settings.path_for(settings.FILES.OUTPUT_DIRECTORY))
    output_file = 'classifier_predictions.csv' 
    classifier_preditions.to_csv(output_file, index=False)