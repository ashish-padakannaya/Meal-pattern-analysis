from helper import get_meal_vectors
from enums.model import ModelType
from svm import trainSVM

if __name__ == "__main__":

    modelType = ModelType.SVM
    
    #get patient dataframe from CSV
    data, labels = get_meal_vectors()

    if(modelType == ModelType.SVM):
        trainSVM(data, labels)
