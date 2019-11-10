import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def getAccuracy(svmModel, testData, testGt):
    correctlyClassified  = 0
    preds = svmModel.predict(testData)

    # print(preds)
    for pred, gt in zip(preds, testGt):
        if pred == gt: correctlyClassified += 1

    return correctlyClassified/len(testData)

def trainSVM(trainingData, labels):
    kf = KFold(n_splits=2)
    kf.get_n_splits(trainingData)

    counter = 0
    svclassifier = SVC(kernel='linear')
    for _ in range(1):
        for train_index, test_index in kf.split(trainingData):
            counter += 1
            svclassifier.fit(trainingData[train_index], labels[train_index])
            getAccuracy(svclassifier, trainingData[test_index], labels[test_index])
            # print("{} {}".format(len(trainingData[test_index]), len(labels[test_index])))
            print("Counter: {} | Accuracy: {}".format(counter, getAccuracy(svclassifier, trainingData[test_index], labels[test_index])))





