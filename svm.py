import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB

def getAccuracy(svmModel, testData, testGt):
    preds = svmModel.predict(testData)
    return f1_score(testGt, preds), accuracy_score(testGt, preds), precision_score(testGt, preds), recall_score(testGt, preds)

def trainSVM(trainingData, labels):
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(trainingData)

    counter = 0
    f1List = []
    accuracyList = []
    presList = []
    recallList = []
    for _ in range(1):
        for train_index, test_index in kf.split(trainingData):
            counter += 1

            svclassifier = SVC(kernel='linear')
            svclassifier.fit(trainingData[train_index], labels[train_index])
            # gnb = GaussianNB()
            # gnb.fit(trainingData[train_index], labels[train_index])
            f1Score, acc, pres, rec = getAccuracy(svclassifier, trainingData[test_index], labels[test_index])
            f1List.append(f1Score)
            accuracyList.append(acc)
            presList.append(pres)
            recallList.append(rec)

    print("-------------------------------------------------------------------")
    print("Accuracy: {}".format(sum(accuracyList)/len(accuracyList)))
    print("F1: {}".format(sum(f1List) / len(f1List)))
    print("Precision: {}".format(sum(presList) / len(presList)))
    print("Recall: {}".format(sum(recallList) / len(recallList)))
