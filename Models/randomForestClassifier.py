import sys
sys.path.append('../')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn.metrics
import timeit
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.externals import joblib 

from helper import get_meal_vectors

x, y = get_meal_vectors(True, True, False)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

kf = KFold(n_splits=10, shuffle=True)

clf = RandomForestClassifier(bootstrap=False, class_weight=None,
                                        criterion='gini', max_depth=None,
                                        max_features=0.05, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=5, min_samples_split=6,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=70, n_jobs=None,
                                        oob_score=False, random_state=23,
                                        verbose=0, warm_start=False)

scores = []

for train_index, test_index in kf.split(x):
    train_data, test_data = x[train_index], x[test_index]
    train_labels, test_labels = y[train_index], y[test_index]

    clf.fit(train_data,train_labels)
    y_pred=clf.predict(test_data)

    scores.append(f1_score(test_labels, y_pred))


print("mean scores - " + str(np.mean(scores)))

saved_model = joblib.dump(clf, 'randomForestClassifier.pkl')

# tpot = TPOTClassifier(verbosity=3, 
#                       scoring="balanced_accuracy", 
#                       random_state=23, 
#                       periodic_checkpoint_folder="tpot_mnst1.txt", 
#                       n_jobs=-1, 
#                       generations=10, 
#                       population_size=100)
# # run three iterations and time them
# times = []
# scores = []
# winning_pipes = []

# for x in range(3):
#     start_time = timeit.default_timer()
#     tpot.fit(X_train, y_train)
#     elapsed = timeit.default_timer() - start_time
#     times.append(elapsed)
#     winning_pipes.append(tpot.fitted_pipeline_)
#     scores.append(tpot.score(X_test, y_test))
#     tpot.export('tpot_mnist_pipeline.py')
# times = [time/60 for time in times]
# print('Times:', times)
# print('Scores:', scores)   
# print('Winning pipelines:', winning_pipes)


