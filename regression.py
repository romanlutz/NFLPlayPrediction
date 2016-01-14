from __future__ import division
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from collections import Counter
from random import random
from evaluate import reg_evaluate
from feat import get_team_features
from sklearn.externals import joblib

'''
estimator = the SVM you wish to use to classify the data
features = a (sample size) x (features) array containing the feature vectors of the data
true_labels = an array of the correct classification for each of the sample feature vectors
kfold = the number of folds to use in the cross validation. Defaults to 5 fold if not specified.

Returns (mean, standard deviation) of the provided estimator's accuracy using kfold validation,
and utilizes all available CPUs for the training and validation.
'''


def compute_regression_results(features, goals, outputfile):
    vec = DictVectorizer()
    vector = vec.fit_transform(features).toarray()
    targets = np.asarray(goals)
    # Consider replacing the above with FeatureHasher for faster computation?
   
    output = open(outputfile, 'w+')
	
    linreg = LinearRegression(normalize=True)
    abs_dif, mse_diff, avg_diff, avg_mse_diff = reg_evaluate(linreg, vector, targets)
    print >> output, "**********************************"
    print >> output, "LinearRegression"
    print >> output, "Average Difference from Goal: ", avg_diff
    print >> output, "Average MSE from Goal: ", avg_mse_diff
    print >> output, "**********************************"
    output.flush()
    joblib.dump(linreg, outputfile + 'linreg.pkl')
	
    rbfsvr = SVR(C = 128, kernel='rbf', gamma=pow(2,-17))
    abs_dif, mse_diff, avg_diff, avg_mse_diff = reg_evaluate(rbfsvr, vector, targets)
    print >> output, "**********************************"
    print >> output, "RBF SVR, C=2048, Gamma= 2^-17"
    print >> output, "Average Difference from Goal: ", avg_diff
    print >> output, "Average MSE from Goal: ", avg_mse_diff
    print >> output, "**********************************"
    output.flush()
    joblib.dump(rbfsvr, outputfile + 'rbfsvr.pkl')
    
#    print >> output, "**********************************"
#    print >> output, "Searching for SVR RBF Parameters"
	
#    rbfparameters = {
#         'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
#         'gamma': [pow(2, x) for x in range(-17, -13, 2)]  # Possible gamma values for the SVM.
#    }
#    search = GridSearchCV(SVR(), rbfparameters, cv=5, n_jobs=-1, verbose=1, scoring="mean_squared_error")
#    search.fit(vector, targets)
#    print >> output, "RBF SVR Best Estimator:"
#    print >> output, search.best_estimator_
#    print >> output, "Best Parameters: ", search.best_params_
#    print >> output, "Mean-Squared-Error Score: ", search.best_score_
#    print >> output, "Grid Scores:"
#    print >> output, search.grid_scores_
#    output.flush()
	
    # rbfparameters = {
         # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
         # 'gamma': [pow(2, x) for x in range(-13, -9, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVR(), rbfparameters, cv=5, n_jobs=-1, verbose=1, scoring="mean_squared_error")
    # search.fit(vector, targets)
    # print >> output, "RBF SVR Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, "Best Parameters: ", search.best_params_
    # print >> output, "Mean-Squared-Error Score: ", search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()
	
    # rbfparameters = {
         # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
         # 'gamma': [pow(2, x) for x in range(-9, -5, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVR(), rbfparameters, cv=5, n_jobs=-1, verbose=1, scoring="mean_squared_error")
    # search.fit(vector, targets)
    # print >> output, "RBF SVR Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, "Best Parameters: ", search.best_params_
    # print >> output, "Mean-Squared-Error Score: ", search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()
	
    # rbfparameters = {
         # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
         # 'gamma': [pow(2, x) for x in range(-5, -1, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVR(), rbfparameters, cv=5, n_jobs=-1, verbose=1, scoring="mean_squared_error")
    # search.fit(vector, targets)
    # print >> output, "RBF SVR Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, "Best Parameters: ", search.best_params_
    # print >> output, "Mean-Squared-Error Score: ", search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()
	
    # rbfparameters = {
         # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
         # 'gamma': [pow(2, x) for x in range(-1, 3, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVR(), rbfparameters, cv=5, n_jobs=-1, verbose=1, scoring="mean_squared_error")
    # search.fit(vector, targets)
    # print >> output, "RBF SVR Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, "Best Parameters: ", search.best_params_
    # print >> output, "Mean-Squared-Error Score: ", search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()

    # rbfparameters = {
         # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
         # 'gamma': [pow(2, x) for x in range(3, 9, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVR(), rbfparameters, cv=5, n_jobs=-1, verbose=1, scoring="mean_squared_error")
    # search.fit(vector, targets)
    # print >> output, "RBF SVR Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, "Best Parameters: ", search.best_params_
    # print >> output, "Mean-Squared-Error Score: ", search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()



    
    output.close()
