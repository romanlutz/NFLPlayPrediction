from __future__ import division

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from postprocessing.evaluate import regression_evaluate

'''
estimator = the SVM you wish to use to classify the data
features = a (sample size) x (features) array containing the feature vectors of the data
targets = an array of the correct classification for each of the sample feature vectors
k_fold = the number of folds to use in the cross validation. Defaults to 5 fold if not specified.

Returns (mean, standard deviation) of the provided estimator's accuracy using kfold validation,
and utilizes all available CPUs for the training and validation.
'''

result_file_name = './results/regression_results_%s.txt'
pkl_file_name = './results/regression_%s.pkl'


def write_result_stats_to_file(file_name, target_name, estimator_name, average_difference, average_mse_difference):
    output = open(file_name % target_name, 'a')
    print >> output, "**********************************"
    print >> output, estimator_name
    print >> output, "Average Difference from Goal: ", average_difference
    print >> output, "Average MSE from Goal: ", average_mse_difference
    print >> output, "**********************************"
    output.flush()
    output.close()


def grid_search_rbf_parameters(vectorized_features, targets, file_name, target_name, estimator_name,
                               c_values, gamma_values, k_fold=5, scoring="mean_squared_error"):
    output = open(file_name % target_name, 'a')
    print >> output, "**********************************"
    print >> output, "Searching for SVR RBF Parameters"
    rbf_parameters = {'C': c_values, 'gamma': gamma_values}
    search = GridSearchCV(SVR(), rbf_parameters, cv=k_fold, n_jobs=-1, verbose=1, scoring=scoring)
    search.fit(vectorized_features, targets)
    print >> output, "%s Best Estimator:" % estimator_name
    print >> output, search.best_estimator_
    print >> output, "Best Parameters: ", search.best_params_
    print >> output, "Mean-Squared-Error Score: ", search.best_score_
    print >> output, "Grid Scores:"
    print >> output, search.cv_results_
    output.flush()
    output.close()


def compute_regression_results(features, targets, target_name, config):
    vectorized_features = features
    #TODO Consider replacing the above with FeatureHasher for faster computation?

    estimators = [
                    (LinearRegression(normalize=True), "LinearRegression"),
                    #(SVR(C=128, kernel='rbf', gamma=pow(2, -1)), "RBF SVR, C=128, Gamma= 2^-1"),
                    (tree.DecisionTreeRegressor(max_depth=10), 'Decision Tree')
                 ]

    for (estimator, estimator_name) in estimators:
        print 'using %s' % estimator_name
        abs_diff, mse_diff, avg_diff, avg_mse_diff = regression_evaluate(estimator, vectorized_features, targets)
        write_result_stats_to_file(result_file_name, target_name, estimator_name, avg_diff, avg_mse_diff)
        #joblib.dump(estimator, pkl_file_name % estimator_name)

    if config['grid_search']:
        grid_search_rbf_parameters(vectorized_features=vectorized_features, targets=targets, file_name=result_file_name,
                               target_name=target_name, estimator_name='RBF SVR',
                               c_values=[pow(2, x) for x in range(-5, 17, 2)],
                               gamma_values=[pow(2, x) for x in range(-17, 9, 2)],
                               k_fold=5, scoring="mean_squared_error")

