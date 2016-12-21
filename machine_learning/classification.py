from __future__ import division

from collections import Counter
from random import random

from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from postprocessing.evaluate import classifier_evaluate_percents

'''
estimator = the SVM you wish to use to classify the data
features = a (sample size) x (features) array containing the feature vectors of the data
true_labels = an array of the correct classification for each of the sample feature vectors
kfold = the number of folds to use in the cross validation. Defaults to 5 fold if not specified.

Returns (mean, standard deviation) of the provided estimator's accuracy using kfold validation,
and utilizes all available CPUs for the training and validation.
'''


result_file_name = './results/classifier_results.txt'


def write_result_stats_to_file(file_name, estimator_name, recall, precision, accuracy, f1):
    output = open(file_name, 'a')
    print >> output, "**********************************"
    print >> output, estimator_name
    print >> output, "Recall:", recall * 100, '%'
    print >> output, "Precision:", precision * 100, '%'
    print >> output, "Accuracy:", accuracy * 100, '%'
    print >> output, "F1:", f1 * 100, '%'
    print >> output, "**********************************"
    output.flush()
    output.close()


def write_search_results_to_file(file_name, estimator_name, search):
    output = open(file_name, 'a')
    print >> output, estimator_name
    print >> output, search.best_estimator_
    print >> output, ""
    print >> output, "Parameters:"
    print >> output, search.best_params_
    print >> output, ""
    print >> output, "Score:"
    print >> output, search.best_score_
    print >> output, "Grid Scores:"
    print >> output, search.grid_scores_
    output.close()
    output.flush()


def test_classifier(estimator, features, true_labels, kfold=5):
    scores = cross_val_score(estimator, features, true_labels, cv=kfold, n_jobs=-2, verbose=1)
    return scores.mean(), scores.std()


def under_sample(vector, true_labels):
    counts = Counter(true_labels)
    total = len(true_labels)
    proportion = {}
    for label in counts:
        proportion[label] = 0.00 + counts[label] / total
    min_prop = min(proportion.values())
    weights = {}
    for label in counts:
        weights[label] = min_prop / proportion[label]
    balanced_dataset = []
    new_labels = []
    for idx, label in enumerate(true_labels):
        if random() < weights[label]:
            new_labels.append(label)
            balanced_dataset.append(vector[idx])
    print Counter(new_labels)
    return balanced_dataset, new_labels


def compare_classification_parameters(features, labels, config):
    #TODO Consider replacing the above with FeatureHasher for faster computation?

    #linsvm = LinearSVC(C=0.03125)
    #linsvm.fit(vectorized_features, labels)

    #rbfsvm = SVC(C=2048, kernel='rbf', gamma=pow(2, -17))
    #rbfsvm.fit(vectorized_features, labels)

    estimators = [
        #(LinearDiscriminantAnalysis(), "SVD LDA"),
        (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), "LSQR LDA"),
        (LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'), "Eigenvalue Decomposition LDA"),
        (SGDClassifier(), "SGDC"),
        (tree.DecisionTreeClassifier(class_weight='balanced', max_depth=10), "Decision Tree"),
        (NearestCentroid(), "NearestCentroid"),
        #(linsvm, "Linear SVM"),
        #(rbfsvm, "RBF SVM, C=2048, Gamma= 2^-17")
    ]

    for (estimator, estimator_name) in estimators:
        print 'using %s' % estimator_name
        (recall, precision, accuracy, f1) = classifier_evaluate_percents(estimator, features, labels)
        write_result_stats_to_file(result_file_name, estimator_name, recall, precision, accuracy, f1)

    if config['grid_search']:
        linear_svm_params = {'C': [pow(2, x) for x in range(-5, 15, 2)]}
        search = GridSearchCV(LinearSVC(class_weight='balanced'), linear_svm_params, cv=3, n_jobs=-1, verbose=1)
        search.fit(features, labels)
        write_search_results_to_file(result_file_name, "Linear SVM Best Estimator", search)

        rbf_parameters = {
            'C': [pow(2, x) for x in range(-5, 17, 2)],  # Possible error weights for the SVM.
            'gamma': [pow(2, x) for x in range(-17, 4, 2)]  # Possible gamma values for the SVM.
            }
        search = GridSearchCV(SVC(class_weight='balanced'), rbf_parameters, cv=3, n_jobs=-1, verbose=1)
        search.fit(features, labels)
        write_search_results_to_file(result_file_name, "RBF SVM Best Estimator", search)

