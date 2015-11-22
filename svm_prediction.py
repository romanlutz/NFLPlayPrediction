from __future__ import division
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import LinearSVC
from collections import Counter
from random import random
from evaluate import clf_evaluate_percents
import matplotlib.pyplot as plt

'''
estimator = the SVM you wish to use to classify the data
features = a (sample size) x (features) array containing the feature vectors of the data
true_labels = an array of the correct classification for each of the sample feature vectors
kfold = the number of folds to use in the cross validation. Defaults to 5 fold if not specified.

Returns (mean, standard deviation) of the provided estimator's accuracy using kfold validation,
and utilizes all available CPUs for the training and validation.
'''


def test_classifier(estimator, features, true_labels, kfold=5):
    scores = cross_validation.cross_val_score(estimator, features, true_labels, cv=kfold, n_jobs=-2, verbose=1)
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
	return (balanced_dataset, new_labels)

def compare_RBF_parameters(features, true_labels):
    feats, labels = under_sample(features,true_labels)
    vec = DictVectorizer()
    vector = vec.fit_transform(feats).toarray()
    labels = np.asarray(labels)
    # Consider replacing the above with FeatureHasher for faster computation?
   
    output = open('./classifier_results.txt', 'w+')
    ldaclassifier = LinearDiscriminantAnalysis()
    (recall,precision,accuracy) = clf_evaluate_percents(ldaclassifier, vector,labels);
    print >> output, "**********************************"
    print >> output, "LDA"
    print >> output, "Recall:", recall * 100, '%'
    print >> output, "Precision:", precision * 100, '%'
    print >> output, "Accuracy:", accuracy * 100, '%'
    print >> output, "**********************************"
    output.flush()
	
    sgdclass = SGDClassifier()
    (recall,precision,accuracy) = clf_evaluate_percents(sgdclass, vector, labels);
    print >> output, "**********************************"
    print >> output, "SGDC"
    print >> output, "Recall:", recall * 100, '%'
    print >> output, "Precision:", precision * 100, '%'
    print >> output, "Accuracy:", accuracy * 100, '%'
    print >> output, "**********************************"
    output.flush()
	
    treeclf = tree.DecisionTreeClassifier()
    (recall,precision,accuracy) = clf_evaluate_percents(treeclf, vector, labels);
    print >> output, "**********************************"
    print >> output, "Decision Tree"
    print >> output, "Recall:", recall * 100, '%'
    print >> output, "Precision:", precision * 100, '%'
    print >> output, "Accuracy:", accuracy * 100, '%'
    print >> output, "**********************************"
    output.flush()
	
	
    ncclf = NearestCentroid()
    (recall,precision,accuracy) = clf_evaluate_percents(ncclf, vector, labels);
    print >> output, "**********************************"
    print >> output, "NearestCentroid"
    print >> output, "Recall:", recall * 100, '%'
    print >> output, "Precision:", precision * 100, '%'
    print >> output, "Accuracy:", accuracy * 100, '%'
    print >> output, "**********************************"
    output.flush()
	
	
    linsvm = LinearSVC(C = 0.03125)
    linsvm.fit(vector,labels)
    (recall,precision,accuracy) = clf_evaluate_percents(linsvm, vector, labels);
    print >> output, "**********************************"
    print >> output, "Linear SVM"
    print >> output, "Recall:", recall * 100, '%'
    print >> output, "Precision:", precision * 100, '%'
    print >> output, "Accuracy:", accuracy * 100, '%'
    print >> output, "**********************************"
    output.flush()
	
    rbfsvm = SVC(C = 2048, kernel='rbf', gamma=pow(2,-17))
    rbfsvm.fit(vector,labels)
    (recall,precision,accuracy) = clf_evaluate_percents(rbfsvm, vector, labels);
    print >> output, "**********************************"
    print >> output, "RBF SVM, C=2048, Gamma= 2^-17"
    print >> output, "Recall:", recall * 100, '%'
    print >> output, "Precision:", precision * 100, '%'
    print >> output, "Accuracy:", accuracy * 100, '%'
    print >> output, "**********************************"
    output.flush()
	
    # linsvmparams = {'C': [pow(2,x) for x in range(-5,15,2)]}
    # search = GridSearchCV(LinearSVC(class_weight='auto'), linsvmparams, cv=3, n_jobs=-1, verbose=1)
    # search.fit(vector, labels)
    # print >> output, "Linear SVM Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, ""
    # print >> output, "Parameters:"
    # print >> output, search.best_params_
    # print >> output, ""
    # print >> output, "Score:"
    # print >> output, search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()

	
    # rbfparameters = {
        # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        # 'gamma': [pow(2, x) for x in range(-17, -13, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    # search.fit(vector, labels)
    # print >> output, "RBF SVM Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, ""
    # print >> output, "Parameters:"
    # print >> output, search.best_params_
    # print >> output, ""
    # print >> output, "Score:"
    # print >> output, search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()
	
    # rbfparameters = {
        # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        # 'gamma': [pow(2, x) for x in range(-13, -9, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    # search.fit(vector, labels)
    # print >> output, "RBF SVM Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, ""
    # print >> output, "Parameters:"
    # print >> output, search.best_params_
    # print >> output, ""
    # print >> output, "Score:"
    # print >> output, search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()
	
	
    # rbfparameters = {
        # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        # 'gamma': [pow(2, x) for x in range(-9, -5, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    # search.fit(vector, labels)
    # print >> output, "RBF SVM Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, ""
    # print >> output, "Parameters:"
    # print >> output, search.best_params_
    # print >> output, ""
    # print >> output, "Score:"
    # print >> output, search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()

    # rbfparameters = {
        # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        # 'gamma': [pow(2, x) for x in range(-5, 1, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    # search.fit(vector, labels)
    # print >> output, "RBF SVM Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, ""
    # print >> output, "Parameters:"
    # print >> output, search.best_params_
    # print >> output, ""
    # print >> output, "Score:"
    # print >> output, search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    # output.flush()

    # rbfparameters = {
        # 'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        # 'gamma': [pow(2, x) for x in range(1, 4, 2)]  # Possible gamma values for the SVM.
    # }
    # search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    # search.fit(vector, labels)
    # print >> output, "RBF SVM Best Estimator:"
    # print >> output, search.best_estimator_
    # print >> output, ""
    # print >> output, "Parameters:"
    # print >> output, search.best_params_
    # print >> output, ""
    # print >> output, "Score:"
    # print >> output, search.best_score_
    # print >> output, "Grid Scores:"
    # print >> output, search.grid_scores_
    
    output.close()
