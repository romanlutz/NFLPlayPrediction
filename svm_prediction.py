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
    # Consider replacing the above with FeatureHasher for faster computation?
   
    output = open('./svmparameters.txt', 'w+')
    ldaclassifier = LinearDiscriminantAnalysis()
    (mean,std) = test_classifier(ldaclassifier, vector,labels, kfold=3);
    print >> output, "LDA: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    sgdclass = SGDClassifier(class_weight='auto')
    (mean, std) = test_classifier(sgdclass, vector, labels, kfold=3);
    print >> output, "SGDC: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    treeclf = tree.DecisionTreeClassifier(class_weight='auto')
    (mean, std) = test_classifier(treeclf, vector, labels, kfold=3);
    print >> output, "Decision Tree: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    ncclf = NearestCentroid()
    (mean, std) = test_classifier(ncclf, vector, labels, kfold=3);
    print >> output, "Nearest Centroid: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    linsvm = LinearSVC(class_weight='auto', C = pow(2,-5))
    linsvm.fit(vector,labels)
    print >> output, "linsvm c 2e-5 guesses"
    print >> output, Counter(linsvm.predict(vector))
    output.flush()
    linsvmparams = {'C': [pow(2,x) for x in range(-5,15,2)]}
    search = GridSearchCV(LinearSVC(class_weight='auto'), linsvmparams, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, labels)
    print >> output, "Linear SVM Best Estimator:"
    print >> output, search.best_estimator_
    print >> output, ""
    print >> output, "Parameters:"
    print >> output, search.best_params_
    print >> output, ""
    print >> output, "Score:"
    print >> output, search.best_score_
    print >> output, "Grid Scores:"
    print >> output, search.grid_scores_
    output.flush()

	
    rbfparameters = {
        'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        'gamma': [pow(2, x) for x in range(-17, -13, 2)]  # Possible gamma values for the SVM.
    }
    search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, labels)
    print >> output, "RBF SVM Best Estimator:"
    print >> output, search.best_estimator_
    print >> output, ""
    print >> output, "Parameters:"
    print >> output, search.best_params_
    print >> output, ""
    print >> output, "Score:"
    print >> output, search.best_score_
    print >> output, "Grid Scores:"
    print >> output, search.grid_scores_
    output.flush()
	
    rbfparameters = {
        'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        'gamma': [pow(2, x) for x in range(-13, -9, 2)]  # Possible gamma values for the SVM.
    }
    search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, labels)
    print >> output, "RBF SVM Best Estimator:"
    print >> output, search.best_estimator_
    print >> output, ""
    print >> output, "Parameters:"
    print >> output, search.best_params_
    print >> output, ""
    print >> output, "Score:"
    print >> output, search.best_score_
    print >> output, "Grid Scores:"
    print >> output, search.grid_scores_
    output.flush()
	
	
    rbfparameters = {
        'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        'gamma': [pow(2, x) for x in range(-9, -5, 2)]  # Possible gamma values for the SVM.
    }
    search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, labels)
    print >> output, "RBF SVM Best Estimator:"
    print >> output, search.best_estimator_
    print >> output, ""
    print >> output, "Parameters:"
    print >> output, search.best_params_
    print >> output, ""
    print >> output, "Score:"
    print >> output, search.best_score_
    print >> output, "Grid Scores:"
    print >> output, search.grid_scores_
    output.flush()

    rbfparameters = {
        'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        'gamma': [pow(2, x) for x in range(-5, 1, 2)]  # Possible gamma values for the SVM.
    }
    search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, labels)
    print >> output, "RBF SVM Best Estimator:"
    print >> output, search.best_estimator_
    print >> output, ""
    print >> output, "Parameters:"
    print >> output, search.best_params_
    print >> output, ""
    print >> output, "Score:"
    print >> output, search.best_score_
    print >> output, "Grid Scores:"
    print >> output, search.grid_scores_
    output.flush()

    rbfparameters = {
        'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        'gamma': [pow(2, x) for x in range(1, 4, 2)]  # Possible gamma values for the SVM.
    }
    search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, labels)
    print >> output, "RBF SVM Best Estimator:"
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
def plot_confusion_matrix(estimator, features, labels):
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0)

    # Run classifier
    y_pred = estimator.fit(X_train, y_train).predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1][0] / (cm[0][0] + cm[1][0])
    precision = cm[1][0] / (cm[1][0] + cm[1][1])
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

    print(cm)
    print "Recall:", recall * 100, '%'
    print "Precision:", precision * 100, '%'
    print "Accuracy:", accuracy * 100, '%'

    # Show confusion matrix in a separate window
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.matshow(cm_normalized)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return cm
