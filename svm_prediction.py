from __future__ import division
    rbfparameters = {
        'C': [pow(2, x) for x in range(-5,17, 2)],  # Possible error weights for the SVM.
        'gamma': [pow(2, x) for x in range(-5, 0, 2)]  # Possible gamma values for the SVM.
    }
    search = GridSearchCV(SVC(class_weight='auto'), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, true_labels)
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


def compare_RBF_parameters(features, true_labels):
    vec = DictVectorizer()
    vector = vec.fit_transform(features).toarray()
    # Consider replacing the above with FeatureHasher for faster computation?
   
    output = open('./svmparameters.txt', 'w+')
    ldaclassifier = LinearDiscriminantAnalysis()
    (mean,std) = test_classifier(ldaclassifier, vector, true_labels, kfold=3);
    print >> output, "LDA: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    sgdclass = SGDClassifier()
    (mean, std) = test_classifier(sgdclass, vector, true_labels, kfold=3);
    print >> output, "SGDC: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    treeclf = tree.DecisionTreeClassifier()
    (mean, std) = test_classifier(treeclf, vector, true_labels, kfold=3);
    print >> output, "Decision Tree: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    ncclf = NearestCentroid()
    (mean, std) = test_classifier(ncclf, vector, true_labels, kfold=3);
    print >> output, "Nearest Centroid: Mean, STD:"
    print >> output, (mean, std)
    output.flush()
    linsvmparams = {'C': [pow(2,x) for x in range(-5,15,2)]}
    search = GridSearchCV(LinearSVC(), linsvmparams, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, true_labels)
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
        'gamma': [pow(2, x) for x in range(-9, -5, 2)]  # Possible gamma values for the SVM.
    }
    search = GridSearchCV(SVC(), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, true_labels)
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
    search = GridSearchCV(SVC(), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, true_labels)
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
    search = GridSearchCV(SVC(), rbfparameters, cv=3, n_jobs=-1, verbose=1)
    search.fit(vector, true_labels)
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
