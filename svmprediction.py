from __future__ import division
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

'''
estimator = the SVM you wish to use to classify the data
features = a (sample size) x (features) array containing the feature vectors of the data
true_labels = an array of the correct classification for each of the sample feature vectors
kfold = the number of folds to use in the cross validation. Defaults to 5 fold if not specified.

Returns (mean, standard deviation) of the provided eestimator's accuracy using kfold validation,
and utilizes all available CPUs for the training and validation.
'''
def test_SVM(estimator, features, true_labels, kfold=5):
	scores = cross_validation.cross_val_score(estimator, features, true_labels, cv=kfold, n_jobs=-2, verbose=1)
	return scores.mean(), scores.std()


def compare_RBF_parameters(features, true_labels):
	vec = DictVectorizer()
	vector = vec.fit_transform(features).toarray()
	#Consider replacing the above with FeatureHasher for faster computation?
	parameters =	{
			'C': [pow(2,x) for x in range(-3,15,2)], #Possible error weights for the SVM.
			'gamma': [pow(2,x) for x in range(-7,7,2)] #Possible gamma values for the SVM.	
			}
	search = GridSearchCV(SVC(), parameters, cv=3, n_jobs=-1, verbose=1)
	search.fit(vector, true_labels)
	print "Best Estimator:"
	print search.best_estimator_
	print
	print "Parameters:"
	print search.best_params_
	print
	print "Score:"
	print search.best_score_

def plot_confusion_matrix(estimator,features,labels):
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0)

    # Run classifier
    y_pred = estimator.fit(X_train, y_train).predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1][0]  / (cm[0][0] + cm[1][0])
    precision = cm[1][0]  / (cm[1][0] + cm[1][1])
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
      

    print(cm)
    print "Recall:",recall*100,'%'
    print "Precision:",precision*100,'%'
    print "Accuracy:",accuracy*100,'%'

    # Show confusion matrix in a separate window
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.matshow(cm_normalized)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return cm