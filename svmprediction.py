import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer

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
			'C': [pow(2,x) for x in range(-5,10,2)], #Possible error weights for the SVM.
			'gamma': [pow(2,x) for x in range(-15,3,3)] #Possible gamma values for the SVM.	
			}
	search = GridSearchCV(SVC(), parameters, cv=5, n_jobs=2, verbose=1)
	search.fit(vector, true_labels)
	print "Best Estimator:"
	print results.best_estimator_
	print
	print "Parameters:"
	print results.best_params_
	print
	print "Score:"
	print results.best_score_
