from __future__ import division
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import os
from sklearn.cross_validation import KFold


# Evaluate classifier
def clf_evaluate(clf, features, labels, k=5):

    cm = [[0,0],[0,0]]
    kf = KFold(len(labels), n_folds=k)
    for train_index, test_index in kf:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_pred = clf.fit(X_train, y_train).predict(X_test)       
        cm = cm + confusion_matrix(y_test, y_pred)

    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    f1 = (2*precision*recall)/(precision+recall)

    print "**********************************"
    print "Accuracy:", accuracy * 100, '%'
    print "Precision:", precision * 100, '%'
    print "Recall:", recall * 100, '%'
    print "F1:",f1 * 100, '%' 
    print "**********************************"
    
    return cm
	
	# Evaluate classifier and return recall, precision, accuracy
def clf_evaluate_percents(clf, features, labels, k=5):

    cm = [[0,0],[0,0]]
    kf = KFold(len(labels), n_folds=k)
    for train_index, test_index in kf:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_pred = clf.fit(X_train, y_train).predict(X_test)       
        cm = cm + confusion_matrix(y_test, y_pred)

    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    f1 = (2*precision*recall)/(precision+recall)

    #print "**********************************"
    #print "Recall:", recall * 100, '%'
    # print "Precision:", precision * 100, '%'
    #print "Accuracy:", accuracy * 100, '%'
    # print "**********************************"
    return (recall, precision, accuracy)

import math
# Evaluate regression estimator
def reg_evaluate(clf, features, labels,k=5):

    abs_diffs = []
    mse_diffs = []
    kf = KFold(len(labels), n_folds=k)
    for train_index, test_index in kf:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        
        y_pred = clf.predict(X_test)
        for idx in range(len(y_pred)):    
            d = abs(y_pred[idx] - y_test[idx])
            abs_diffs.append(d)
            d = d*d
            mse_diffs.append(d)
            
    avg_abs_diff = sum(abs_diffs)/len(abs_diffs)
    avg_mse_diff = math.sqrt(sum(mse_diffs)/len(mse_diffs))
    print "MAE:",avg_abs_diff,'/ RMSE:',avg_mse_diff
    return abs_diffs,mse_diffs,avg_abs_diff,avg_mse_diff
    

# Create a plot of the confusion matrix
def plot_confusion_matrix(cm):
    plt = create_confusion_matrix_plot(cm)
    plt.show()

def save_confusion_matrix(cm, file_path):
    plt = create_confusion_matrix_plot(cm)
    plt.savefig(file)

def create_confusion_matrix_plot(cm):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100.0
    plt.matshow(cm_normalized)
    width = len(cm)
    height = len(cm[0])
    for x in xrange(width):
        for y in xrange(height):
            plt.gca().annotate("{:5.2f} %".format(cm_normalized[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    tick_labels = ['Fail','Success']
    plt.xticks(range(width), tick_labels)
    plt.yticks(range(height), tick_labels)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

    
# Create a plot of the decision tree
def plot_tree(clf,feature_names):
    tree.export_graphviz(clf,out_file='tree.dot',class_names=['Fail','Success'],feature_names=feature_names)
    os.system("dot -Tpng tree.dot -o tree.png")
    os.system("tree.png")
