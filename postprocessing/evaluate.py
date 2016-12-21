from __future__ import division

import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix as confusion_matrix_func
from sklearn.model_selection import KFold


def predict_superbowl(encoder, classifier):
    # Predict result for play

    """Superbowl 49 example: The Seahawks decided to pass the football from the 1 yard line, ending in an interception.
    We let the estimator predict this specific play to judge the quality of the actual play call by Seattle's coaches.
    """
    for p in [0, 1]:
        for side in ['left', 'middle', 'right']:
            X = defaultdict(float)
            X['team'] = "SEA"
            X['opponent'] = "NE"
            X['time'] = 26
            X['position'] = 1
            X['half'] = 2
            X['togo'] = 1
            X['shotgun'] = 1
            X['pass'] = p
            if p == 1:
                X['passlen'] = 'short'

            X['side'] = side
            X['qbrun'] = 0
            X['down'] = 2
            X = encoder.transform(X)

            y_pred = classifier.predict(X)
            print p, side, y_pred
    return y_pred


# Evaluate classifier
def classifier_evaluate(classifier, features, labels, k=5):
    confusion_matrix = [[0, 0], [0, 0]]
    k_fold = KFold(len(labels), n_folds=k)
    for train_index, test_index in k_fold:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        confusion_matrix = confusion_matrix + confusion_matrix_func(y_test, y_pred)

    return confusion_matrix


def get_stats_from_confusion_matrix(confusion_matrix):
    print confusion_matrix
    recall = 0 if confusion_matrix[1][1] == 0 else confusion_matrix[1][1] / \
                                                   (confusion_matrix[1][1] + confusion_matrix[1][0])
    precision = 0 if confusion_matrix[1][1] == 0 else confusion_matrix[1][1] / \
                                                      (confusion_matrix[1][1] + confusion_matrix[0][1])
    accuracy = 0 if confusion_matrix[1][1] == 0 and confusion_matrix[0][0] == 0 else \
        (confusion_matrix[0][0] + confusion_matrix[1][1]) / \
               (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])
    f1 = 0 if precision == 0 or recall == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, accuracy, f1


# Evaluate classifier and return recall, precision, accuracy
def classifier_evaluate_percents(classifier, features, labels, k=5):
    confusion_matrix = [[0, 0], [0, 0]]
    k_fold = KFold(n_splits=k)
    for train_index, test_index in k_fold.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        confusion_matrix = confusion_matrix + confusion_matrix_func(y_test, y_pred)

    recall, precision, accuracy, f1 = get_stats_from_confusion_matrix(confusion_matrix)

    return recall, precision, accuracy, f1


# Evaluate regression estimator
def regression_evaluate(classifier, features, labels, k=5):
    abs_diffs = []
    mse_diffs = []
    k_fold = KFold(len(labels), n_folds=k)

    for train_index, test_index in k_fold:
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        for idx in range(len(y_pred)):
            d = abs(y_pred[idx] - y_test[idx])
            abs_diffs.append(d)
            d = d * d
            mse_diffs.append(d)

    avg_abs_diff = sum(abs_diffs) / len(abs_diffs)
    avg_mse_diff = math.sqrt(sum(mse_diffs) / len(mse_diffs))
    print "MAE:",
    print("%.4f" % avg_abs_diff),
    print '/ RMSE:',
    print ("%.4f" % avg_mse_diff)
    return abs_diffs, mse_diffs, avg_abs_diff, avg_mse_diff


# Create a plot of the confusion matrix
def plot_confusion_matrix(cm):
    plt = create_confusion_matrix_plot(cm)
    plt.show()


def save_confusion_matrix(confusion_matrix, file_path):
    plt = create_confusion_matrix_plot(confusion_matrix)
    plt.savefig(file_path)


def create_confusion_matrix_plot(confusion_matrix):
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100.0
    plt.matshow(confusion_matrix_normalized)
    width = len(confusion_matrix)
    height = len(confusion_matrix[0])
    for x in xrange(width):
        for y in xrange(height):
            plt.gca().annotate("{:5.2f} %".format(confusion_matrix_normalized[x][y]), xy=(y, x),
                               horizontalalignment='center',
                               verticalalignment='center')

    tick_labels = ['Fail', 'Success']
    plt.xticks(range(width), tick_labels)
    plt.yticks(range(height), tick_labels)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


# Create a plot of the decision tree
def plot_tree(classifier, feature_names):
    tree.export_graphviz(classifier, out_file='tree.dot', class_names=['Fail', 'Success'], feature_names=feature_names)
    os.system("dot -Tpng tree.dot -o tree.png")
    os.system("tree.png")
