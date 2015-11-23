import random
from feat import extract_features, encode_categorical_features
from svm_prediction import compare_RBF_parameters
from regression import compute_regression_results
from neural_network_prediction import neural_network_prediction
from regression import compute_regression_results
import numpy as np

random.seed(0)  # keep seed fixed for reproducibility


def main_valentin():
    (features, labels, _, _) = extract_features(2014, 2014)
    compare_RBF_parameters(features, labels)


def main_brendan():
    (features, labels, yards, progress) = extract_features(2014, 2014)
    #compare_RBF_parameters(features, labels)
    compute_regression_results(features,yards, "./regression_results_yards.txt")
    compute_regression_results(features,progress, "./regression_results_progress.txt")


def main_roman():
    (features, labels, _, _) = extract_features(2014, 2014)
    features, enc = encode_categorical_features(features, sparse=False)
    print enc.vocabulary_

    # hold out 10000 as test set
    train_x = np.array(features[:-10000])
    train_y = np.array(labels[:-10000])
    test_x = np.array(features[-10000:])
    test_y = np.array(labels[-10000:])
    neural_network_prediction(train_x, train_y, test_x, test_y)

def __main__():
    main_brendan()


__main__()
