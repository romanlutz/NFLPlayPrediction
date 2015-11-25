import random

import numpy as np

from feat import extract_features, encode_categorical_features
from neural_networks.neural_network_prediction import neural_network_prediction
from svm_prediction import compare_RBF_parameters

random.seed(0)  # keep seed fixed for reproducibility


def main_valentin():
    (features, labels, _, _) = extract_features(2014, 2014)
    compare_RBF_parameters(features, labels)


def main_brendan():
    (features, labels, _, _) = extract_features(2014, 2014)
    compare_RBF_parameters(features, labels)


def main_roman():
    (features, success_labels, yard_labels, progress_labels) = extract_features(2009, 2014)
    features, enc = encode_categorical_features(features, sparse=False)
    print enc.vocabulary_

    # hold out 10000 as test set
    neural_network_prediction(features=features,
                              labels=progress_labels,
                              k=5,
                              team='all',
                              target_name='progress',
                              regression_task=True,
                              epochs        = [10, 100, 1000],
                              hidden_layers = [1, 10, 100],
                              hidden_units  = [10, 100],
                              load_previous=True)

def __main__():
    main_roman()


__main__()
