import random

import numpy as np

from feat import extract_features, encode_categorical_features
from neural_networks.neural_network_prediction import neural_network_prediction
from svm_prediction import compare_RBF_parameters
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
    (features, success_labels, yard_labels, progress_labels) = extract_features(2009, 2014)
    features, enc = encode_categorical_features(features, sparse=False)
    print enc.vocabulary_

    configurations = {'success':  {'labels': success_labels, 'target': 'success', 'regression': False},
                      'yards':    {'labels': yard_labels, 'target': 'yards', 'regression': True},
                      'progress': {'labels': progress_labels, 'target': 'progress', 'regression': True}}

    selected_configuration = 'success'

    neural_network_prediction(features=features,
                              labels=configurations[selected_configuration]['labels'],
                              k=5,
                              team='all', suffix='_100_5_100_tanh',
                              target_name=configurations[selected_configuration]['target'],
                              regression_task=configurations[selected_configuration]['regression'],
                              epochs        = [100],
                              hidden_layers = [5],
                              hidden_units  = [100],
                              load_previous = True,
                              tanh=True,
                              sigmoid=False,
                              linear=False)

def __main__():
    main_roman()


__main__()
