import random
from features import extract_features
from svmprediction import compare_RBF_parameters

random.seed(0)  # keep seed fixed for reproducibility


def main_valentin():
    (features, labels, _, _) = extract_features(2014, 2014)
    compare_RBF_parameters(features, labels)

def main_brendan():
    (features, labels, _, _) = extract_features(2014, 2014)
    compare_RBF_parameters(features, labels)

def main_roman():
    (features, labels, _, _) = extract_features(2014, 2014)

def __main__():
    main_roman()

__main__()
