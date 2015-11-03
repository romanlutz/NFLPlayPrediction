import random
from features import extract_features
from svmprediction import compare_RBF_parameters

random.seed(0)  # keep seed fixed for reproducibility
(features, labels, _, _) = extract_features(2014, 2014)
compare_RBF_parameters(features,labels)
