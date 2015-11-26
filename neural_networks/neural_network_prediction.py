from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pickle
import os
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from evaluate import plot_confusion_matrix


def neural_network_prediction(features, labels, target_name, regression_task=True, k=5, team='all',
                              epochs=[100], hidden_layers=[10], hidden_units=[10],
                              sigmoid=True, tanh=True, linear=True, load_previous=True):

    # k-fold cross-validation
    kf = KFold(len(labels), n_folds=k)

    directory = 'neural_networks/trained_models/' + team
    if not os.path.isdir(directory):
        os.mkdir(directory)

    output_file = open(directory + "/neural_net_results_" + target_name + ".txt", "w")
    if regression_task:
        output_file.write('epochs & hidden layers & hidden units & hidden class & RMSE(all) & MAE(all)\\\\ \n')
    else:
        output_file.write('epochs & hidden layers & hidden units & hidden class & accuracy & precision & recall \\\\ \n')

    hidden_classes = []
    if sigmoid:
        hidden_classes.append(SigmoidLayer)
    if tanh:
        hidden_classes.append(TanhLayer)
    if linear:
        hidden_classes.append(LinearLayer)

    for number_of_epochs in epochs:
        for number_of_hidden_layers in hidden_layers:
            for number_of_hidden_units in hidden_units:
                for hidden_class in hidden_classes:
                    if hidden_class == TanhLayer:
                        hidden_class_name = 'Tanh'
                    elif hidden_class == LinearLayer:
                        hidden_class_name = 'Linear'
                    else:
                        hidden_class_name = 'Sigmoid'

                    configuration = {'epochs': number_of_epochs,
                                     'layers': number_of_hidden_layers,
                                     'units':  number_of_hidden_units,
                                     'class':  hidden_class_name}

                    predictions = np.array([])
                    for train_index, test_index in kf:
                        train_x, test_x = np.array(features[train_index]), np.array(features[test_index])
                        train_y, test_y = np.array(labels[train_index]), np.array(labels[test_index])

                        ds, number_of_features = initialize_dataset(regression_task, train_x, train_y)

                        file_name = directory + '/' + target_name + '_' + hidden_class_name + \
                                    '_epochs=%d_layers=%d_units=%d.pickle' % \
                                    (number_of_epochs, number_of_hidden_layers, number_of_hidden_units)

                        net = build_and_train_network(load_previous, file_name, ds, number_of_features, \
                            number_of_epochs, number_of_hidden_layers, number_of_hidden_units, hidden_class)

                        np.concatenate(predictions, predict(net, test_x))

                    evaluate_accuracy(predictions, labels, regression_task, output_file, configuration)

    output_file.close()


def initialize_dataset(regression_task, train_x, train_y):
    number_of_features = train_x.shape[1]
    if regression_task:
        ds = SupervisedDataSet(number_of_features, 1)
    else:
        ds = ClassificationDataSet(number_of_features, nb_classes=2, class_labels=['no success', '1st down or TD'])

    ds.setField('input', train_x)
    ds.setField('target', train_y.reshape((len(train_y), 1)))
    return ds, number_of_features


def build_and_train_network(load_previous, file_name, ds, number_of_features,
                            number_of_epochs, number_of_hidden_layers, number_of_hidden_units, hidden_class):
    if load_previous:
        print 'trying to load previously trained network'
        try:
            with open(file_name, 'r') as net_file:
                net = pickle.load(net_file)
            load_failed = False
        except:
            load_failed = True
            print 'failed to load previously trained network'

    if (not load_previous) or load_failed:
        print 'creating new network'
        # define number of units per layer
        layers = [number_of_features]
        layers.extend([number_of_hidden_units]*number_of_hidden_layers)
        layers.append(1)

        #Build Neural Network
        net = buildNetwork(
               *layers,
               bias = True,
               hiddenclass = hidden_class,
               outclass = LinearLayer
               )

    trainer = BackpropTrainer(net, ds, learningrate=0.01, lrdecay=1.0, momentum=0.0, weightdecay=0.0, verbose=True)

    trainer.trainUntilConvergence(maxEpochs=number_of_epochs)

    with open(file_name, 'w') as net_file:
        pickle.dump(net, net_file)
    print 'saved new network to file ' + file_name

    return net


def predict(net, feature_vectors):
    predictions = []
    for x in feature_vectors:
        predictions.append(net.activate(x)[0])
    return np.array(predictions)


def evaluate_accuracy(predictions, labels, regression_task, output_file, configuration):
    if regression_task:
        evaluate_regression(predictions, labels, output_file, configuration)
    else:
        evaluate_classification(predictions, labels, output_file, configuration)


def evaluate_classification(predictions, labels, output_file, configuration):
    cm = confusion_matrix(labels, predictions)
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    plot_confusion_matrix(cm)

    # format output for LaTeX
    output_file.write('%d & %d & %d & %s & %f & %f & %f \\\\ \n' %
               (configuration['epochs'], configuration['layers'], configuration['units'], configuration['class'],
                accuracy, precision, recall))


def evaluate_regression(predictions, labels, output_file, configuration):
    # format output for LaTeX
    output_file.write('%d & %d & %d & %s & %f & %f \\\\ \n' %
               (configuration['epochs'], configuration['layers'], configuration['units'], configuration['class'],
                mean_squared_error(labels, predictions)**0.5, mean_absolute_error(labels, predictions)))
