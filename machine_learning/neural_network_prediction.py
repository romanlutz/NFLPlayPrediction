import os
import pickle

import numpy as np
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from postprocessing.evaluate import save_confusion_matrix


def neural_network_prediction(data, config, measure, team='all',
                              layer_type='tanh', load_previous=True):
    regression_task = config['prediction_method_%s' % measure] == 'regression'

    # k-fold cross-validation
    k_fold = KFold(n_splits=config['neural_net_config']['k_fold'])

    model_directory = 'machine_learning/trained_models/' + team
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    suffix = '%s_%s_%s_%s' % (config['neural_net_config']['epochs'],
                              config['neural_net_config']['hidden_layers'],
                              config['neural_net_config']['hidden_units'],
                              layer_type)
    result_file_name = "results/neural_networks_" + measure + suffix + ".txt"
    output_file = open(result_file_name, "w")
    if regression_task:
        output_file.write('epochs & hidden layers & hidden units & hidden class & RMSE(all) & MAE(all)\\\\ \n')
    else:
        output_file.write(
            'epochs & hidden layers & hidden units & hidden class & accuracy & precision & recall \\\\ \n')

    hidden_class = SigmoidLayer
    if 'layer_type' == 'tanh':
        hidden_class = TanhLayer
    if 'layer_type' == 'linear':
        hidden_class = LinearLayer

    for number_of_epochs in config['neural_net_config']['epochs']:
        for number_of_hidden_layers in config['neural_net_config']['hidden_layers']:
            for number_of_hidden_units in config['neural_net_config']['hidden_units']:

                configuration = {'epochs': number_of_epochs,
                                 'layers': number_of_hidden_layers,
                                 'units': number_of_hidden_units,
                                 'class': layer_type}

                predictions = np.array([])

                # try:
                for i in range(1):
                    cross_val_index = 1
                    for train_index, test_index in k_fold.split(data['categorical_features']):
                        train_x, test_x = np.array(data['categorical_features'][train_index]), np.array(data['categorical_features'][test_index])
                        train_y, test_y = np.array(data[measure][train_index]), np.array(data[measure][test_index])

                        ds, number_of_features = initialize_dataset(regression_task, train_x, train_y)

                        file_name = model_directory + '/' + measure + '_' + layer_type + \
                                    '_epochs=%d_layers=%d_units=%d_part=%d.pickle' % \
                                    (number_of_epochs, number_of_hidden_layers, number_of_hidden_units, cross_val_index)

                        net = build_and_train_network(load_previous, file_name, ds, number_of_features,
                                                    number_of_epochs, number_of_hidden_layers, number_of_hidden_units,
                                                    hidden_class)

                        predictions = np.concatenate((predictions, predict(net, test_x)))
                        cross_val_index += 1

                    evaluate_accuracy(predictions, data[measure], regression_task, result_file_name, output_file, configuration)

                    # except:
                    #    pass

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
            print 'succeed to load previously trained network'
        except:
            load_failed = True
            print 'failed to load previously trained network'

    if (not load_previous) or load_failed:
        print 'creating new network'
        # define number of units per layer
        layers = [number_of_features]
        layers.extend([number_of_hidden_units] * number_of_hidden_layers)
        layers.append(1)

        # Build Neural Network
        net = buildNetwork(
            *layers,
            bias=True,
            hiddenclass=hidden_class,
            outclass=LinearLayer
        )

        trainer = BackpropTrainer(net, ds, learningrate=0.01, lrdecay=1.0, momentum=0.0, weightdecay=0.0, verbose=True)

        trainer.trainUntilConvergence(maxEpochs=number_of_epochs)

        print 'trained new network'

    with open(file_name, 'w') as net_file:
        pickle.dump(net, net_file)
    print 'saved new network to file ' + file_name

    return net


def predict(net, feature_vectors):
    predictions = []
    for x in feature_vectors:
        predictions.append(net.activate(x)[0])
    return np.array(predictions)


def evaluate_accuracy(predictions, labels, regression_task, output_file_name, output_file, configuration):
    if regression_task:
        evaluate_regression(predictions, labels, output_file, configuration)
    else:
        evaluate_classification(predictions, labels, output_file_name, output_file, configuration)


def evaluate_classification(predictions, labels, output_file_name, output_file, configuration):
    print labels[:10], [0 if p < 0.5 else 1 for p in predictions[:10]]
    cm = confusion_matrix(labels, [0 if p < 0.5 else 1 for p in predictions])
    print cm, cm[0][0], cm[0][1], cm[1][0], cm[1][1], float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    recall = float(cm[1][1]) / float(cm[1][1] + cm[1][0]) \
        if cm[1][1] + cm[1][0] > 0 else 0
    precision = float(cm[1][1]) / float(cm[1][1] + cm[0][1]) \
        if cm[1][1] + cm[0][1] > 0 else 0
    accuracy = float(cm[0][0] + cm[1][1]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]) \
        if cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] > 0 else 0
    f1 = float(2 * precision * recall) / float(precision + recall) if precision + recall > 0 else 0
    save_confusion_matrix(cm, output_file_name[:-7] + '.png')

    # format output for LaTeX
    output_file.write('%d & %d & %d & %s & %f & %f & %f & %f \\\\ \n' %
                      (configuration['epochs'], configuration['layers'], configuration['units'], configuration['class'],
                       accuracy, precision, recall, f1))


def evaluate_regression(predictions, labels, output_file, configuration):
    # format output for LaTeX
    output_file.write('%d & %d & %d & %s & %f & %f \\\\ \n' %
                      (configuration['epochs'], configuration['layers'], configuration['units'], configuration['class'],
                       mean_squared_error(labels, predictions) ** 0.5, mean_absolute_error(labels, predictions)))
