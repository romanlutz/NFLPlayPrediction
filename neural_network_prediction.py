from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy import asarray


def neural_network_prediction(train_x, train_y, test_x, test_y):

    print len(train_x)
    number_of_features = train_x.shape[1]

    ds = SupervisedDataSet(number_of_features, 1)

    ds.setField('input', train_x)
    ds.setField('target', train_y.reshape((len(train_y), 1)))

    file = open("neural_net_output.txt", "w")
    file.write('epochs & hidden units & hidden class & RMSE(all) & MAE(all)\\\\ \n')

    for number_of_epochs in [10, 50, 100, 1000]:
        for number_of_hidden_units in [10, 25, 50, 100]:
            for hidden_class in [SigmoidLayer, TanhLayer]:
                if hidden_class == TanhLayer:
                    hidden_class_name = 'Tanh Layer'
                elif hidden_class == LinearLayer:
                    hidden_class_name = 'Linear Layer'
                else:
                    hidden_class_name = 'Sigmoid Layer'

                #Build Neural Network
                net = buildNetwork(number_of_features, # input
                       number_of_hidden_units, # number of hidden units
                       1, # output neurons
                       bias = True,
                       hiddenclass = hidden_class,
                       outclass = LinearLayer
                       )


                trainer = BackpropTrainer(net, ds, learningrate=0.01, lrdecay=1.0, momentum=0.0, weightdecay=0.0, verbose=True)

                trainer.trainUntilConvergence(maxEpochs=number_of_epochs)

                predictions = []
                for x in test_x:
                    predictions.append(net.activate(x)[0])

                predictions = np.array(predictions)

                # format output for LaTeX
                file.write('%d & %d & %s & %f & %f \\\\ \n' %
                           (number_of_epochs,
                            number_of_hidden_units,
                            hidden_class_name,
                            mean_squared_error(test_y, predictions)**0.5,
                            mean_absolute_error(test_y, predictions)))

    file.close()
