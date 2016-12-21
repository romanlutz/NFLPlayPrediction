import random

from machine_learning.classification import compare_classification_parameters
from machine_learning.neural_network_prediction import neural_network_prediction
from machine_learning.regression import compute_regression_results
from preprocessing.analysis import apply_pca, apply_kernel_pca, apply_anova_f_value_test, \
     apply_variance_threshold_selection, plot_progress_measure
from preprocessing.features import extract_features

random.seed(0)  # keep seed fixed for reproducibility


def use_classification(data, config, target_name='success'):
    compare_classification_parameters(data['categorical_features'], data[target_name], config)

def use_regression(data, config, target_name='progress'):
    compute_regression_results(data['categorical_features'], data[target_name], target_name, config)

    #features, labels, yards, progress = get_team_features("NO", features, labels, "team")
    #compute_regression_results(features, yards, "./regression_results_team_NO_yards")
    #compute_regression_results(features, progress, "./regression_results_team_NO_progress")
    
def use_neural_networks(data, config, measure='progress', layer_type='tanh'):
    neural_network_prediction(data=data,
                              config=config,
                              team='all',
                              measure=measure,
                              load_previous=True,
                              layer_type=layer_type)

def __main__():
    config = {
        'start_year': 2009,
        'end_year': 2014,

        'predict_yards': True,
        'predict_progress': True,
        'predict_success': True,

        'prediction_method_success': 'classification',
        'prediction_method_yards': 'regression',
        'prediction_method_progress': 'regression',

        'use_neural_networks': True,
        'neural_net_config': {
            'k_fold': 5,
            'epochs': [1],
            'hidden_layers': [1],
            'hidden_units': [1],
            'load_previous': True,
            'tanh': True,
            'sigmoid': True,
            'linear': True
        },

        'grid_search': False
    }

    print config
    print 'getting data'

    data = extract_features(start_year=config['start_year'], end_year=config['end_year'])

    print 'finished getting data'
    print 'starting prediction'

    for measure in ['success', 'yards', 'progress']:
        if config['predict_%s' % measure]:
            print 'predicting %s measure' % measure
            if config['prediction_method_%s' % measure] == 'regression':
                use_regression(data, config, measure)
            else:
                use_classification(data, config, measure)

            if config['use_neural_networks']:
                for layer_type in ['tanh', 'sigmoid', 'linear']:
                    if config['neural_net_config'][layer_type]:
                        use_neural_networks(data, config, measure=measure, layer_type=layer_type)

    apply_pca(data['categorical_features'])
    apply_kernel_pca(data['categorical_features'], data['success'])
    apply_anova_f_value_test(data['categorical_features'], data['success'], data['encoder'])
    apply_variance_threshold_selection(data['categorical_features'], data['success'], data['encoder'])
    plot_progress_measure()



__main__()
