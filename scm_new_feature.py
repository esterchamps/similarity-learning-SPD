import pickle
import tensorflow as tf
import numpy as np
import os
from dataset_utils2 import load_data

MODEL_PATH = 'training_results_2\\evaluated_model_27022024_182439.h5'
DATASET_PATH = 'PREPROCESSING_PROCOTOL_2_CEDAR\\'
PATH_TO_SAVE_SCM_DICTS = 'SCM_NEW_FEATURES_CEDAR_PROTOCOL_2\\'
NUM_OF_WRITERS_TO_LOAD = 55
TOTAL_OF_WRITERS = 55
FIRST_WRITER = 1


def calculate_signature_covariance_matrix(feature_map):
    feature_map = feature_map.reshape(feature_map.shape[0], -1)
    S = feature_map.shape[1]
    mu = np.mean(feature_map, axis=1, keepdims=True)
    inner_summup = (feature_map - mu) @ (feature_map - mu).T
    covariance_matrix = (1 / (S - 1)) * inner_summup

    return covariance_matrix

def calculate_feature_map(intermediate_layer_model, signature, input1, input2):
    normalized_signature = signature
    intermediate_output = intermediate_layer_model([tf.expand_dims(normalized_signature, 0), input1, input2])
    intermediate_output_as_numpy = intermediate_output.numpy()
    intermediate_output_as_numpy = intermediate_output_as_numpy.squeeze()
    intermediate_output_as_numpy_transposed = np.transpose(intermediate_output_as_numpy, (2, 0, 1))

    return intermediate_output_as_numpy_transposed

def calculate_signature_covariance_matrices(data, intermediate_layer_model, input1, input2):
    dict_to_return = {}
    
    for individual_data in data:
        genuine_signatures_covariance_matrices = []
        for gen_signature in individual_data['genuine_signatures']:
            gen_feature_map = calculate_feature_map(intermediate_layer_model, gen_signature, input1, input2)
            gen_covariance_matrix = \
                calculate_signature_covariance_matrix(gen_feature_map)
            genuine_signatures_covariance_matrices.append(gen_covariance_matrix)

        forged_signatures_covariance_matrices = []
        for forg_signature in individual_data['forged_signatures']:
            forg_feature_map = calculate_feature_map(intermediate_layer_model, forg_signature, input1, input2)
            forg_covariance_matrix = \
                calculate_signature_covariance_matrix(forg_feature_map)
            forged_signatures_covariance_matrices.append(forg_covariance_matrix)
        
        individual_data['genuine_scm'] = genuine_signatures_covariance_matrices
        individual_data['forged_scm'] = forged_signatures_covariance_matrices
        dict_to_return[individual_data['individual']] = individual_data

    return dict_to_return

model = tf.keras.models.load_model(MODEL_PATH)
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)

input1 = np.ones((531,))
input2 = np.ones((1,))

if not os.path.exists(PATH_TO_SAVE_SCM_DICTS):
    os.mkdir(PATH_TO_SAVE_SCM_DICTS)

datasets = []

for writers in range(FIRST_WRITER, TOTAL_OF_WRITERS, NUM_OF_WRITERS_TO_LOAD):
    
    print()
    print('Indivíduo: '+str(writers))
    print()
    print('Carregando base de dados...')
    dataset = load_data(DATASET_PATH, NUM_OF_WRITERS_TO_LOAD, writers)
    print('Base de dados carregada.')
    print()
    print('Calculando matrizes de covariância...')
    scm_dataset = \
        calculate_signature_covariance_matrices(dataset, intermediate_layer_model, input1, input2)
    print('Matrizes de covariância calculadas.')
    print()
    print('Limpando...')

    for individual, individual_data in scm_dataset.items():
        individual_data.pop('genuine_signatures')
        individual_data.pop('forged_signatures')

    '''
    def transform(data_dict):
        data_dict.pop('genuine_signatures')
        data_dict.pop('forged_signatures')
        data_dict.pop('genuine_signatures_mask')
        data_dict.pop('forged_signatures_mask')
        return data_dict

    dataset = list(map(transform, dataset))
    '''

    print()
    print('Salvando...')

    filename = 'scm_cedar_protocol_2_new_features_1212_' + str(writers) + '-' + str(writers + NUM_OF_WRITERS_TO_LOAD - 1) + '.pkl'

    with open(os.path.join(PATH_TO_SAVE_SCM_DICTS, filename), 'wb') as pickle_file:
        pickle.dump(scm_dataset, pickle_file)

    print()
    print('Salvo.')
    print()

