from dataset_utils2 import apply_preprocessing, calculate_signature_covariance_matrices, load_data_cedar
import pickle

DATASET_PATH = 'preprocessed-images\\'
PATH_TO_SAVE_PREPROCESSED_IMAGES = 'TEST_PREPROCESSED_PROTOCOL_1_CEDAR\\'
directory_path_gen = 'C:\\Users\\ester\\faculdade\\tcc\\database_cedar\\signatures\\full_org'
directory_path_forg = 'C:\\Users\\ester\\faculdade\\tcc\\database_cedar\\signatures\\full_forg'

NUM_OF_WRITERS_TO_LOAD = 5
TOTAL_OF_WRITERS = 5
FIRST_WRITER = 1

print('Iniciando...')

for writers in range(FIRST_WRITER, TOTAL_OF_WRITERS, NUM_OF_WRITERS_TO_LOAD):
    
    print()
    print('Indivíduo: '+str(writers))
    print()
    print('Carregando base de dados...')
    dataset = load_data_cedar(directory_path_gen, directory_path_forg)
    print('Base de dados carregada.')
    print()
    print('Realizando pré-processamento...')
    apply_preprocessing(dataset, PATH_TO_SAVE_PREPROCESSED_IMAGES)
    print('Pré-processamento concluído.')
    print()
    print('Calculando matrizes de covariância...')
    scm_dataset = calculate_signature_covariance_matrices(dataset)
    print('Matrizes de covariância calculadas.')
    print()
    print('Limpando...')

    for individual, individual_data in scm_dataset.items():
        individual_data.pop('genuine_signatures')
        individual_data.pop('forged_signatures')
        individual_data.pop('genuine_signatures_mask')
        individual_data.pop('forged_signatures_mask')

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

    filename = 'scm_cedar_protocol_1_' + str(writers) + '-' + str(writers + NUM_OF_WRITERS_TO_LOAD - 1) + '.pkl'

    with open(filename, 'wb') as pickle_file:
        pickle.dump(scm_dataset, pickle_file)

    print()
    print('Salvo.')
    print()