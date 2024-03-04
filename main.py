from itertools import product
import random
import pickle
import os
from wi_sv_model import SignatureVerificationOnSPDManifolds

COVARIANCE_MATRICES_PATH = 'SCM_NEW_FEATURES_12_12\\'
TRAINING_RESULT_PATH = 'TRAINING_RESULTS_CEDAR_PROTOCOL_2_12_12\\'
WRITERS_PER_FILE = 55
#directory_path_gen = 'C:\\Users\\ester\\faculdade\\tcc\\database_cedar\\signatures\\full_forg'
#directory_path_forg = 'C:\\Users\\ester\\faculdade\\tcc\\database_cedar\\signatures\\full_org'
#NUM_OF_WRITERS_TO_LOAD = 15
#m = 1
#p = 10
n = 12
epoch = 300

def load_scm(directory_path, num_of_files_to_load = None, num_of_writers_to_load = None):
    '''
    Parametros:
    - directory_path -> diretório com os arquivos pickle contendo as
        matrizes de covariância.
    - num_of_files_to_load -> quantidade de arquivos a serem carregados.
        Cada arquivo possui as matrizes de covariância de 200 indivíduos.
    '''
    count = 0
    data = {}

    if num_of_files_to_load != None:
        for _, _, files in os.walk(directory_path):

            for file in files:
                if count == num_of_files_to_load:
                    break
                
                try:
                    with open(os.path.join(directory_path, file), 'rb') as pickle_file:
                        print(file)
                        pickle_file.seek(0)
                        loaded_data = pickle.load(pickle_file)
                        data.update(loaded_data)

                except Exception as e:
                    continue

                count += 1
    
    elif num_of_writers_to_load != None:
        remaining = num_of_writers_to_load

        for _, _, files in os.walk(directory_path):
            while len(data.keys()) != num_of_writers_to_load:
                file = random.choice(files)

                with open(os.path.join(directory_path, file), 'rb') as pickle_file:
                    pickle_file.seek(0)

                    num_writers = WRITERS_PER_FILE if remaining - WRITERS_PER_FILE > 0 else remaining

                    loaded_data = pickle.load(pickle_file)
                    
                    writers = random.sample(list(loaded_data.items()), num_writers)

                    for writer in writers:
                        data.update({writer[0]: writer[1]})

                    remaining -= WRITERS_PER_FILE
                    
                    if remaining <= 0:
                        break

                count += 1
    
    return data

def split_dataset(data):
    individuals = list(data.keys())
    random.shuffle(individuals)
    total_individuals = len(data)
    folds = []

    for _ in range(0, 2):
        development_set = individuals[:(total_individuals // 2) - 1]
        test_set = individuals[(total_individuals // 2):]
        fold = (development_set, test_set)
        folds.append(fold)

    return folds

def main():

    if not os.path.exists(TRAINING_RESULT_PATH):
        os.mkdir(TRAINING_RESULT_PATH)

    num_of_writers = [55]#, 75, 100, 160]
    num_of_genuine_per_writer = [24]#, 15, 24, 24]
    num_of_forged_per_writer = [24]#, 16, 30, 30]
    m_p_values = [(1, 12)]#(1, 7), (1, 8), (1, 9), (1, 10), (2, 3), (2, 4), (2, 5), (3, 3)]#[(1, 48), (2, 24)]
    forgery_type = ['mixed', 'same']
    combinations = list(product(range(0, 1), m_p_values, forgery_type))

    for comb in combinations:
        writer_index = comb[0]
        m_value = comb[1][0]
        p_value = comb[1][1]
        current_forgery_type = comb[2]

        print('Inicio do treinamento.')
        print()
        print('Carregando base de dados...')
        dataset = load_scm(COVARIANCE_MATRICES_PATH, num_of_writers_to_load=num_of_writers[writer_index])
        print('Base de dados carregada.')
        print()
        print('Gerando repartições do dataset...')
        dataset_folds = split_dataset(dataset)
        print('Partições geradas.')
        print()
        print('Iniciando treinamento...')

        model =\
            SignatureVerificationOnSPDManifolds(epoch = epoch,
                                                max_iterations = epoch*30,
                                                writers_loaded = num_of_writers[writer_index],
                                                num_of_genuine_per_writer = num_of_genuine_per_writer[writer_index],
                                                num_of_forged_per_writer = num_of_forged_per_writer[writer_index],
                                                training_batch_size = 200,
                                                validation_batch_size = 200,
                                                learning_rate = 1e-3,
                                                m = m_value,
                                                p = p_value,
                                                n = n)
        
        model.set_data(dataset, dataset_folds, current_forgery_type)
        model.fit(TRAINING_RESULT_PATH)
        print('Concluído.')


if __name__ == "__main__":
    main()