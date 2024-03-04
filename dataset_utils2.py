import os
import random
import pickle
import cv2 as cv
from image_utils import load_grayscale_image, otsu_thresholding, thin_operation, get_most_optimal_thinning_level
from covariance_matrix_utils import calculate_signature_covariance_matrix, calculate_feature_map


GENUINE_SIGNATURES_PREFIX = 'c-'
FORGED_SIGNATURES_PREFIX = 'cf-'

GENUINE_SIGNATURES_PREFIX = 'c-'
FORGED_SIGNATURES_PREFIX = 'cf-'

TRAINING_SET_RATIO = 0.7
VALIDATION_SET_RATIO = 0.2

NUM_GENUINE_PER_WRITER = 24
NUM_FORGED_PER_WRITER = 30
NUM_TOTAL_SCM_FILES = 20
WRITERS_PER_FILE = 15

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

def load_data_cedar(directory_path_gen, directory_path_forg):
    data = []
    genuine_sigs = os.listdir(directory_path_gen)
    forgery_sigs = os.listdir(directory_path_forg)

    genuine_sigs.pop()
    forgery_sigs.pop()

    #for gen, forg in zip(genuine_sigs, forgery_sigs):

    for writer in range(1, 56):
        
        curr_data = {
            'individual': writer,
            'genuine_signatures': [],
            'forged_signatures': []
        }

        for sig_index in range(1, 25):
            curr_data['genuine_signatures'].append(load_grayscale_image(os.path.join(directory_path_gen, 'original_'+str(writer)+'_'+str(sig_index)+'.png')))
            curr_data['forged_signatures'].append(load_grayscale_image(os.path.join(directory_path_forg, 'forgeries_'+str(writer)+'_'+str(sig_index)+'.png')))

        data.append(curr_data)
    
    return data

def load_data(directory_path, num_of_individuals, first_individual=1):
    data = []
    individuals_counter = 0

    for subdir, _, files in os.walk(directory_path):
        if subdir == directory_path:
            continue

        individual = int(os.path.basename(subdir))

        if individual >= first_individual and individual < (first_individual + num_of_individuals):
            genuine_signatures = [
                load_grayscale_image(os.path.join(subdir, file)) for file in files if file.startswith(GENUINE_SIGNATURES_PREFIX)
            ]
            #directory_path antes de subdir para paths completos
            forged_signatures = [
                load_grayscale_image(os.path.join(subdir, file)) for file in files if file.startswith(FORGED_SIGNATURES_PREFIX)
            ]
            
            data.append({
                'individual': individual,
                'genuine_signatures': genuine_signatures,
                'forged_signatures': forged_signatures
            })

            individuals_counter += 1

            if individuals_counter == num_of_individuals:
                break
    
    return data

def apply_more_simple_preprocessing(data):
    for individual_data in data:

        preprocessed_genuine_signatures = []
        for sig in individual_data['genuine_signatures']:
            otsu_sig = otsu_thresholding(sig)
            preprocessed_genuine_signatures.append(otsu_sig)

        preprocessed_forged_signatures = []
        for sig in individual_data['forged_signatures']:
            otsu_sig = otsu_thresholding(sig)
            preprocessed_forged_signatures.append(otsu_sig)

        individual_data['genuine_signatures_mask'] = preprocessed_genuine_signatures
        individual_data['forged_signatures_mask'] = preprocessed_forged_signatures

def get_binary_masks(data):
    for individual_data in data:

        preprocessed_genuine_signatures = []
        for sig in individual_data['genuine_signatures']:
            otsu_sig = otsu_thresholding(sig)
            preprocessed_genuine_signatures.append(otsu_sig)

        preprocessed_forged_signatures = []
        for sig in individual_data['forged_signatures']:
            otsu_sig = otsu_thresholding(sig)
            preprocessed_forged_signatures.append(otsu_sig)

        individual_data['genuine_signatures_mask'] = preprocessed_genuine_signatures
        individual_data['forged_signatures_mask'] = preprocessed_forged_signatures

def apply_centralized_resizing_preprocessing(data):
    for individual_data in data:

        preprocessed_genuine_signatures = []
        for sig in individual_data['genuine_signatures']:
            otsu_sig = otsu_thresholding(sig)
            preprocessed_genuine_signatures.append(otsu_sig)

        preprocessed_forged_signatures = []
        for sig in individual_data['forged_signatures']:
            otsu_sig = otsu_thresholding(sig)
            preprocessed_forged_signatures.append(otsu_sig)

        individual_data['genuine_signatures_mask'] = preprocessed_genuine_signatures
        individual_data['forged_signatures_mask'] = preprocessed_forged_signatures

def apply_preprocessing(data, path_to_save_preprocessed_images):
    '''
    Aplica o pré-processamento nas imagens de assinaturas.
    Aplica limiarização de Otsu + erosão.
    '''

    if not os.path.exists(path_to_save_preprocessed_images):
        os.mkdir(path_to_save_preprocessed_images)

    for individual_data in data:
        person_number = individual_data['individual']
        person_folder = os.path.join(path_to_save_preprocessed_images, str(person_number))

        if not os.path.exists(person_folder):
            os.mkdir(person_folder)

        preprocessed_genuine_signatures = []
        thresholding_genuine = []
        for sig in individual_data['genuine_signatures']:
            thres, otsu_sig = otsu_thresholding(sig)
            thresholding_genuine.append(thres)
            preprocessed_genuine_signatures.append(otsu_sig)

        preprocessed_forged_signatures = []
        thresholding_forged = []
        for sig in individual_data['forged_signatures']:
            thres, otsu_sig = otsu_thresholding(sig)
            thresholding_forged.append(thres)
            preprocessed_forged_signatures.append(otsu_sig)

        motl = get_most_optimal_thinning_level(preprocessed_genuine_signatures)
        individual_data['motl'] = round(motl)

        individual_data['genuine_signatures_mask'] = \
            list(map(lambda sig: thin_operation(sig, individual_data['motl']), preprocessed_genuine_signatures))
        
        individual_data['forged_signatures_mask'] = \
            list(map(lambda sig: thin_operation(sig, individual_data['motl']), preprocessed_forged_signatures))
        
        img_counter = 1
        for gen_sig, gen_sig_mask in zip(individual_data['genuine_signatures'], individual_data['genuine_signatures_mask']):
            cv.imwrite(os.path.join(person_folder, GENUINE_SIGNATURES_PREFIX + str(individual_data['individual']) + '-' +str(img_counter) + '.jpg'), gen_sig * gen_sig_mask)
            img_counter+=1
        
        img_counter = 1
        for fog_sig, fog_sig_mask in zip(individual_data['forged_signatures'], individual_data['forged_signatures_mask']):
            cv.imwrite(os.path.join(person_folder, FORGED_SIGNATURES_PREFIX + str(individual_data['individual']) + '-' +str(img_counter) + '.jpg'), fog_sig * fog_sig_mask)
            img_counter+=1

def calculate_signature_covariance_matrices(data):
    '''
    Calcula as matrizes de covariância para todas as assinaturas do conjunto
    de dados carregado.
    '''

    dict_to_return = {}
    
    for individual_data in data:

        genuine_signatures_covariance_matrices = []
        for signature, signature_mask in zip(individual_data['genuine_signatures'], individual_data['genuine_signatures_mask']):
            feature_map = calculate_feature_map(signature, signature_mask)
            covariance_matrix = calculate_signature_covariance_matrix(feature_map)
            genuine_signatures_covariance_matrices.append(covariance_matrix)

        forged_signatures_covariance_matrices = []
        for signature, signature_mask in zip(individual_data['forged_signatures'], individual_data['forged_signatures_mask']):
            feature_map = calculate_feature_map(signature, signature_mask)
            covariance_matrix = calculate_signature_covariance_matrix(feature_map)
            forged_signatures_covariance_matrices.append(covariance_matrix)
        
        individual_data['genuine_scm'] = genuine_signatures_covariance_matrices
        individual_data['forged_scm'] = forged_signatures_covariance_matrices

        dict_to_return[individual_data['individual']] = individual_data

    return dict_to_return

def create_dicts(data, forgery_type):
    similar_pairs_dict = {}
    disimilar_pairs_dict = {}

    for individual, individual_data in data.items():
        person_number = int(individual)

        for i in range(NUM_GENUINE_PER_WRITER):
            for j in range(i + 1, NUM_GENUINE_PER_WRITER):
                if i != j:
                    key = (person_number, person_number)
                    value = (i, j)
                    similar_pairs_dict.setdefault(key, set()).add(value)

        if forgery_type == 'genuine_genuine_mixed_individuals':
            for _, other_d in data.items():
                if other_d['individual'] != person_number and (other_d['individual'], person_number) not in disimilar_pairs_dict:
                    for genuine_index in range(NUM_GENUINE_PER_WRITER):
                        for other_genuine_index in range(NUM_GENUINE_PER_WRITER):
                            key = (person_number, other_d['individual'])
                            value = (genuine_index, other_genuine_index)
                            disimilar_pairs_dict.setdefault(key, set()).add(value)
        
        elif forgery_type == 'genuine_forged_mixed_individuals':
            for _, other_d in data.items():
                if other_d['individual'] != person_number:
                    for genuine_index in range(NUM_GENUINE_PER_WRITER):
                        for forged_index in range(NUM_FORGED_PER_WRITER):
                            key = (person_number, other_d['individual'])
                            value = (genuine_index, forged_index)
                            disimilar_pairs_dict.setdefault(key, set()).add(value)

        elif forgery_type == 'genuine_forged_same_individuals':
            for genuine_index in range(NUM_GENUINE_PER_WRITER):
                for forged_index in range(NUM_FORGED_PER_WRITER):
                    key = (person_number, person_number)
                    value = (genuine_index, forged_index)
                    disimilar_pairs_dict.setdefault(key, set()).add(value)

    for key, value in similar_pairs_dict.items():
        similar_pairs_dict[key] = list(similar_pairs_dict[key])

    for key, value in disimilar_pairs_dict.items():
        disimilar_pairs_dict[key] = list(disimilar_pairs_dict[key])
    
    return similar_pairs_dict, disimilar_pairs_dict


def split_dataset(data, num_folds):
        
    individuals = list(data.keys())
    random.shuffle(individuals)
    part_size = len(individuals) // num_folds
    remainder = len(individuals) % num_folds
    folds = [individuals[i * part_size + min(i, remainder):(i + 1) * part_size + min(i + 1, remainder)] for i in range(num_folds)]
    
    #dicts_to_return = []

    #for fold in folds:
    #    individuals_data = {d : data.get(d) for d in fold}
    #    similar_pairs_dict, disimilar_pairs_dict = create_dicts(individuals_data, forgery_type)
    #    dicts_to_return.append((similar_pairs_dict, disimilar_pairs_dict))
    
    return folds

    training_individuals_number = random.sample(individuals, int(len(data) * TRAINING_SET_RATIO))
    training_individuals = {d : data.get(d) for d in training_individuals_number}
    remaining_individuals = {d : data.get(d) for d in data.keys() if not d in training_individuals_number}

    validation_individuals_number = random.sample(list(remaining_individuals.keys()), int(len(data) * VALIDATION_SET_RATIO))
    validation_individuals = {d : data.get(d) for d in validation_individuals_number}

    test_individuals = {d : data.get(d) for d in remaining_individuals.keys() if not d in validation_individuals_number}

    similar_pairs_training_dict, disimilar_pairs_training_dict = create_dicts(training_individuals, forgery_type)
    similar_pairs_validation_dict, disimilar_pairs_validation_dict = create_dicts(validation_individuals, forgery_type)
    similar_pairs_test_dict, disimilar_pairs_test_dict = create_dicts(test_individuals, forgery_type)

    return similar_pairs_training_dict, disimilar_pairs_training_dict, \
        similar_pairs_validation_dict, disimilar_pairs_validation_dict, \
        similar_pairs_test_dict, disimilar_pairs_test_dict