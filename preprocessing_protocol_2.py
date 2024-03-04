import pickle
import cv2 as cv
import numpy as np
import os
from skimage.transform import resize
from covariance_matrix_utils import calculate_feature_map, calculate_signature_covariance_matrix
from dataset_utils2 import load_data_cedar

DATASET_PATH = 'PATHPATH\\'
PATH_TO_SAVE_PREPROCESSED_IMAGES = 'PREPROCESSING_PROCOTOL_2_CEDAR\\'
PATH_TO_SAVE_SCM_DICTS = 'SCM_PROCOTOL_2_CEDAR\\'
NUM_OF_WRITERS_TO_LOAD = 55
TOTAL_OF_WRITERS = 55
FIRST_WRITER = 1
GENUINE_SIGNATURES_PREFIX = 'c-'
FORGED_SIGNATURES_PREFIX = 'cf-'

directory_path_gen = 'C:\\Users\\ester\\faculdade\\tcc\\database_cedar\\signatures\\full_org'
directory_path_forg = 'C:\\Users\\ester\\faculdade\\tcc\\database_cedar\\signatures\\full_forg'

def load_grayscale_image(image_path):
    '''
    Carrega a imagem em escala de cinzas e devolve um ndarray.
    '''

    image = cv.imread(image_path)
    if (len(image.shape) == 3):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_array = np.array(gray_image)
    else:
        image_array = np.array(image)

    return image_array

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

def apply_centralized_resizing_preprocessing_and_calculate_scm(data, path_to_save_preprocessed_images):
    def preprocess(image_array):
        threshold, _ = cv.threshold(image_array, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        image_array[image_array > threshold] = 255
        inverted =  255.0 - image_array

        aspect_ratio = 15 / 22  # Target aspect ratio
        height, width = inverted.shape
        current_ratio = height / width

        closest_width = (round(width / 22) * 22) + 22
        closest_height = (round(height / 15) * 15) + 15

        image_to_resize = inverted

        if not np.allclose(aspect_ratio, current_ratio):
            padding_w = 0
            padding_h = 0

            if aspect_ratio * closest_width > height:

                padding_w = int((closest_width - width) / 2)
                padding_h = int((aspect_ratio * closest_width - height) / 2)
            
            elif closest_height / aspect_ratio > width:
                
                padding_w = int((closest_height / aspect_ratio - width) / 2)
                padding_h = int((closest_height - height) / 2)

            padded = np.pad(inverted, ((padding_h, padding_h), (padding_w, padding_w)), mode='constant', constant_values=0)
            image_to_resize = padded

        resized_image = resize(image_to_resize, (150, 220), anti_aliasing=True).astype(np.uint8)
        return resized_image

    if not os.path.exists(path_to_save_preprocessed_images):
        os.mkdir(path_to_save_preprocessed_images)

    dict_to_return = {}

    for individual_data in data:
        person_number = individual_data['individual']

        person_folder = os.path.join(path_to_save_preprocessed_images, str(person_number))

        if not os.path.exists(person_folder):
            os.mkdir(person_folder)

        genuine_signatures_covariance_matrices = []
        image_counter = 1
        for sig_genuine in individual_data['genuine_signatures']:
            preprocessed_image_genuine = preprocess(sig_genuine)

            cv.imwrite(os.path.join(person_folder, GENUINE_SIGNATURES_PREFIX + str(person_number) + '-' +str(image_counter) + '.jpg'), preprocessed_image_genuine)
            _, binary_mask_of_preprocessed_image_genuine = cv.threshold(preprocessed_image_genuine, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)

            feature_map_genuine = calculate_feature_map(preprocessed_image_genuine, binary_mask_of_preprocessed_image_genuine)
            covariance_matrix = calculate_signature_covariance_matrix(feature_map_genuine)
            genuine_signatures_covariance_matrices.append(covariance_matrix)

            image_counter+=1

        forged_signatures_covariance_matrices = []
        image_counter = 1
        for sig_forged in individual_data['forged_signatures']:
            preprocessed_image_forged = preprocess(sig_forged)

            cv.imwrite(os.path.join(person_folder, FORGED_SIGNATURES_PREFIX + str(person_number) + '-' +str(image_counter) + '.jpg'), preprocessed_image_forged)
            _, binary_mask_of_preprocessed_image_forged = cv.threshold(preprocessed_image_forged, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)

            feature_map_forged = calculate_feature_map(preprocessed_image_forged, binary_mask_of_preprocessed_image_forged)
            covariance_matrix = calculate_signature_covariance_matrix(feature_map_forged)
            forged_signatures_covariance_matrices.append(covariance_matrix)

            image_counter+=1
        
        individual_data['genuine_scm'] = genuine_signatures_covariance_matrices
        individual_data['forged_scm'] = forged_signatures_covariance_matrices

        dict_to_return[person_number] = individual_data

    return dict_to_return

print('Iniciando...')

if not os.path.exists(PATH_TO_SAVE_SCM_DICTS):
    os.mkdir(PATH_TO_SAVE_SCM_DICTS)

for writers in range(FIRST_WRITER, TOTAL_OF_WRITERS, NUM_OF_WRITERS_TO_LOAD):
    
    print()
    print('Indivíduo: '+str(writers))
    print()
    print('Carregando base de dados...')
    #dataset = load_data(DATASET_PATH, NUM_OF_WRITERS_TO_LOAD, writers)
    dataset = load_data_cedar(directory_path_gen, directory_path_forg)
    print('Base de dados carregada.')
    print()
    print('Realizando pré-processamento...')
    scm_dataset = apply_centralized_resizing_preprocessing_and_calculate_scm(dataset, PATH_TO_SAVE_PREPROCESSED_IMAGES)
    print('Pré-processamento concluído.')
    print('Matrizes de covariância calculadas.')
    print()
    print('Salvando...')

    for individual, individual_data in scm_dataset.items():
        individual_data.pop('genuine_signatures')
        individual_data.pop('forged_signatures')

    filename = 'scm_cedar_protocol_2_' + str(writers) + '_' + str(writers + NUM_OF_WRITERS_TO_LOAD - 1) + '.pkl'

    with open(os.path.join(PATH_TO_SAVE_SCM_DICTS, filename), 'wb') as pickle_file:
        pickle.dump(scm_dataset, pickle_file)

    print()
    print('Salvo.')
    print()
