import random
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from transformation_utils import apply_point_to_set_transformation, calculate_dij_and_its_eigenvalues_and_eigenvectors, calculate_Dij

NUM_GENUINE_PER_WRITER = 24
NUM_FORGED_PER_WRITER = 30

def get_badtch(num_elements, individuals_curr_fold, source_data, same_or_mixed, origin_list_1, origin_list_2):

    list_of_already_selected = []
    list_of_pairs = []

    first_pair_element_max_index = NUM_GENUINE_PER_WRITER if origin_list_1 == 'genuine_scm' else NUM_FORGED_PER_WRITER
    second_pair_element_max_index = NUM_GENUINE_PER_WRITER if origin_list_2 == 'genuine_scm' else NUM_FORGED_PER_WRITER

    if same_or_mixed == 'same':
        while len(list_of_pairs) < num_elements:
            random_individual = random.choice(individuals_curr_fold)
            first_pair_element = random.randint(0, first_pair_element_max_index - 1)
            second_pair_element = random.randint(0, second_pair_element_max_index - 1)

            while first_pair_element_max_index == second_pair_element_max_index and first_pair_element == second_pair_element:
                second_pair_element = random.randint(0, second_pair_element_max_index - 1)

            identifier = (random_individual, first_pair_element, second_pair_element)

            if identifier not in list_of_already_selected:
                list_of_already_selected.append(identifier)
                pair = (source_data[random_individual][origin_list_1][first_pair_element],
                        source_data[random_individual][origin_list_2][second_pair_element])
                list_of_pairs.append(pair)

    elif same_or_mixed == 'mixed':
        while len(list_of_pairs) < num_elements:
            random_individual_1 = random.choice(individuals_curr_fold)
            random_individual_2 = random.choice(individuals_curr_fold)
            
            while random_individual_1 == random_individual_2:
                random_individual_2 = random.choice(individuals_curr_fold)

            first_pair_element = random.randint(0, first_pair_element_max_index - 1)
            second_pair_element = random.randint(0, second_pair_element_max_index - 1)

            identifier = (random_individual_1, random_individual_2, first_pair_element, second_pair_element)

            if identifier not in list_of_already_selected:
                list_of_already_selected.append(identifier)
                pair = (source_data[random_individual_1][origin_list_1][first_pair_element],
                        source_data[random_individual_2][origin_list_2][second_pair_element])
                list_of_pairs.append(pair)
                
    return np.array(list_of_pairs)
