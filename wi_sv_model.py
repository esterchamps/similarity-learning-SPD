import random
import numpy as np
import os
import pickle
from datetime import datetime
from itertools import product
from sklearn.metrics import roc_curve, roc_auc_score
from transformation_utils import apply_point_to_set_transformation, calculate_dij_and_its_eigenvalues_and_eigenvectors, calculate_Dij
from gradient_utils import calculate_dL_dA, calculate_dL_dW, calculate_dL_dWR, update_W_parameter, calculate_dL_dM, calculate_dL_dMR, update_M_parameter

class SignatureVerificationOnSPDManifolds(object):

    def __init__(
                self,
                n = 10,
                m = 2,
                p = 5,
                writers_loaded = 0,
                num_of_genuine_per_writer = 24,
                num_of_forged_per_writer = 30,
                percent_of_genuine_signatures = 0.7,
                percent_of_forged_signatures = 0.7,
                learning_rate = 10e-4,
                max_iterations = 1000,
                epoch = 100,
                training_batch_size = 200,
                validation_batch_size = 200,
                zeta_s = 1,
                zeta_d = 20.0,
                sci = 0.01):

        self.n = n
        self.m = m
        self.p = p
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.training_batch_size = training_batch_size
        self.val_batch_size = validation_batch_size
        self.zeta_s = zeta_s
        self.zeta_d = zeta_d
        self.sci = sci
        self.num_of_genuine_per_writer = num_of_genuine_per_writer
        self.num_of_forged_per_writer = num_of_forged_per_writer
        self.percent_of_genuine_signatures = percent_of_genuine_signatures
        self.percent_of_forged_signatures = percent_of_forged_signatures
        self.writers_loaded = writers_loaded
        self.max_iterations = max_iterations

        self.init_learnable_parameters()
        self.cleaning_global_parameters()
        self.cross_validation_completed = False
        self.fit_results = {}

    def init_learnable_parameters(self):
        self.A = np.random.rand(self.m, 2)
        self.M = np.eye(self.m)
        self.M0 = np.eye(self.m)
        self.W = np.zeros((self.n, self.m*self.p))
        for j in range(self.m):
            Q = np.random.randn(self.n, self.p)
            for i in range(j):
                Q -= np.dot(self.W[:, i*self.p:(i+1)*self.p], np.dot(self.W[:, i*self.p:(i+1)*self.p].T, Q))
            Q, _ = np.linalg.qr(Q)
            self.W[:, j*self.p:(j+1)*self.p] = Q

    def cleaning_global_parameters(self):
        self.aucs = []
        self.eers = []
        self.training_loss = []
        self.validation_loss = []
        self.minimum_loss = 10e+5
        self.best_A = np.zeros((self.m, 2))
        self.best_W = np.zeros((self.m, 2))
        self.best_M = np.zeros((self.m, self.m))

    def get_next_fold(self):
        '''
        Altera o conjunto de validacao, atribuindo a ele o proximo fold.
        Os folds restantes sao unidos para formar o proximo conjunto de treinamento.
        '''
        self.test_fold = self.dataset_folds[self.next_fold][1]

        development_fold_size = len(self.dataset_folds[self.next_fold][0])

        self.val_fold = self.dataset_folds[self.next_fold][0][:int(development_fold_size*0.7)]
        #self.curr_s_pairs_val, self.curr_d_pairs_val = self.splitted_dataset_indexes[self.next_val_fold]
        #self.
        train_fold = self.dataset_folds[self.next_fold][0][int(development_fold_size*0.7):]

        #self.curr_s_pairs_train = {}
        #self.curr_d_pairs_train = {}

        # for split_count in range(len(self.dataset_folds)):
        #     if split_count != self.next_fold:
        #         train_fold.extend(self.dataset_folds[split_count])
        
        self.train_data = {d : self.datasource.get(d) for d in train_fold}

        self.curr_s_train_dict, self.curr_d_train_dict = self.create_dicts(self.train_data)

        self.next_fold += 1
        if self.next_fold == len(self.dataset_folds):
            self.cross_validation_completed = True
        
        self.end_current_fold_training = False

    def set_data(
                self,
                datasource,
                folds,
                mixed_or_same_in_d_pairs):
        '''
        datasource: dicionario contendo as matrizes de covariancia, separadas
        por indivíduo.

        splitted_dataset_indexes: lista de tuplas. O primeiro elemento da tupla
        eh um dicionario contendo indices de pares similares, o segundo elemento
        da tupla eh um dicionario contendo indices de pares dissimilares. Cada
        tupla representa um fold da validacao cruzada.

        mixed_or_same_in_d_pairs: pode ser 'mixed' ou 'same'. Informa se os
        pares dissimilares sao formados por assinaturas de um mesmo individuo
        ou se sao formados por assinaturas de individuos diferentes.
        '''
        self.datasource = datasource
        
        # cria um dicionario para armazenar o resultado dos treinamentos
        # o numero de treinamentos sera a quantidade de folds restantes apos
        # separacao do conjunto de testes (necessario para que cada fold
        # seja o conjunto de validacao 1 vez)
        self.fit_results = {training_number : {} for training_number in range(1, len(folds) + 1)}

        self.dataset_folds = folds
        self.next_fold = 0

        self.mixed_or_same_in_s_pairs = 'same'
        self.mixed_or_same_in_d_pairs = mixed_or_same_in_d_pairs

        self.src_lst_1 = 'genuine_scm'
        self.src_lst_2 = 'forged_scm' if self.mixed_or_same_in_d_pairs == 'same' else 'genuine_scm'
    
    def run_validation(self, similar_training_batch, disimilar_training_batch, num_iterations):
        s_val = self.get_batch(self.val_batch_size, self.val_fold, self.mixed_or_same_in_s_pairs, self.src_lst_1, self.src_lst_1)
        d_val = self.get_batch(self.val_batch_size, self.val_fold, self.mixed_or_same_in_d_pairs, self.src_lst_1, self.src_lst_2)

        s_pairs_of_sets_val = apply_point_to_set_transformation(s_val, self.W, self.m, self.p)
        d_pairs_of_sets_val = apply_point_to_set_transformation(d_val, self.W, self.m, self.p)

        S_dij_val, _, _ = calculate_dij_and_its_eigenvalues_and_eigenvectors(s_pairs_of_sets_val, self.A)
        D_dij_val, _, _ = calculate_dij_and_its_eigenvalues_and_eigenvectors(d_pairs_of_sets_val, self.A)

        S_Dij_val = calculate_Dij(S_dij_val, self.M)
        D_Dij_val = calculate_Dij(D_dij_val, self.M)
        total_val = np.concatenate((S_Dij_val, D_Dij_val))
        labels_val = np.concatenate((np.ones_like(S_Dij_val), np.zeros_like(D_Dij_val)))

        self.validation_loss.append(self.loss_function(s_val, d_val, S_Dij_val, D_Dij_val))
        print('loss validation: ' + str(self.validation_loss[-1]))

        probabilities = np.zeros_like(total_val, dtype=float)
        probabilities[total_val < self.zeta_s] = 1.0
        probabilities[total_val > self.zeta_d] = 0.0
        in_between_mask = (total_val >= self.zeta_s) & (total_val <= self.zeta_d)
        probabilities[in_between_mask] = 1 - ((total_val[in_between_mask] - self.zeta_s) / (self.zeta_d - self.zeta_s))

        auc = roc_auc_score(labels_val, probabilities)
        continue_training = False
        #self.aucs.append(auc)

        fpr, tpr, _ = roc_curve(labels_val, probabilities)
        fnr = 1 - tpr
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        self.eers.append(EER)

        if auc < max(self.aucs, default=1) and num_iterations < self.max_iterations:
            self.aucs.append(auc)
            new_similar_training_batch = self.get_batch_from_dict(self.training_batch_size, self.curr_s_train_dict, self.mixed_or_same_in_s_pairs , self.src_lst_1, self.src_lst_1)
            new_disimilar_training_batch = self.get_batch_from_dict(self.training_batch_size, self.curr_d_train_dict, self.mixed_or_same_in_d_pairs, self.src_lst_1, self.src_lst_2)
            continue_training = True
            return auc, new_similar_training_batch, new_disimilar_training_batch, continue_training

        elif auc >= min(self.aucs, default=0) and num_iterations > self.max_iterations:
            self.aucs.append(auc)
            return auc, similar_training_batch, disimilar_training_batch, False

        else:
            self.aucs.append(auc)
            return auc, similar_training_batch, disimilar_training_batch, True
        
    def fit(self, result_path):
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M%S")

        while not self.cross_validation_completed:
            self.get_next_fold()
            print('Fold: '+str(self.next_fold))

            s_pairs = self.get_batch_from_dict(self.training_batch_size, self.curr_s_train_dict, self.mixed_or_same_in_s_pairs , self.src_lst_1, self.src_lst_1)
            d_pairs = self.get_batch_from_dict(self.training_batch_size, self.curr_d_train_dict, self.mixed_or_same_in_d_pairs , self.src_lst_1, self.src_lst_2)

            self.init_learnable_parameters()
            self.cleaning_global_parameters()

            iteration = 0
            restarts = 0

            while restarts < 3:
                iteration += 1
                try:
                    s_pairs_of_sets = apply_point_to_set_transformation(s_pairs, self.W, self.m, self.p)
                    d_pairs_of_sets = apply_point_to_set_transformation(d_pairs, self.W, self.m, self.p)

                    S_dij, S_dij_eigenvalues, S_dij_eigenvectors = calculate_dij_and_its_eigenvalues_and_eigenvectors(s_pairs_of_sets, self.A)
                    D_dij, D_dij_eigenvalues, D_dij_eigenvectors = calculate_dij_and_its_eigenvalues_and_eigenvectors(d_pairs_of_sets, self.A)

                    S_Dij = calculate_Dij(S_dij, self.M)
                    D_Dij = calculate_Dij(D_dij, self.M)

                    MT_plus_M = self.M.T + self.M

                    dL_dA = calculate_dL_dA(s_pairs, d_pairs, S_dij_eigenvalues, D_dij_eigenvalues, S_dij, D_dij, S_Dij, D_Dij, self.A, self.zeta_s, self.zeta_d, MT_plus_M)
                    self.A = self.A - self.learning_rate * dL_dA

                    dL_dW = calculate_dL_dW(s_pairs, d_pairs, s_pairs_of_sets, d_pairs_of_sets, S_dij, D_dij, S_Dij, D_Dij, S_dij_eigenvalues, D_dij_eigenvalues, S_dij_eigenvectors, D_dij_eigenvectors, self.zeta_s, self.zeta_d, MT_plus_M, self.W, self.A, self.m, self.p)
                    dL_dWR = calculate_dL_dWR(dL_dW, self.W)
                    self.W = update_W_parameter(self.W, self.learning_rate, dL_dWR)

                    dL_dM = calculate_dL_dM(s_pairs, d_pairs, S_dij, D_dij, S_Dij, D_Dij, self.M, self.M0, self.zeta_s, self.zeta_d, self.sci)
                    dL_dMR = calculate_dL_dMR(dL_dM, self.M)
                    self.M = update_M_parameter(self.M, self.learning_rate, dL_dMR)                        
                    
                    #print('loss training: ' + str(self.loss_function(s_pairs, d_pairs, S_Dij, D_Dij)))

                except Exception as e:
                    print('Ocorreu uma excecao. Reiniciando...')
                    print(e)
                    self.init_learnable_parameters()
                    iteration = 0
                    restarts += 1
                    continue

                if iteration % self.epoch == 0:

                    self.training_loss.append(self.loss_function(s_pairs, d_pairs, S_Dij, D_Dij))
                    print('loss training: ' + str(self.training_loss[-1]), end=', ')

                    auc, s_pairs, d_pairs, is_to_continue = \
                        self.run_validation(s_pairs, d_pairs, iteration)

                    if auc == max(self.aucs):
                        self.best_A = np.copy(self.A)
                        self.best_M = np.copy(self.M)
                        self.best_W = np.copy(self.W)

                    if not is_to_continue:
                        break
            
            best_auc = max(self.aucs)
            best_eer = self.eers[self.aucs.index(best_auc)]
            print('Maior AUC: '+str(best_auc))
            print('EER equivalente (validacao): '+str(best_eer))

            EER_test = self.run_test()

            print('EER (teste): '+str(EER_test))

            self.fit_results[self.next_fold] = {
                'W': self.best_W,
                'M': self.best_M,
                'A': self.best_A,
                'aucs': self.aucs,
                'val_eers': self.eers,
                'train_loss': self.training_loss,
                'val_loss': self.validation_loss,
                'test_eer': EER_test,
                'test_users': self.test_fold
            }

        filename = 'forgery_' + self.mixed_or_same_in_d_pairs + '_individuals' \
                   + '_n' + str(self.n) + '_m' + str(self.m) + '_p' + str(self.p) \
                   + '_writers' + str(self.writers_loaded) \
                   + '_epoch' + str(self.epoch) + '_max_iterations' + str(self.max_iterations) \
                   + '_batch' + str(self.training_batch_size + self.val_batch_size) \
                   + '_' + dt_string + '.pkl'
        
        to_save = {
            'fold_results': self.fit_results,
            'n': self.n,
            'm': self.m,
            'p': self.p,
            'zeta_s': self.zeta_s,
            'zeta_d': self.zeta_d,
            'sci': self.sci,
            'same_or_mixed_individuals_in_forgery': self.mixed_or_same_in_d_pairs,
            'genuine_per_writer': self.num_of_genuine_per_writer,
            'forged_per_writer': self.num_of_forged_per_writer,
            'writers': self.writers_loaded,
            'epoch': self.epoch,
            'max_iterations': self.max_iterations,
            'training_batch_size': self.training_batch_size,
            'validation_batch_size': self.val_batch_size,
            'learning_rate': self.learning_rate
        }

        with open(os.path.join(result_path, filename), 'wb') as pickle_file:
            pickle.dump(to_save, pickle_file)

    def run_test(self):
        W = self.best_W
        A = self.best_A
        M = self.best_M

        # first approach
        self.percent_of_genuine_signatures = 1.0
        self.percent_of_forged_signatures = 1.0

        self.test_data = {d : self.datasource.get(d) for d in self.test_fold}

        s_test_dict, d_test_dict = self.create_dicts(self.test_data)
        
        s_test = self.get_batch_from_dict(10e+6, s_test_dict, self.mixed_or_same_in_s_pairs, self.src_lst_1, self.src_lst_1, is_test=True)
        d_test = self.get_batch_from_dict(10e+6, d_test_dict, self.mixed_or_same_in_d_pairs, self.src_lst_1, self.src_lst_2, is_test=True)

        s_pairs_of_sets_test = apply_point_to_set_transformation(s_test, W, self.m, self.p)
        d_pairs_of_sets_test = apply_point_to_set_transformation(d_test, W, self.m, self.p)

        S_dij_test, _, _ = calculate_dij_and_its_eigenvalues_and_eigenvectors(s_pairs_of_sets_test, A)
        D_dij_test, _, _ = calculate_dij_and_its_eigenvalues_and_eigenvectors(d_pairs_of_sets_test, A)

        S_Dij_test = calculate_Dij(S_dij_test, M)
        D_Dij_test = calculate_Dij(D_dij_test, M)
        total_test = np.concatenate((S_Dij_test, D_Dij_test))
        labels_test = np.concatenate((np.ones_like(S_Dij_test), np.zeros_like(D_Dij_test)))

        probabilities = np.zeros_like(total_test, dtype=float)
        probabilities[total_test < self.zeta_s] = 1.0
        probabilities[total_test > self.zeta_d] = 0.0
        in_between_mask = (total_test >= self.zeta_s) & (total_test <= self.zeta_d)
        probabilities[in_between_mask] = 1 - ((total_test[in_between_mask] - self.zeta_s) / (self.zeta_d - self.zeta_s))

        fpr, tpr, _ = roc_curve(labels_test, probabilities)
        fnr = 1 - tpr
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        return EER

    def loss_function(self, S, D, distances_S, distances_D):
        """
        Loss function.
        Receives:
        A list of similar pairs (S);
        A list of dissimilar pairs (D);
        A list of already calculated set-to-set distances (Dij) between each pair in S, with same shape as S (distances_S);
        A list of already calculated set-to-set distances (Dij) between each pair in D, with same shape as D (distances_D);
        A threshold value for distances of sets from same class (zeta_s);
        A threshold value for distances of sets from different classes (zeta_d);
        Learnable parameter M (m x m);
        The identity matrix Im (m x m);
        The length m (m);
        A regularization scaling factor (sci).
        """

        L_s = distances_S - self.zeta_s
        L_s[L_s < 0] = 0
        L_s = L_s ** 2

        L_d = self.zeta_d - distances_D
        L_d[L_d < 0] = 0
        L_d = L_d ** 2

        L = (1 / len(S)) * L_s + (1 / len(D)) * L_d

        m_times_m0_inverse = self.M @ np.linalg.inv(self.M0)
        regularization = np.trace(m_times_m0_inverse) - np.log(np.linalg.det(m_times_m0_inverse)) - self.m

        L += self.sci * regularization

        return np.sum(L)
    
    def get_batch_from_dict(self, num_elements, source_pairs_dict, same_or_mixed, origin_list_1, origin_list_2, is_test=False):

        list_of_pairs = []

        if same_or_mixed == 'same':
            while len(list_of_pairs) < num_elements:
                if len(source_pairs_dict.keys()) > 0:
                    random_key = random.choice(list(source_pairs_dict.keys()))
                    pairs = source_pairs_dict[random_key]
                    pair_index = random.randint(0, len(pairs) - 1)
                    pair_with_indexes = pairs[pair_index]
                    pair = (self.datasource[random_key[0]][origin_list_1][pair_with_indexes[0]],
                            self.datasource[random_key[0]][origin_list_2][pair_with_indexes[1]])
                    list_of_pairs.append(pair)

                    if is_test:
                        pairs.pop(pair_index)

                        if len(pairs) == 0:
                            source_pairs_dict.pop(random_key)
                else:
                    break

        elif same_or_mixed == 'mixed':
            while len(list_of_pairs) < num_elements:
                if len(source_pairs_dict.keys()) > 0:
                    random_key = random.choice(list(source_pairs_dict.keys()))
                    pairs = source_pairs_dict[random_key]
                    pair_index = random.randint(0, len(pairs) - 1)
                    pair_with_indexes = pairs[pair_index]
                    pair = (self.datasource[random_key[0]][origin_list_1][pair_with_indexes[0]],
                            self.datasource[random_key[1]][origin_list_2][pair_with_indexes[1]])
                    list_of_pairs.append(pair)
                    if is_test:
                        pairs.pop(pair_index)

                        if len(pairs) == 0:
                            source_pairs_dict.pop(random_key)
                else:
                    break
        
        #if len(list_of_pairs) < num_elements:
        #    self.end_current_fold_training = True

        return np.array(list_of_pairs)
    
    def get_batch(self, num_elements, individuals_curr_fold, same_or_mixed, origin_list_1, origin_list_2):

        list_of_already_selected = []
        list_of_pairs = []

        first_pair_element_max_index = self.num_of_genuine_per_writer if origin_list_1 == 'genuine_scm' else self.num_of_forged_per_writer
        second_pair_element_max_index = self.num_of_genuine_per_writer if origin_list_2 == 'genuine_scm' else self.num_of_forged_per_writer

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
                    pair = (self.datasource[random_individual][origin_list_1][first_pair_element],
                            self.datasource[random_individual][origin_list_2][second_pair_element])
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
                    pair = (self.datasource[random_individual_1][origin_list_1][first_pair_element],
                            self.datasource[random_individual_2][origin_list_2][second_pair_element])
                    list_of_pairs.append(pair)
                    
        return np.array(list_of_pairs)
    
    def create_dicts(self, data):
        similar_pairs_dict = {}
        disimilar_pairs_dict = {}
        num_of_genuine = int(self.percent_of_genuine_signatures * self.num_of_genuine_per_writer)
        num_of_forged = int(self.percent_of_forged_signatures * self.num_of_forged_per_writer)

        genuine_signatures_indexes = random.sample(range(self.num_of_genuine_per_writer), num_of_genuine)
        forged_signatures_indexes = random.sample(range(self.num_of_forged_per_writer), num_of_forged)

        for person_number, _ in data.items():
            genuines = self.generate_combinations(genuine_signatures_indexes, genuine_signatures_indexes, True)
            key = (person_number, person_number)
            
            similar_pairs_dict.setdefault(key, set()).update(genuines)

            if self.mixed_or_same_in_d_pairs == 'mixed':
                forgeries = self.generate_combinations(genuine_signatures_indexes, genuine_signatures_indexes, False)

                for other_d, _ in data.items():
                    if other_d != person_number and (other_d, person_number) not in disimilar_pairs_dict:
                        key = (person_number, other_d)
                        disimilar_pairs_dict.setdefault(key, set()).update(forgeries)

            elif self.mixed_or_same_in_d_pairs == 'same':
                forgeries = self.generate_combinations(genuine_signatures_indexes, forged_signatures_indexes, False)

                key = (person_number, person_number)
                disimilar_pairs_dict.setdefault(key, set()).update(forgeries)
        
        for key in similar_pairs_dict:
            similar_pairs_dict[key] = list(similar_pairs_dict[key])

        for key in disimilar_pairs_dict:
            disimilar_pairs_dict[key] = list(disimilar_pairs_dict[key])
        
        return similar_pairs_dict, disimilar_pairs_dict
    
    def generate_combinations(self, list1, list2, avoid_equal=False, avoid_reverse=True):
        combinations = set()
        
        if avoid_equal:
            for x, y in product(list1, list2):
                if x != y:
                    combinations.add((x, y))
        else:
            combinations.update(product(list1, list2))

        if avoid_reverse:
            # Remove reverse combinations
            combinations = {tuple(sorted(pair)) for pair in combinations}
        
        return list(combinations)