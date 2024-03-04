import numpy as np
import math

def get_alpha_beta_divergence(matrix1_p_x_p, matrix2_p_x_p, alpha, beta):
    '''
    Cálculo da subdistância gA(·, ·).
    Retorna uma tupla com: valor da divergência/distância, vetor de autovalores e matrix de autovetores.
    '''
    
    inverse2 = np.linalg.inv(matrix2_p_x_p)
    m1_dot_inverse2 = matrix1_p_x_p @ inverse2

    eigenvalues, eigenvectors = np.linalg.eig(m1_dot_inverse2)

    divergences = \
        (1 / (alpha * beta)) \
        * np.log(np.divide((alpha * np.power(eigenvalues, beta) + beta * np.power(eigenvalues, -alpha)), (alpha + beta)))
    
    divergence = math.fsum(divergences)

    return divergence, eigenvalues, eigenvectors

def get_set_of_m_SPD_matrices(covariance_matrix_n_x_n, learnable_parameter_W_n_x_mp, m, p):
    '''
    Transformação de ponto em conjunto, Ts(·). Gera uma lista com m projeções, fW(·), de menor dimensão.
    '''

    Zi = np.transpose(learnable_parameter_W_n_x_mp) @ covariance_matrix_n_x_n @ learnable_parameter_W_n_x_mp

    rows, cols = Zi.shape

    block_diagonal_matrices = []
    start_row, start_col = 0, 0

    while len(block_diagonal_matrices) < m:
        end_row = min(start_row + p, rows)
        end_col = min(start_col + p, cols)

        block_diagonal_matrix = Zi[start_row:end_row, start_col:end_col]
        block_diagonal_matrices.append(block_diagonal_matrix)

        start_row += p
        start_col += p

    return np.array(block_diagonal_matrices)

def calculate_point_to_point_distance(learnable_parameter_M_m_x_m, local_distances_vector_Rm):
    '''
    Distância entre conjuntos, Ds(·, ·). Calculada com a função de integração hM (·).
    '''

    sum_of_d_M_d_terms = np.einsum('kl,k,l', learnable_parameter_M_m_x_m, local_distances_vector_Rm, local_distances_vector_Rm)

    return sum_of_d_M_d_terms

def calculate_dij_and_its_eigenvalues_and_eigenvectors(pairs_of_sets, A):
    '''
    pairs_of_sets -> list of tuples with 2 elements, each with size m.
    A -> m x 2

    dij_of_sets -> list (same length as pairs_of_sets) of lists with size m.
    eigenvalues_of_dij -> list (same length as pairs_of_sets) of lists (length m) of lists (p).
    '''

    dij_of_sets = []
    eigenvalues_of_dij = []
    eigenvectors_of_dij = []

    for Xi_set, Xj_set in pairs_of_sets:
        dij = []
        eigenvalues = []
        eigenvectors = []

        for Xki, Xkj, Ak in zip(Xi_set, Xj_set, A):

            alpha, beta = Ak
            dkij, lambda_kij, Ukij = get_alpha_beta_divergence(Xki, Xkj, alpha, beta)
            
            dij.append(dkij)
            eigenvalues.append(lambda_kij)
            eigenvectors.append(Ukij)
        
        dij_of_sets.append(dij)
        eigenvalues_of_dij.append(eigenvalues)
        eigenvectors_of_dij.append(eigenvectors)
    
    return np.array(dij_of_sets), np.array(eigenvalues_of_dij), np.array(eigenvectors_of_dij)

def apply_point_to_set_transformation(scm_pairs, W, m, p):

    pairs_of_sets = []

    for Xi, Xj in scm_pairs:
        Xi_set = get_set_of_m_SPD_matrices(Xi, W, m, p)
        Xj_set = get_set_of_m_SPD_matrices(Xj, W, m, p)

        pairs_of_sets.append((Xi_set, Xj_set))

    return np.array(pairs_of_sets)

def calculate_Dij(list_of_dij, M):
    list_of_Dij = []
    
    for dij in list_of_dij:
        Dij = calculate_point_to_point_distance(M, dij)
        list_of_Dij.append(Dij)
    
    return np.array(list_of_Dij)