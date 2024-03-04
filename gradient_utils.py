import numpy as np
import math

def calculate_dL_dDij_S(Dij_S, zeta_s):
    Dij_minus_zeta_s = Dij_S - zeta_s
    Dij_minus_zeta_s[Dij_minus_zeta_s < 0] = 0
    dL_dDij = 2 * Dij_minus_zeta_s

    return dL_dDij

def calculate_dL_ddij_S(dij_S, Dij_S, zeta_s, MT_plus_M):
    num_pairs, m = dij_S.shape

    dL_dDij = calculate_dL_dDij_S(Dij_S, zeta_s)
    
    reshaped_dij_S = dij_S.reshape(num_pairs, 1, m)
    reshaped_dL_dDij = dL_dDij[:, np.newaxis, np.newaxis]

    dL_dij_S = reshaped_dL_dDij * np.matmul(reshaped_dij_S, MT_plus_M)
    return dL_dij_S.reshape(num_pairs, m)

def calculate_dL_dDij_D(Dij_D, zeta_d):
    zeta_d_minus_Dij = zeta_d - Dij_D
    zeta_d_minus_Dij[zeta_d_minus_Dij < 0] = 0
    dL_dDij = -2 * zeta_d_minus_Dij

    return dL_dDij

def calculate_dL_ddij_D(dij_D, Dij_D, zeta_d, MT_plus_M):
    num_pairs, m = dij_D.shape

    dL_dDij = calculate_dL_dDij_D(Dij_D, zeta_d)

    reshaped_dij_D = dij_D.reshape(num_pairs, 1, m)
    reshaped_dL_dDij = dL_dDij[:, np.newaxis, np.newaxis]

    dL_dij_D = reshaped_dL_dDij * np.matmul(reshaped_dij_D, MT_plus_M)
    return dL_dij_D.reshape(num_pairs, m)

def calculate_dL_dA(
  S,
  D,
  eignv_S,
  eignv_D,
  dij_S, 
  dij_D, 
  Dij_S, 
  Dij_D, 
  A, 
  zeta_s, 
  zeta_d,
  MT_plus_M):

  """
  Euclidean gradient of the loss function L with respect to the learnable parameter A.

  S, list of genuine, genuine pairs / tuples.
  eignv_S -> list (same length as S) of lists (length m) of lists (p).
  dij_S -> list (same length as S) of lists with size m.
  Dij_S -> list (same length as S).

  """

  def term1_ddkij_dalpha_k(alpha_k, beta_k, lambda_kij):
    term1_ddkij_dalpha_k = \
    (alpha_k * np.power(lambda_kij, beta_k) -
     alpha_k * beta_k * np.power(lambda_kij, -alpha_k) * np.log(lambda_kij)) / \
    (alpha_k * np.power(lambda_kij, beta_k) + beta_k * np.power(lambda_kij, -alpha_k))

    return term1_ddkij_dalpha_k
  
  def term1_ddkij_dbeta_k(alpha_k, beta_k, lambda_kij):
    term1_ddkij_dbeta_k = \
    (beta_k * np.power(lambda_kij, -alpha_k) -
     alpha_k * beta_k * np.power(lambda_kij, beta_k) * np.log(lambda_kij))/ \
    (alpha_k * np.power(lambda_kij, beta_k) + beta_k * np.power(lambda_kij, -alpha_k))
    
    return term1_ddkij_dbeta_k

  m, _ = A.shape
  _, _, p = eignv_S.shape
  dL_dA = np.zeros_like(A)

  for k in range(m):
    alpha_k, beta_k = A[k]

    term2_ddkij_dalpha_k = np.full((p,), alpha_k / (beta_k + alpha_k)) 
    term2_ddkij_dbeta_k = np.full((p,), beta_k / (beta_k + alpha_k))

    dL_dij_S = calculate_dL_ddij_S(dij_S, Dij_S, zeta_s, MT_plus_M)
    dL_dij_D = calculate_dL_ddij_D(dij_D, Dij_D, zeta_d, MT_plus_M)

    for Dij, dL_dij, lambda_ij in zip(Dij_S, dL_dij_S, eignv_S):

      term1_ddkij_dalpha_k_S = term1_ddkij_dalpha_k(alpha_k, beta_k, lambda_ij[k])
      term1_ddkij_dbeta_k_S = term1_ddkij_dbeta_k(alpha_k, beta_k, lambda_ij[k])

      term3_ddkij_dalpha_or_beta_S = \
      np.log((alpha_k * np.power(lambda_ij[k], beta_k) + beta_k * np.power(lambda_ij[k], -alpha_k)) / \
      (alpha_k + beta_k))

      ddkij_dalpha_k_S = (1 / np.power(alpha_k, 2) * beta_k) * (term1_ddkij_dalpha_k_S - term2_ddkij_dalpha_k - term3_ddkij_dalpha_or_beta_S)
      ddkij_dbeta_k_S = (1 / alpha_k * np.power(beta_k, 2)) * (term1_ddkij_dbeta_k_S - term2_ddkij_dbeta_k - term3_ddkij_dalpha_or_beta_S)

      ddkij_dalpha_k_S = math.fsum(ddkij_dalpha_k_S)
      ddkij_dbeta_k_S = math.fsum(ddkij_dbeta_k_S)

      dL_dA[k, 0] += (1 / len(S)) * dL_dij[k] * ddkij_dalpha_k_S
      dL_dA[k, 1] += (1 / len(S)) * dL_dij[k] * ddkij_dbeta_k_S
  
    for Dij, dL_dij, lambda_ij in zip(Dij_D, dL_dij_D, eignv_D):
      
      term1_ddkij_dalpha_k_D = term1_ddkij_dalpha_k(alpha_k, beta_k, lambda_ij[k])
      term1_ddkij_dbeta_k_D = term1_ddkij_dbeta_k(alpha_k, beta_k, lambda_ij[k])

      term3_ddkij_dalpha_or_beta_D = \
      np.log((alpha_k * np.power(lambda_ij[k], beta_k) + beta_k * np.power(lambda_ij[k], -alpha_k)) / \
      (alpha_k + beta_k))

      ddkij_dalpha_k_D = (1 / np.power(alpha_k, 2) * beta_k) * (term1_ddkij_dalpha_k_D - term2_ddkij_dalpha_k - term3_ddkij_dalpha_or_beta_D)
      ddij_dbeta_k_D = (1 / alpha_k * np.power(beta_k, 2)) * (term1_ddkij_dbeta_k_D - term2_ddkij_dbeta_k - term3_ddkij_dalpha_or_beta_D)

      ddkij_dalpha_k_D = math.fsum(ddkij_dalpha_k_D)
      ddij_dbeta_k_D = math.fsum(ddij_dbeta_k_D)

      dL_dA[k, 0] += (1 / len(D)) * dL_dij[k] * ddkij_dalpha_k_D
      dL_dA[k, 1] += (1 / len(D)) * dL_dij[k] * ddij_dbeta_k_D

  return np.clip(dL_dA, -150, 150)

def calculate_dL_dW(
    pairs_S,
    pairs_D,
    low_dim_sets_S,
    low_dim_sets_D,
    dij_S,
    dij_D,
    Dij_S,
    Dij_D,
    dij_eigenvalues_S,
    dij_eigenvalues_D,
    dij_eigenvectors_S,
    dij_eigenvectors_D,
    zeta_s,
    zeta_d,
    MT_plus_M,
    W, A, m, p):

    '''
    Calculates the Euclidean gradient of L with respect to W.

    S, list of genuine, genuine pairs of SPD matrices.
    D, list of genuine, forged pairs of SPD matrices.
    low_dim_sets_S, list of genuine, genuine pairs of low-dimensional sets of SPD matrices.
    low_dim_sets_D, list of genuine, forged pairs of low-dimensional sets of SPD matrices.
    '''

    dL_dW = np.zeros_like(W)

    dL_ddij_S = calculate_dL_ddij_S(dij_S, Dij_S, zeta_s, MT_plus_M)
    dL_ddij_D = calculate_dL_ddij_D(dij_D, Dij_D, zeta_d, MT_plus_M)

    zipped_collections = \
        zip(pairs_S + pairs_D, \
            low_dim_sets_S + low_dim_sets_D, \
            dij_eigenvalues_S + dij_eigenvalues_D, \
            dij_eigenvectors_S + dij_eigenvectors_D, \
            dL_ddij_S + dL_ddij_D)

    for Xi_Xj, Xi_set_Xj_set, lambda_ij, Uij, dL_ddij in zipped_collections:
        Xi, Xj = Xi_Xj
        Xi_set, Xj_set = Xi_set_Xj_set
        
        for k in range(m):
            alpha_k, beta_k = A[k]

            Wk = W[:, k*p : (k+1)*p]

            dL_dlambda_kij = \
                dL_ddij[k] * (1 / alpha_k*beta_k) * \
                ((alpha_k*beta_k*np.power(lambda_ij[k], beta_k-1) - alpha_k*beta_k*np.power(lambda_ij[k], -alpha_k-1)) / \
                 (alpha_k*np.power(lambda_ij[k], beta_k) + beta_k*np.power(lambda_ij[k], -alpha_k)))

            dL_dsigma_kij = np.diag(dL_dlambda_kij)

            transpose_Xki = np.transpose(Xi_set[k])
            inverse_transpose_Xki = np.linalg.inv(transpose_Xki)
            transpose_Xkj = np.transpose(Xj_set[k])
            inverse_transpose_Xkj = np.linalg.inv(transpose_Xkj)

            dL_dXki = Uij[k] @ dL_dsigma_kij @ np.transpose(Uij[k]) @ inverse_transpose_Xki
            dL_dXkj = -1 * inverse_transpose_Xkj @ transpose_Xki @ Uij[k] @ dL_dsigma_kij @ np.transpose(Uij[k]) @ inverse_transpose_Xkj

            dL_dWk = np.transpose(Xi) @ Wk @ dL_dXki + \
                     Xi @ Wk @ np.transpose(dL_dXki) + \
                     np.transpose(Xj) @ Wk @ dL_dXkj + \
                     Xj @ Wk @ np.transpose(dL_dXkj)

            dL_dW[:, k*p : (k+1)*p] = dL_dWk
    
    return dL_dW

def calculate_dL_dM(S, D, S_dij, D_dij, Dij_S, Dij_D, M, M0, zeta_s, zeta_d, sci):
  '''
  Euclidean gradient of the loss function with respect to the learnable parameter M.

  Receives:
  List of dij vectors representing the subdistance measures for D pairs (D_dij);
  List of dij vectors representing the subdistance measures for S pairs (S_dij);
  List of set-to-set distances for S pairs (distances_S);
  '''

  m, _ = M.shape
  num_pairs, _, _, _ = S.shape

  # (50,)
  dL_dDij_S = calculate_dL_dDij_S(Dij_S, zeta_s).reshape(num_pairs, 1)
  dL_dDij_D = calculate_dL_dDij_D(Dij_D, zeta_d).reshape(num_pairs, 1)

  term1_before_sum = (1 / len(S)) * np.matmul(S_dij.reshape(num_pairs, m, 1), (dL_dDij_S * S_dij).reshape(num_pairs, 1, m))

  term2_before_sum = (1 / len(D)) * np.matmul(D_dij.reshape(num_pairs, m, 1), (dL_dDij_D * D_dij).reshape(num_pairs, 1, m))

  last_dimensions_shape = term1_before_sum.shape[1:]

  term1 = np.array([math.fsum(term1_before_sum[index]) for index in np.ndindex(last_dimensions_shape)]).reshape(last_dimensions_shape)

  term2 = np.array([math.fsum(term2_before_sum[index]) for index in np.ndindex(last_dimensions_shape)]).reshape(last_dimensions_shape)

  #term1 = (1 / len(S)) * np.sum(np.matmul(S_dij.reshape(num_pairs, m, 1), (dL_dDij_S * S_dij).reshape(num_pairs, 1, m)), axis=0)

  #term2 = (1 / len(D)) * np.sum(np.matmul(D_dij.reshape(num_pairs, m, 1), (dL_dDij_D * D_dij).reshape(num_pairs, 1, m)), axis=0)

  term3 = sci * (np.linalg.inv(M0) - np.linalg.inv(M))

  dL_dM = term1 + term2 + term3

  return dL_dM

def calculate_dL_dWR(dL_dW, W):
    '''
    Calculates the Riemannian gradient of L, with respect to W, from the Euclidean one.
    '''

    dL_dWR = dL_dW - W @ ((1 / 2) * (np.transpose(W) @ dL_dW + np.transpose(dL_dW) @ W))

    return dL_dWR

def calculate_dL_dMR(dL_dM, M):
    '''
    Calculates the Riemannian gradient of L, with respect to M, from the Euclidean one.
    '''

    dL_dMR = M @ ((1 / 2) * (dL_dM + dL_dM.T) @ M)

    return dL_dMR

def update_W_parameter(W_tminus1, learning_rate, dL_dWR):
    W_t, _ = np.linalg.qr(W_tminus1 - learning_rate*dL_dWR)
    return W_t

def update_M_parameter(M_tminus1, learning_rate, dL_dMR):
    
    eigenvalues_Mtminus1, eigenvectors_Mtminus1 = np.linalg.eigh(M_tminus1)

    # M_tminus1 = eigenvectors_Mtminus1 @ diag_Mtminus1 @ eigenvectors_Mtminus1.T
    eigenvalues_Mtminus1_power_1_by_2 = eigenvalues_Mtminus1 ** (1/2)
    eigenvalues_Mtminus1_power_minus_1_by_2 = eigenvalues_Mtminus1 ** (-1/2)
    
    M_tminus1_squareroot = eigenvectors_Mtminus1 @ np.diag(eigenvalues_Mtminus1_power_1_by_2) @ eigenvectors_Mtminus1.T
    M_tminus1_negative_squareroot = eigenvectors_Mtminus1 @ np.diag(eigenvalues_Mtminus1_power_minus_1_by_2) @ eigenvectors_Mtminus1.T

    inner_term = -learning_rate * (M_tminus1_negative_squareroot @ dL_dMR @ M_tminus1_negative_squareroot)

    eigenvalues_inner, eigenvectors_inner = np.linalg.eigh(inner_term)

    expm = eigenvectors_inner @ np.diag(np.exp(eigenvalues_inner)) @ eigenvectors_inner.T

    Mt = M_tminus1_squareroot @ expm @ M_tminus1_squareroot

    return Mt