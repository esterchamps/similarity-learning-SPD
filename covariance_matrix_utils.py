import numpy as np

def calculate_feature_map(image, binary_mask):
    '''
    Obtém o mapa de features da imagem. O mapa consiste na aplicação de
    10 funções sob os pixels da imagem: gradientes de primeiro e segundo grau,
    magnitude do gradiente, direção do gradiente e normalização dos valores
    das coordenadas.
    Recebe a imagem (em escala de cinzas) e a máscara binária da assinatura.
    O mapa final terá apenas os valores resultantes correspondentes aos píxels
    da assinatura.
    Retorna um ndarray.
    '''

    image = image / 255.0
    h, w = image.shape
    Ix = np.gradient(image, axis=1)
    Iy = np.gradient(image, axis=0)
    Ixx = np.gradient(Ix, axis=1)
    Ixy = np.gradient(Ix, axis=0)
    Iyy = np.gradient(Iy, axis=0)
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    gradient_direction = np.arctan2(Iy, Ix)
    xn = np.tile(np.arange(w) / w, (h, 1))
    yn = np.tile(np.arange(h) / h, (w, 1)).T

    feature_map = np.stack([
        image, Ix, Iy, Ixx, Ixy, Iyy, gradient_magnitude, gradient_direction, xn, yn
    ], axis=0)

    feature_map_only_signature_pixels = feature_map * binary_mask[np.newaxis, :, :]
    
    return feature_map_only_signature_pixels

def calculate_signature_covariance_matrix(feature_map):
    '''
    Calcula a matriz de covariância, dado um mapa de features.
    Retorna um ndarray com dimensões (10, 10).
    '''

    feature_map = feature_map.reshape(10, -1)

    S = feature_map.shape[1]
    mu = np.mean(feature_map, axis=1, keepdims=True)

    inner_summup = (feature_map - mu) @ (feature_map - mu).T
    covariance_matrix = (1 / (S - 1)) * inner_summup
    
    return covariance_matrix
