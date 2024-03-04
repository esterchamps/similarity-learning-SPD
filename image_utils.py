import cv2 as cv
import numpy as np
from scipy import ndimage

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

def thin_operation(img, th_level):
    '''
    Aplica th_level operações de erosão.
    Recebe um ndarray contendo uma máscara binária de uma imagem.
    Devolve uma máscara binária também, com o resultado da operação.
    '''
        
    #img = ndimage.binary_opening(img, iterations=th_level)
    img = ndimage.binary_erosion(img, iterations=th_level)
        
    return img.astype(int)

def otsu_thresholding(image_array):
    '''
    Aplica a limiarização de Otsu numa imagem.
    Recebe um ndarray.
    Retorna um ndarray onde os pixels brancos representam o objeto e os pixels pretos
    representam o plano de fundo.
    '''

    thres, otsu_result = cv.threshold(image_array, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu_inverted = otsu_result^1

    return thres, otsu_inverted

def calculate_patch_densities(image):
    '''
    Calcula as densidades de cada bloco 5x5 centrado em um pixel de assinatura.
    A densidade de um bloco é uma fração de quantos pixels do bloco são pixels
    de assinatura (pixels brancos, iguais a 1).
    Recebe uma máscara binária da assinatura.
    Retorna um array com os valores das densidades.
    '''

    ld = []
    pad = 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 1:
                patch = padded_image[i:i+5, j:j+5]
                patch_density = np.sum(patch) / 25
                ld.append(patch_density)
    
    if len(ld) == 0:
        ld.append(0)
                
    return np.array(ld)

def get_optimal_thinning_level(image_array):
    '''
    Calcula a quantidade ótima de operações de abertura/fechamento na imagem.
    Recebe um ndarray com a máscara binária da imagem.
    Retorna um inteiro, correspondente ao número ótimo de operações de
    abertura/fechamento.
    '''

    pd = []
    th_level = 0

    while True:
        thinned_image = thin_operation(image_array, th_level)

        ld = calculate_patch_densities(thinned_image)
        ld_mean = np.mean(ld)
        if len(pd) > 1 and np.abs(pd[-1] - ld_mean) < 0.12:
            break
        pd.append(ld_mean)
        th_level += 1

    pd_diff = np.abs(np.diff(pd))
    otl = np.argmax(pd_diff) + 1

    return otl

def get_most_optimal_thinning_level(genuine_signatures_of_an_individual_preprocessed_with_ostu):
    '''
    Retorna a quantidade ótima de operações de abertura/fechamento para o conjunto
    de imagens de assinaturas genuínas de um indivíduo.
    Recebe a lista de assinaturas genuínas de um único indivíduo.
    Retorna um inteiro, correspondente ao número ótimo de operações de abertura/fechamento.
    '''

    otl = []
    index = 0
    for signature_image in genuine_signatures_of_an_individual_preprocessed_with_ostu:

        signature_otl = get_optimal_thinning_level(signature_image)
        
        otl.append(signature_otl)
        index += 1
    
    motl = np.mean(otl)

    return motl