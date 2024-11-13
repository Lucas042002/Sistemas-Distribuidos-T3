# Information Gain

from typing import Counter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import etl
#import utility_ig    as ut


#Paso 1: Normalizar por metodo sigmoidal
def norm_data_sigmoidal(X):
    epsilon = 1e-10
    mu_x = np.mean(X)
    sigma_x = np.std(X)
    u = (X - mu_x) / (sigma_x + epsilon)
    X_normalized = 1 / (1 + np.exp(-u))
    return X_normalized
# Paso 2: Crear vectores-embedding
def create_embedding_vectors(X, m, tau):
    N = len(X)
    M = N - (m - 1) * tau  # número total de vectores embebidos
    embedding_vectors=[]
    for i in range(M):
        vector = [X[i + j * tau] for j in range(m)]
        embedding_vectors.append(vector)
    return np.array(embedding_vectors)
# Paso 3: Mapear cada vector-embedding en c-símbolos
def map_to_symbols(embedding_vectors, c):
    Y = []
    for i in range(embedding_vectors.shape[0]):
        symbols = np.round(c * embedding_vectors[i] + 0.5).astype(int)
        Y.append(symbols)
    return np.array(Y)

def convertir_y(Y, c, m):
    k = []
    Yi_minus_1 = Y - 1
    c_elevado = [c**i for i in range(m)]
    punto = np.dot(Yi_minus_1, c_elevado)
    aux = 1 + punto
    k.append(aux)

    return k

def entropy(probabilities,c,m):
    r = c ** m
    # Calcular DE
    de = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    # Calcular la Entropía de Dispersión normalizada (nDE)
    nde = de / np.log2(r)
    #print(de,nde)
    return nde

# Dispersion entropy
def entropy_disp(data_class, c, m, tau): 
    N = data_class.shape[0]
    #Paso 1: Normalizar los datos
    data_class = norm_data_sigmoidal(data_class)
    data_class = np.array(data_class)
    #Paso 2: Crear vectore embedding
    X = create_embedding_vectors(data_class, m, tau)
    #Paso 3: Mapear vectores embedding
    Y = map_to_symbols(X, c)
    #Paso 4: Convertir vectores de simbolos
    k_valores = convertir_y(Y, c, m)
    k_valores = np.array(k_valores[0])
    #Se aplanan los K valores
    k_valores_flattened = k_valores.reshape(-1)
    #Paso 5: Calcular la frecuencia de cada K_valores
    pattern_counts = Counter(k_valores_flattened) 
    #Calcular la probabilidad de cada patrón
    total_patterns = N - (m - 1) * tau
    #Paso 6: Calcular la probabilidad de capa patrón de dispersión
    probabilities = np.array([count / total_patterns for count in pattern_counts.values()])
    #Paso 7: Calcular la Entropía de Dispersión
    # Número total de patrones posibles
    nde = entropy(probabilities, c, m)
    return nde

# Entropía condicional H(Y|X)
def conditional_entropy_disp(X, Y, c,m):
    np.set_printoptions(suppress=True, precision=4)
    N = X.shape[0]
    
    #Normalizar los X datos
    X = norm_data_sigmoidal(X)
    #Calcular el número de bins
    num_bins = int(np.sqrt(N))
    #Calcular la cantidad de bins por columna
    bins = np.linspace(np.min(X), np.max(X), num_bins + 1)
    #Calcular los indices de los bins
    data_binned = np.digitize(X, bins) - 1
    #Obtener indices no repetidos
    datos_unicos = np.unique(data_binned)
    Hyx = 0
    for j in datos_unicos:
        dij = (data_binned == j)
        if np.sum(dij) == 0:
            continue
        frecuencia_categorias = Y[dij].value_counts(normalize=True).values
        Hyx = Hyx + (1/N * np.sum(dij) * entropy(frecuencia_categorias, c,m)) 
    
    return (Hyx)

#Information gain
def inform_gain(data_class,config): 
    m = int(config[0][0])  # dimensión del vector-embedding
    c = int(config[0][2])   # número de símbolos

    X = data_class.iloc[:,:-1]
    Y = data_class.iloc[:,-1]

    #Paso 1: Calcular la entropia de dispersión de las etiquetas Y
    prob_clase_Y = Y.value_counts(normalize=True).values
    print(prob_clase_Y)
    hy = entropy(prob_clase_Y,c,m)
    #Paso 2: Calcular la entropia condicional de Y dado x
    #Cant_column = 
    information_gain = []
    for i in range(X.shape[1]):
        hyx_i = conditional_entropy_disp(X.iloc[:, i], Y, c,m)
        ig_i = hy - hyx_i
        information_gain.append(ig_i)

    fig, ax = plt.subplots()
    plt.bar(range(len(information_gain)), information_gain, color='green')
    plt.xlabel('Caracteristica')
    plt.ylabel('IG')
    plt.title('IG por Caracteristica')
    plt.show()
    return()
    
# Load dataClass 
def load_data():   
    config = etl.config()
    data_class = etl.cargar_datos()
    return config, data_class

# Beginning ...
def main():    
    config, data_class = load_data()
    inform_gain(data_class, config)
       
if __name__ == '__main__':   
	 main()





