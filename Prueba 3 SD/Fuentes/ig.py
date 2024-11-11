# Information Gain

from typing import Counter
import numpy as np
import pandas as pd
import etl
#import utility_ig    as ut


# Normalised by use sigmoidal
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

# Dispersion entropy
def entropy_disp(data_class,config):  
    m = 3   # dimensión del vector-embedding
    tau = 2 # distancia entre elementos consecutivos
    c = 3   # número de símbolos
    N = data_class.shape[0]

    
    if len(data_class.shape) > 1:
        for columna in  data_class:
            data_class[columna] = norm_data_sigmoidal(data_class[columna])

    else:
        data_class = norm_data_sigmoidal(data_class)

    data_class.to_csv("norm_data_sigmoidal.csv", index=False, header=False)
    #print(data_class)
    data_class = np.array(data_class)
    
    if len(data_class.shape) > 1:
        X = [create_embedding_vectors(data_class[:, i], m, tau) for i in range(data_class.shape[1])]
        Y = np.array([[map_to_symbols(X[i, j], c) for j in range(X.shape[1])] for i in range(X.shape[0])])    
        k_valores = np.array([[convertir_y(Y[i, j], c, m) for j in range(Y.shape[1])] for i in range(Y.shape[0])]) 

    else:
        X = create_embedding_vectors(data_class, m, tau)
        print(X)
        Y = map_to_symbols(X, c)
        print(Y)
        k_valores = convertir_y(Y, c, m)
        k_valores = np.array(k_valores[0])
        print(k_valores)

    
    # print(embedded_data[1])
    # print(Y[1])
    # print(k_valores[1])
    
    # Paso 5: Aplanar el vector Y
    k_valores_flattened = k_valores.reshape(-1)
    print(k_valores_flattened.shape)


    # Paso 6: Calcular la probabilidad de cada patrón de dispersión
    # # Contar la frecuencia de cada patrón único en el vector aplanado
    pattern_counts = Counter(k_valores_flattened) 
    # # Calcular la probabilidad de cada patrón
    total_patterns = N - (m - 1) * tau

    probabilities = {pattern: count / total_patterns for pattern, count in pattern_counts.items()}

    # Paso 7: Calcular la Entropía de Dispersión
    # Número total de patrones posibles
    r = c ** m
    # Calcular DE
    de = -sum(p * np.log2(p) for p in probabilities.values())
    # Calcular la Entropía de Dispersión normalizada (nDE)
    nde = de / np.log2(r)
    print(de,nde)

    return()

#Information gain
def inform_gain(data_class): 
    return()
    
# Load dataClass 
def load_data():   
    config = etl.config()
    data_class = etl.cargar_datos()
    return config, data_class

# Beginning ...
def main():    
    config, data_class = load_data()
    Y = data_class.iloc[:,-1]
    print(Y)
    entropy_disp(Y,config)
       
if __name__ == '__main__':   
	 main()

