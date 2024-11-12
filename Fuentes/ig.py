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
    config = np.array(config) 
    m = int(config[0])  # dimensión del vector-embedding
    tau = int(config[1]) # distancia entre elementos consecutivos
    c = int(config[2])   # número de símbolos
    N = data_class.shape[0]

    
    data_class = norm_data_sigmoidal(data_class)
    #print(data_class)
    data_class.to_csv("norm_data_sigmoidal.csv", index=False, header=False)
    #print(data_class)
    data_class = np.array(data_class)
    

    X = create_embedding_vectors(data_class, m, tau)
    #print(X)
    Y = map_to_symbols(X, c)
    #print(Y)
    k_valores = convertir_y(Y, c, m)
    k_valores = np.array(k_valores[0])
    #print(k_valores)
    
    
    # Paso 5: Aplanar el vector Y
    k_valores_flattened = k_valores.reshape(-1)
    #print(k_valores_flattened.shape)


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

    return de

# Entropía condicional H(Y|X)
def conditional_entropy_disp(X, Y, config, bins):
    # Normalizar y crear bins en X
    X = norm_data_sigmoidal(X)
    print("aca",X)
    n_bins = int(bins)  # cantidad de bins para discretizar X
    X_binned = np.digitize(X, bins=np.linspace(0, 1, n_bins + 1))  # Binear X en n_bins
    
    # Para cada bin en X, calcular la entropía de Y
    cond_entropy = 0
    total_samples = len(Y)
    for bin_val in range(1, n_bins + 1):
        # Seleccionar muestras donde X está en el bin actual
        indices = np.where(X_binned == bin_val)[0]
        print(indices)
        if len(indices) == 0:
            continue
        
        # Calcular entropía de Y en el subconjunto donde X está en el bin actual
        Y_subset = Y[indices]
        print("hola",np.array(Y_subset))
        subset_entropy = entropy_disp(Y_subset, config)
        
        # Ponderar la entropía del subconjunto
        cond_entropy += (len(indices) / total_samples) * subset_entropy
    
    return cond_entropy

#Information gain
def inform_gain(data_class,config): 
    X = np.array(data_class.iloc[:,:-1])
    Y = data_class.iloc[:,-1]
    
    #hy = entropy_disp(Y,config)
    bins = np.sqrt(data_class.shape[0])

    # Aplicamos la función para cada columna de X
    n_features = X.shape[1]
    hyx_total = []
    # Sumar H(Y|X_i) para cada columna X_i de X
    for i in range(n_features):
        hyx_i = conditional_entropy_disp(X[:, i], Y, config,bins)
        hyx_total.append(hyx_i)

    print(hyx_total)   
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





