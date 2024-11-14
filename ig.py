# Information Gain

from typing import Counter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import etl
#import utility_ig    as ut


#Normalizar por metodo sigmoidal
def norm_data_sigmoidal(X):
    epsilon = 1e-10
    mu_x = np.mean(X)
    sigma_x = np.std(X)
    u = (X - mu_x) / (sigma_x + epsilon)
    X_normalized = 1 / (1 + np.exp(-u))
    return X_normalized

#Crear vectores-embedding
def create_embedding_vectors(X, m, tau):
    N = len(X)
    M = N - (m - 1) * tau  # número total de vectores embebidos
    embedding_vectors=[]
    for i in range(M):
        vector = [X[i + j * tau] for j in range(m)]
        embedding_vectors.append(vector)
    return np.array(embedding_vectors)

#Mapear cada vector-embedding en c-símbolos
def map_to_symbols(embedding_vectors, c):
    Y = [np.round((map * c) + 0.5) for map in embedding_vectors]
    return np.array(Y)

#Convierte el vector Y en un patron de simbolos
def convertir_y(Y, c, m):
    k = []
    Yi_minus_1 = Y - 1
    c_elevado = [c**i for i in range(m)]
    punto = np.dot(Yi_minus_1, c_elevado)
    aux = 1 + punto
    k.append(aux)

    return k

#Calcula la entropia de Shannon
def entropy(probabilities,c,m):
    r = c ** m
    #Calcular DE
    de = -np.sum(probabilities * np.log2(probabilities) + 1e-10 )
    #Calcular la Entropía de Dispersión normalizada (nDE)
    nde = de / np.log2(r)
    return nde


#Calcula la Dispersion entropy
def entropy_disp(data_class, c, m, tau): 
    N = data_class.shape[0]

    # Verificar si el tamaño de los datos es suficiente para crear los vectores de embedding
    min_length = (m - 1) * tau + 1
    if N < min_length:
        # Rellenar con ceros hasta alcanzar la longitud mínima
        data_class = np.pad(data_class, (0, min_length - N), 'constant', constant_values=0)
        N = data_class.shape[0]

    #Normalizar los datos
    data_class = norm_data_sigmoidal(data_class)
    data_class = np.array(data_class)
    #Crear vectore embedding
    X = create_embedding_vectors(data_class, m, tau)
    #Mapear vectores embedding
    Y = map_to_symbols(X, c)
    #Convertir vectores de simbolos
    k_valores = convertir_y(Y, c, m)
    k_valores = np.array(k_valores[0])
    #Calcular la frecuencia de cada K_valores
    pattern_counts = Counter(k_valores) 
    #Calcular la probabilidad de cada patrón
    total_patterns = N - (m - 1) * tau
    #Calcular la probabilidad de capa patrón de dispersión
    probabilities = np.array([count / total_patterns for count in pattern_counts.values()])
    #Calcular la Entropía de Dispersión
    nde = entropy(probabilities, c, m)
    return nde


#Calcula la Entropía condicional H(Y|X)
def conditional_entropy_disp(X, Y, c, m, tau):
    np.set_printoptions(suppress=True, precision=4)
    #Se obtiene el valor de la cantidad de muestras
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
        #Calcula la frecuencia de las categorias segun el bin actual
        # Obtener las clases de las muestras en el bin actual
        categorias_en_bin = Y[dij].values  # Esto dará las clases directamente en el bin `j`
        Hyx = Hyx + (1/N * np.sum(dij) * entropy_disp(categorias_en_bin, c, m, tau))   
    return (Hyx)

#Information gain
def inform_gain(data_class,config): 
    m = int(config[0][0])  #Dimensión del vector-embedding
    tau = int(config[0][1]) #Número de tau
    c = int(config[0][2])   #Número de símbolos
    top_K = int(config[0][3]) #Top k indicadores
    X = data_class.iloc[:,:-1] #Se filtran las distintas caracteristicas del dataset
    Y = data_class.iloc[:,-1] #Se filtran las clases del dataset

    #Se calcula las probabilidad de aparicion de las clases
    #Calcular la entropia de dispersión de las etiquetas Y
    hy = entropy_disp(Y,c,m,tau)
    print(hy)
    #Calcular la ganancia de información
    information_gain = []
    for i in range(X.shape[1]):
        #Calcular la entropia condicional de Y dado x
        hyx_i = conditional_entropy_disp(X.iloc[:, i], Y, c, m, tau)
        ig_i = hy - hyx_i
        information_gain.append(ig_i)

    fig, ax = plt.subplots()
    #Se genera un grafico mostrando los resultados
    plt.stem(range(len(information_gain)), information_gain, basefmt=" ", linefmt="green", markerfmt="go")    
    plt.xlabel('Caracteristica')
    plt.ylabel('IG')
    plt.title('IG por Caracteristica')
    plt.show()

    top_indices = np.sort(information_gain)[-top_K:][::-1]  
    #Se genera un grafico mostrando las top_k mas relevantes
    plt.stem(range(len(top_indices)), top_indices, basefmt=" ", linefmt="green", markerfmt="go")    
    plt.xlabel('Característica')
    plt.ylabel('IG')
    plt.title('IG por Característica')
    # plt.show()

    information_gain = np.array(information_gain)
    # Obtenemos los índices ordenados de mayor a menor según el IG
    idx_sorted = np.argsort(-information_gain)  # Índices en orden descendente
    # Crear DataFrame para guardar los índices ordenados
    idx_variable_df = pd.DataFrame(idx_sorted, columns=['Feature_Index'])
    # Guardar en archivo CSV
    idx_variable_df.to_csv('Idx_variable.csv', index=False)
    # Leer el archivo CSV
    idx_variable_df = pd.read_csv('Idx_variable.csv')
    # Sumar 1 a todos los índices en la columna 'Feature_Index'
    idx_variable_df['Feature_Index'] += 1
    # Guardar el DataFrame actualizado de nuevo en el archivo CSV
    idx_variable_df.to_csv('Idx_variable.csv', index=False)

    # Filtrar el dataset para incluir solo las características más relevantes
    data_ig = data_class.iloc[:, idx_sorted[:top_K]]

    # Guardar el nuevo dataset en un archivo CSV
    data_ig.to_csv('DataIG.csv', index=False, header = False)

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





