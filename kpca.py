import numpy as np
import pandas as pd
import etl

# Gaussian Kernel Function
def kernel_gauss(X, sigma):
    # Calcular la norma cuadrada de cada fila de X (X es de tamaño (N, d))
    sq_norms = np.sum(X**2, axis=1)
    # Aplicar la propiedad del cuadrado del binomio para calcular distancias cuadradas
    dist_sq_matrix = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(X, X.T)
    # Aplicar la función Gaussiana al resultado para obtener la matriz del kernel
    K = np.exp(-dist_sq_matrix / (2 * sigma ** 2))
    
    return K

# Kernel-PCA
def kpca_gauss(X, sigma, top_k):
    # Paso 1: Calcular la matriz del Kernel
    K = kernel_gauss(X, sigma)
    
    # Paso 2: Centrar en media la matriz del Kernel
    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    K_centered = K - one_N @ K - K @ one_N + one_N @ K @ one_N
    
    # Paso 3: Calcular los valores y vectores propios del Kernel
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    
    # Paso 4: Ordenar los valores propios y vectores propios en orden descendente
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Paso 5: Seleccionar los top_k componentes principales
    eigvecs_top_k = eigvecs[:, :top_k]
    eigvals_top_k = eigvals[:top_k]
    
    # Normalización de vectores propios (opcional pero recomendable)
    for i in range(top_k):
        eigvecs_top_k[:, i] /= np.sqrt(eigvals_top_k[i])
    
    # Proyectar los datos
    X_kpca = K_centered @ eigvecs_top_k
    
    return X_kpca

# Cargar y preparar los datos
def load_data(filepath, num_samples=3000):
    data = pd.read_csv(filepath)
    data = data.iloc[:num_samples]  # Seleccionar las primeras num_samples
    config = etl.config()

    # Guardar las primeras num_samples en un nuevo archivo
    data.to_csv("Data_2.csv", index=False)

    # Separar características y etiquetas
    X = data.iloc[:, :-1].values  # Características

    # Parámetros del kernel KPCA
    sigma =  int(config[0][4]) # Ancho del kernel, puedes ajustar este valor
    top_k = int(config[0][5])    # Número de componentes principales a conservar, puedes ajustar este valor


    return X, sigma, top_k


# Función principal
def main():
    # Cargar datos
    X, sigma, top_k = load_data("DataIG.csv", num_samples=3000)
    
    
    # Aplicar KPCA
    X_kpca = kpca_gauss(X, sigma, top_k)
    
    # Guardar los resultados en un archivo CSV
    df_kpca = pd.DataFrame(X_kpca, columns=[f"PC{i+1}" for i in range(top_k)])
    df_kpca.to_csv("DataKpca.csv", index=False)
    print("KPCA realizado y datos guardados en DataKpca.csv")

if __name__ == '__main__':
    main()
