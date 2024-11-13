import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Función de Configuración ###

def config():
    config = pd.read_csv(r'C:\Users\lucas\Desktop\Prueba 3 SD\config.csv', header=None)
    parameters = config.values
    m = int(parameters[0][0])      # Dimensión de embedding
    tau = int(parameters[1][0])    # Factor de retardo
    c = int(parameters[2][0])      # Número de símbolos
    vRK = int(parameters[3][0])    # Top-K variables relevantes
    sigma = float(parameters[4][0])  # Ancho del kernel (para KPCA)
    vMRK = int(parameters[5][0])   # Top-K variables menos redundantes (para KPCA)
    return m, tau, c, vRK, sigma, vMRK

### Función de Normalización Sigmoidal ###

def norm_data_sigmoidal(column):
    promX = column.mean()
    desvX = column.std()
    epsilon = 1e-10
    u = (column - promX) / (desvX + epsilon)
    return 1 / (1 + np.exp(-u))

### Funciones para el Cálculo de la Entropía de Dispersión (DE) ###

def embVector(column, m, tau):
    N = len(column)
    M = N - (m - 1) * tau
    X = []
    for i in range(1, M + 1):
        embV = [column[j - 1] for j in range(i, i + (m - 1) * tau + 1, tau)]
        X.append(embV)
    return np.array(X)

def map_symbols(column, c):
    Y = [np.round((val * c) + 0.5) for val in column]
    return np.array(Y)

def YaPatron(column, c):
    k = []
    for symbols in column:
        pattern = sum((symbols[j] - 1) * (c ** j) for j in range(len(symbols)))
        k.append(pattern + 1)
    return np.array(k)

def contFrenc(column):
    patron = np.unique(column)
    frec = [np.sum(column == p) for p in patron]
    return np.array(frec), np.array(patron)

def probDisp(frec, N, m, tau):
    prob = [f / (N - (m - 1) * tau) for f in frec]
    return np.array(prob)

def entropy_disp(prob, c, m):
    r = c ** m
    DE = -np.sum(prob * np.log2(prob + 1e-10))
    return DE / np.log2(r)

### Función para el Cálculo de la Ganancia de Información (IG) ###

def inform_gain(X, Y, m, tau, c):
    # Paso 1: Calcular Entropía de Dispersión de las etiquetas
    class_probs = Y.value_counts(normalize=True).values
    Hy = entropy_disp(class_probs, c, m)
    print(Hy)
    ig_scores = []
    cont = 0
    # Paso 2: Calcular Entropía Condicional Hyx para cada variable en X
    for i in range(X.shape[1]):
        X_i = X.iloc[:, i]
        # Discretizar X_i en bins
        bins = np.linspace(X_i.min(), X_i.max(), int(np.sqrt(len(X_i))) + 1)
        X_binned = np.digitize(X_i, bins) - 1
        Hyx = 0
        for b in np.unique(X_binned):
            mask = (X_binned == b)
            if np.sum(mask) == 0:
                continue

            prob_y_in_bin = Y[mask].value_counts(normalize=True).values
            Hyx += (np.sum(mask) / len(Y)) * entropy_disp(prob_y_in_bin, c, m)
        
        # Ganancia de Información IG
        ig_scores.append(Hy - Hyx)

    return np.array(ig_scores)

### Función de Visualización ###

def plot_ig_scores(ig_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(ig_scores)), ig_scores, color='blue')
    plt.xlabel('Variables')
    plt.ylabel('Ganancia de Información')
    plt.title('Ganancia de Información para cada Variable')
    plt.show()

### Función para Cargar los Datos ###

def load_data():
    dataClass = pd.read_csv(r'C:\Users\lucas\Desktop\Prueba 3 SD\DataClasss.csv', header=None)
    X = dataClass.drop(dataClass.columns[41], axis=1)
    Y = dataClass[41]
    return X, Y

### Función Principal ###

def main():
    # Cargar datos y parámetros
    X, Y = load_data()
    m, tau, c, vRK, sigma, vMRK = config()

    # Normalizar cada columna de X
    Xnorm = X.apply(norm_data_sigmoidal, axis=0)
    
    # Calcular la ganancia de información
    ig_scores = inform_gain(Xnorm, Y, m, tau, c)
    
    # Seleccionar las Top-K variables con mayor IG
    top_k_indices = np.argsort(ig_scores)[-vRK:]
    
    # Guardar las variables seleccionadas y las etiquetas en DataIG.csv
    X_top_k = X.iloc[:, top_k_indices]
    data_selected = pd.concat([X_top_k, Y], axis=1)
    data_selected.to_csv('DataIG.csv', index=False)
    
    # Visualización (Opcional)
    plot_ig_scores(ig_scores)
    
    print("Proceso completado. Archivo DataIG.csv creado.")

if __name__ == '__main__':
    main()
