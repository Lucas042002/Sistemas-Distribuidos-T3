#-------------------------------------------------------
# Load Data from File: KDDTrain.txt
#--------------------------------------------------------

import numpy   as np
import pandas as pd
import os
#import utility_etl  as ut

# Load parameters from config.csv
def config():
    config = pd.read_csv("config.csv", header=None)
    return config

# Beginning 

def guardar_array_a_csv(array_datos, nombre_archivo="Data.csv"):
    df = pd.DataFrame(array_datos)
    df.to_csv(nombre_archivo, index=False, header=False)
# Definimos los conjuntos de clases de ataque
ataque_dos = {'neptune', 'teardrop', 'smurf', 'pod', 'back', 'land', 
              'apache2', 'processtable', 'mailbomb', 'udpstorm'}
ataque_probe = {'ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan'}

# Función para convertir clases en formato numérico
def convertir_clase(clase_original):
    if clase_original in ataque_dos:
        return 2  # Ataque DOS
    elif clase_original in ataque_probe:
        return 3  # Ataque Probe
    elif clase_original == 'normal':
        return 1  # Normal

def guardar_datos_clase(datos, nombre_archivo):
    df = pd.DataFrame(datos)
    df.to_csv(nombre_archivo, index=False, header=False)

def separar_clases(data):
    # Crear listas para almacenar los datos de cada clase
    data_new=[]
    datos_clase1 = []  # Clase Normal
    datos_clase2 = []  # Clase Ataque DOS
    datos_clase3 = []  # Clase Ataque Probe
    Y_vector = []
    # Recorrer el array y clasificar cada fila según el valor de la penúltima columna
    for fila in data:
        # Convertimos el valor de la penúltima columna usando nuestra función
        clase_numerica = convertir_clase(fila[-2])  # Penúltima columna es la clase original
        nueva_fila = fila[:-2] # Conservar todos los elementos menos el penúltimo        
        data_new.append(nueva_fila)
        Y_vector.append(clase_numerica)
        # Clasificamos cada fila según su clase
        if clase_numerica == 1:
            datos_clase1.append(nueva_fila)
        elif clase_numerica == 2:
            datos_clase2.append(nueva_fila)
        elif clase_numerica == 3:
            datos_clase3.append(nueva_fila)
    # Guardamos cada clase en su archivo correspondiente
    guardar_datos_clase(datos_clase1, "class1.csv")
    guardar_datos_clase(datos_clase2, "class2.csv")
    guardar_datos_clase(datos_clase3, "class3.csv")

    print("Los archivos classe1.csv, class2.csv y class3.csv se han guardado con éxito.")
    return np.array(data_new)


def pasar_a_numeros(raw_data):
    # Diccionarios para asignar valores numéricos a cada categoría
    protocolo_mapping = {'tcp': 1, 'udp': 2, 'icmp': 3}
    servicio_mapping = {servicio: idx for idx, servicio in enumerate(raw_data[2].unique())}
    flag_mapping = {flag: idx for idx, flag in enumerate(raw_data[3].unique())}

    # Aplicar las conversiones utilizando el método .map()
    raw_data[1] = raw_data[1].map(protocolo_mapping)
    raw_data[2] = raw_data[2].map(servicio_mapping)
    raw_data[3] = raw_data[3].map(flag_mapping)

    
    return raw_data.copy()

##REVISAR##
def seleccionar_muestras(archivo_datos, archivo_indices, M):
    # Leer los datos y los índices desde los archivos
    datos = pd.read_csv(archivo_datos)
    indices = pd.read_csv(archivo_indices, header=None).squeeze("columns")
    
    # Seleccionar las primeras M filas de los índices especificados
    indices_muestras = indices[:M] - 2
    #Para ordenar las muestras
    muestras_seleccionadas = datos.loc[indices_muestras].copy()
    # muestras_seleccionadas['index_original'] = indices_muestras.values  # Guardar los índices originales para ordenar después

    return muestras_seleccionadas

def unir_clases():

    # Determinar el valor de M basado en el tamaño de idx_class1.csv
    indices_clase1 = pd.read_csv("idx_class1.csv", header=None).squeeze("columns")
    M = len(indices_clase1)
    # Seleccionar muestras para cada clase usando los primeros M índices dados
    muestras_clase1 = seleccionar_muestras("Data.csv", "idx_class1.csv", M)
    muestras_clase2 = seleccionar_muestras("Data.csv", "idx_class2.csv", M)
    muestras_clase3 = seleccionar_muestras("Data.csv", "idx_class3.csv", M)
    muestras_clase1['clase'] = 1
    muestras_clase2['clase'] = 2
    muestras_clase3['clase'] = 3

    # Unir (apilar) todas las muestras seleccionadas en un solo DataFrame
    data_combined = pd.concat([muestras_clase1, muestras_clase2, muestras_clase3], ignore_index=True)

    # Guardar el archivo combinado
    data_combined.to_csv("DataClasss.csv", index=False, header=False)
    print("Archivo 'DataClasss.csv' generado con éxito.")
    return data_combined

def cargar_datos():
    raw_data = pd.read_csv("KDDTrain.txt", header=None)
    processed_data = pasar_a_numeros(raw_data)
    processed_data = np.array(processed_data)
    processed_data = separar_clases(processed_data)
    guardar_array_a_csv(processed_data)  
    return unir_clases()

def main():
    cargar_datos()

      
if __name__ == '__main__':   
	 main()

