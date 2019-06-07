# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing

# Colocamos la semilla a 123456789
np.random.seed(123456789)
DEBUG = False # Indica si se hacen o no prints

def leeFichero(fichero="../dataset/aps_failure_training_set.csv"):
    '''
    @brief Función que lee los datos de un fichero del dataset de APS
    @param fichero Ruta del fichero a leer
    @return Devuelve una lista de python con los elementos puestos como strings
    '''
    if DEBUG:
        print("Leyendo el fichero...\n\n\n")
    # Abrimos el fichero
    f = open('../dataset/aps_failure_training_set.csv', 'r')
    dataset = []
    # Lo leemos con el módulo csv
    with f as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        # Para cada línea en el fichero
        for line in file:
            # Los datos empiezan en la línea 21
            if i>=21:
                # Añadimos la línea separada al dataset
                dataset.append(line)
            i+=1
    return dataset

def eliminaCaracteristicasMalas(dataset, porcentaje=0.7):
    '''
    @brief Función que analiza las características que no aportan suficiente información
    con el objeto de borrarlas del dataset
    @param dataset Dataset del que queremos comprobar si tenemos características
    con un alto número de NAs
    @param porcentaje Porcentaje umbral para marcar una característica como poco relevante
    @return Devuelve el dataset con las características con un alto número de
    NAs eliminadas
    '''
    if DEBUG:
        print("Comprobamos las características que tienen más de un 70% de NAs...")
    # Convertimos la lista de listas en un numpy array
    dataset = np.array(dataset)
    # Inicializamos un vector de booleanos a falso para marcar las características a quitar
    deberiamos_quitarla =[False]*len(dataset[0])
    # Mantenemos un conteo de NAs por característica
    numeros_na = [0]*len(dataset[0])

    # Para cada fila en el dataset
    for d in dataset:
        # Para cada característica
        for i in range(len(d)):
            # Si es un NA sumamos 1 en la posición correspondiente del conteo de NAs
            if "na" in d[i]:
                numeros_na[i]+=1

    # Recorremos el vector de conteo comprobando si
    for i in range(len(numeros_na)):
        # Si la proporción de NAs y elementos en el dataset es mayor que el porcentaje
        # es que hay mas de un porcentaje % de NAs en la característica
        if (numeros_na[i]/len(dataset))>porcentaje:
            # La marcamos para quitar
            deberiamos_quitarla[i]=True


    # Vector de índices de características para quitar
    indices =[]
    for deberiamos,index in zip(deberiamos_quitarla,range(len(deberiamos_quitarla))):
        # Si hemos marcado la característica añadimos el índice
        if deberiamos:
            indices.append(index)
            if DEBUG:
                print("Debemos quitar la característica " + str(index))
    if DEBUG:
        print("")

    # Eliminamos las columnas de las características que debemos quitar
    dataset1 = np.delete(dataset,indices,1)
    if DEBUG:
        print("Número de características original: " + str(len(dataset[0])))
        print("Número de características nuevo: " + str(len(dataset1[0])))
    return dataset1

def eliminaInstanciasMalas(dataset1, porcentaje=0.15):
    '''
    @brief Función que comrpueba si una instancia tiene más de un cierto porcentaje
    de valores NA para eliminarla por no tener suficiente información
    @param dataset1 Dataset del que queremos eliminar instancias irrelevantes
    @param porcentaje Porcentaje umbral de NAs para marcar la instancia para eliminar
    @param Devuelve un dataset con las instancias poco útiles para eliminar.
    '''
    # Mantenemos los índices de las instancias con muchos NAs
    instancias_malas = []
    # Para cada elemento del dataset
    for i in range(len(dataset1)):
        # Contamos el número de NAs en la instancia
        num_nas = 0
        # Para cada característica
        for v in dataset1[i]:
            # Si es NA sumamos 1
            if "na" in v:
                num_nas+=1
        # Si excede nuestro porcentaje umbral entonces añadimos el índice para quitarla
        if num_nas/len(dataset1[i])>porcentaje and dataset1[i][0]=="neg":
            instancias_malas.append(i)

    if DEBUG:
        print("\n\n\nTenemos " + str(len(instancias_malas)) + " instancias negativas con más de un 15% de NAs")

    # Eliminamos las instancias
    dataset2 = np.delete(dataset1,instancias_malas,0)
    if DEBUG:
        print("Número de instancias del dataset1: " + str(len(dataset1)))
        print("Número de instancias del dataset nuevo: " + str(len(dataset2)))
        print("Número de instancias negativas: " + str(len([d for d in dataset2 if d[0]=="neg"])))
        print("Número de instancias positivas: " + str(len([d for d in dataset2 if d[0]=="pos"])))
    return dataset2

def convierteNumpy(dataset2):
    '''
    @brief Función que convierte a formato numpy el dataset
    @param dataset2 Dataset que queremos convertir a formato numpy
    '''
    if DEBUG:
        print("Convertimos el dataset a numérico con formato numpy")
    # Convertimos los datos a formato numpy
    for i in range(len(dataset2)):
        nueva_fila = []
        for v in dataset2[i]:
            # Si es un NA lo convertimos en np.nan
            if "na" in v:
                nueva_fila.append(np.nan)
            # Si es de clase negativa lo pasamos a 0
            elif 'neg' in v:
                nueva_fila.append(0)
            # Si es de clase positiva lo convertimos a 1
            elif 'pos' in v:
                nueva_fila.append(1)
            # En otro caso lo pasamos a float
            else:
                nueva_fila.append(float(v))
        dataset2[i]=nueva_fila
    return dataset2

def imputaDatos(dataset2, imputacion="mediana"):
    '''
    @brief Función que se encarga de imputar datos, es decir, sustituir los NA
    por otros valores en función de la estrategia definida por el parámetro
    imputacion
    @param dataset2 Dataset sobre el que queremos hacer la imputación de valores
    @param imputacion Procedimiento a emplear para imputar datos. En este caso
    pueden ser o bien media o bien mediana
    @return Devuelve el conjunto de datos sin valores perdidos
    '''
    # Vamos a hacer dos tipos de imputación de valores
    dataset_imputado=[]
    # Si la estrategia es media
    if imputacion=="media":
        if DEBUG:
            print("Hacemos imputación de valores con la media")
        # Indicamos que los valores perdidos son np.nan y los reemplazamos por la media
        # de valores de la característica
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataset_imputado = imp.fit_transform(dataset2)
    # Si la estrategia es mediana
    elif imputacion=="mediana":
        if DEBUG:
            print("Hacemos imputación de valores con la mediana")
        # Indicamos que los valores perdidos son np.nan y los reemplazamos por la mediana
        # de valores de la característica
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        dataset_imputado = imp.fit_transform(dataset2)
    # Si no es ni media ni mediana no hacemos nada
    else:
        if DEBUG:
            print("Ese no es un método de imputacion")
        dataset_imputado=dataset2
    return dataset_imputado

def balanceaDataset(dataset):
    '''
    @brief Función que balancea las clases negativas y positivas del dataset
    @param dataset Dataset sobre el que queremos aplicar el balanceo
    @return Devuelve un dataset con las clases positiva y negativa equilibradas
    '''
    if DEBUG:
        print("Balanceando el dataset")
    # Aplicamos la estrategia SMOTE con un porcentaje del 40%
    sm = SMOTE(sampling_strategy=0.4,random_state = 123456789)
    X_res, y_res = sm.fit_resample(dataset[:,1:],dataset[:,0])
    return X_res, y_res

def nComponentsCriterion(vratio, explained_var=0.95):
    '''
    @brief Función que da el número de componentes que explican al menos un 95%
    de la varianza dado el resultado de un PCA o FA.
    @param vratio Porcentaje de varianza explicada para cada componente
    @param explained_var Mínimo que requerimos de varianza explicada
    @return Devuelve el número de componentes que explican al menos un
    95% de la varianza
    '''
    index = 0
    sum_var = 0
    # Vamos acumulando la varianza
    for i in range(len(vratio)):
        sum_var+=vratio[i]
        # Si pasamos el porcentaje que queremos actualizamos el índice con el actual
        if sum_var>=explained_var:
            index=i
            break
    # Devolevemos el indice que será el número de componentes
    return index

def reduceDimensionalidad(train, test):
    '''
    @brief Función que reduce la dimensionalidad del dataset aplicando Factor Analysis
    @param train Dataset de entrenamiento
    @param test Dataset de test
    @return Devuelve ambos conjuntos con dimensionalidad reducida
    '''
    # Creamos Factor Analysis y lo ajustamos solo con el train
    fa = FactorAnalysis().fit(train)
    # Obtenemos la varianza por componente
    fa_transform = fa.transform(train)
    explained_variance = np.var(fa_transform, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    # Sacamos el número de componentes según el criterio
    n_comp = nComponentsCriterion(explained_variance_ratio)
    # Ajustamos de nuevo el train con el número de componentes óptimo
    fa = FactorAnalysis(n_components=n_comp).fit(train)
    # Reducimos la dimensionalidad de ambos conjuntos
    return fa.transform(train), fa.transform(test)


def visualizaDatos(dataset,labels):
    '''
    @brief Función que visualiza los datos haciendo una reducción de dimensionalidad TSNE
    @param dataset Dataset que queremos representar
    @param labels Etiquetas de las instancias del conjunto
    '''
    # Pinto el dataset en dos dimensiones
    dataset_reduced = TSNE(n_components=2).fit_transform(dataset)
    neg = np.array([dataset_reduced[i] for i in range(len(dataset_reduced)) if labels[i]==0])
    pos = np.array([dataset_reduced[i] for i in range(len(dataset_reduced)) if labels[i]==1])
    plt.scatter(neg[:,0], neg[:,1], c="green", label="Negativo")
    plt.scatter(pos[:,0], pos[:,1], c="red", label="Positivo")
    plt.legend()
    plt.title("Conjunto de datos")
    plt.show()



def obtenerDatosTrain(fichero="../dataset/aps_failure_training_set.csv", imputacion="mediana"):
    '''
    @brief Función que obtiene los datos de train procesados
    @param fichero Fichero del que leer los datos
    @param imputacion Técnica para imputar valores perdidos
    @return Devuelve el dataset de train y sus etiquetas
    '''
    # Leemos el fichero
    dataset = leeFichero(fichero)
    # Eliminamos las características con más de un cierto porcentaje de NAs
    dataset1 = eliminaCaracteristicasMalas(dataset)
    # Elimina las instancias que no aportan información
    dataset2 = eliminaInstanciasMalas(dataset1)
    # Convertimos a formato numpy
    dataset3 = convierteNumpy(dataset2)
    # Imputamos valores perdidos
    dataset4 = imputaDatos(dataset3, imputacion=imputacion)
    # Equilibramos las clases
    dataset, labels = balanceaDataset(dataset4)
    return dataset,labels

def obtenerDatosTest(fichero="../dataset/aps_failure_test_set.csv", imputacion="mediana"):
    '''
    @brief Función que obtiene los datos de test procesados
    @param fichero Fichero del que leer los datos
    @param imputacion Técnica para imputar valores perdidos
    @return Devuelve el dataset de test y sus etiquetas
    '''
    # Leemos el fichero
    dataset = leeFichero(fichero)
    # Eliminamos las características con más de un cierto porcentaje de NAs
    dataset1 = eliminaCaracteristicasMalas(dataset)
    # Convertimos a formato numpy
    dataset2 = convierteNumpy(dataset1)
    # Imputamos valores perdidos
    dataset3 = imputaDatos(dataset2, imputacion=imputacion)
    labels = dataset3[:,0]
    dataset = dataset3[:,1:]
    return dataset,labels

def obtenerDatos(fichero_train = "../dataset/aps_failure_training_set.csv", fichero_test ="../dataset/aps_failure_test_set.csv", imputacion="mediana"):
    '''
    @brief Función que obtiene los datos de train y test procesados
    @param fichero_train Ruta al fichero con los datos de train
    @param fichero_testt Ruta al fichero con los datos de test
    @param imputacion Técnica de imputación de valores perdidos
    @return Devuelve tanto el conjunto de test como el de train con sus etiquetas
    '''
    # Obtenemos los datos de train
    data_train, labels_train = obtenerDatosTrain(fichero_train, imputacion)
    # Obtenemos los datos de test
    data_test, labels_test = obtenerDatosTest(fichero_test, imputacion)
    # Escalamos los datos
    std_scaler = preprocessing.StandardScaler().fit(data_train)
    dataset_train_norm = std_scaler.transform(data_train)
    dataset_test_norm = std_scaler.transform(data_test)
    #print("Reduciendo dimensionalidad")
    #data_train_red, data_test_red = reduceDimensionalidad(data_train, data_test)
    #print("Se ha reducido de " + str(len(data_train[0])) + " características a " + str(len(data_train_red[0])))
    #print("Fin lectura y preprocesamiento")
    return dataset_train_norm, labels_train, dataset_test_norm, labels_test
