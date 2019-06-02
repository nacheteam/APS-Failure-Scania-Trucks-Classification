# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

np.random.seed(123456789)

def leeFichero(fichero="../dataset/aps_failure_training_set.csv"):
    print("Leyendo el fichero...\n\n\n")
    f = open('../dataset/aps_failure_training_set.csv', 'r')
    dataset = []
    with f as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        for line in file:
            if i>=21:
                dataset.append(line)
            i+=1
    return dataset

def eliminaCaracteristicasMalas(dataset):
    print("Comprobamos las características que tienen más de un 70% de NAs...")
    dataset = np.array(dataset)
    deberiamos_quitarla =[False]*len(dataset[0])
    numeros_na = [0]*len(dataset[0])
    for d in dataset:
        for i in range(len(d)):
            if "na" in d[i]:
                numeros_na[i]+=1

    for i in range(len(numeros_na)):
        if (numeros_na[i]/len(dataset))>0.7:
            deberiamos_quitarla[i]=True


    indices =[]
    for deberiamos,index in zip(deberiamos_quitarla,range(len(deberiamos_quitarla))):
        if deberiamos:
            indices.append(index)
            print("Debemos quitar la característica " + str(index))
    print("")

    dataset1 = np.delete(dataset,indices,1)
    print("Número de características original: " + str(len(dataset[0])))
    print("Número de características nuevo: " + str(len(dataset1[0])))
    return dataset1

def eliminaInstanciasMalas(dataset1):
    instancias_malas = []
    for i in range(len(dataset1)):
        num_nas = 0
        for v in dataset1[i]:
            if "na" in v:
                num_nas+=1
        if num_nas/len(dataset1[i])>0.15 and dataset1[i][0]=="neg":
            instancias_malas.append(i)

    print("\n\n\nTenemos " + str(len(instancias_malas)) + " instancias negativas con más de un 15% de NAs")

    dataset2 = np.delete(dataset1,instancias_malas,0)
    print("Número de instancias del dataset1: " + str(len(dataset1)))
    print("Número de instancias del dataset nuevo: " + str(len(dataset2)))
    print("Número de instancias negativas: " + str(len([d for d in dataset2 if d[0]=="neg"])))
    print("Número de instancias positivas: " + str(len([d for d in dataset2 if d[0]=="pos"])))
    return dataset2

def convierteNumpy(dataset2):
    print("Convertimos el dataset a numérico con formato numpy")
    # Convertimos los datos a formato numpy
    for i in range(len(dataset2)):
        nueva_fila = []
        for v in dataset2[i]:
            if "na" in v:
                nueva_fila.append(np.nan)
            #CONVIERTO LAS ETIQUETAS A NÚMEROS
            elif 'neg' in v:
                nueva_fila.append(0)
            elif 'pos' in v:
                nueva_fila.append(1)
            else:
                nueva_fila.append(float(v))
        dataset2[i]=nueva_fila
    return dataset2

def imputaDatos(dataset2, imputacion="mediana"):
    # Vamos a hacer dos tipos de imputación de valores
    dataset_imputado=[]
    if imputacion=="media":
        print("Hacemos imputación de valores con la media")
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataset_imputado = imp.fit_transform(dataset2)
    elif imputacion=="mediana":
        print("Hacemos imputación de valores con la mediana")
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        dataset_imputado = imp.fit_transform(dataset2)
    else:
        print("Ese no es un método de imputacion")
        dataset_imputado=dataset2
    return dataset_imputado

def visualizaDatos(dataset,labels):
    # Pinto el dataset de la mediana en dos dimensiones
    dataset_reduced = TSNE(n_components=2).fit_transform(dataset)
    neg = np.array([dataset_reduced[i] for i in range(len(dataset_reduced)) if labels[i]==0])
    pos = np.array([dataset_reduced[i] for i in range(len(dataset_reduced)) if labels[i]==1])
    plt.scatter(neg[:,0], neg[:,1], c="green", label="Negativo")
    plt.scatter(pos[:,0], pos[:,1], c="red", label="Positivo")
    plt.legend()
    plt.title("Conjunto de datos")
    plt.show()

def obtenerDatosTrain(fichero="../dataset/aps_failure_training_set.csv", imputacion="mediana"):
    dataset = leeFichero(fichero)
    dataset1 = eliminaCaracteristicasMalas(dataset)
    dataset2 = eliminaInstanciasMalas(dataset1)
    dataset3 = convierteNumpy(dataset2)
    dataset4 = imputaDatos(dataset3, imputacion=imputacion)
    labels = dataset4[:,0]
    dataset = dataset4[:,1:]
    return dataset,labels

def obtenerDatosTest(fichero="../dataset/aps_failure_test_set.csv", imputacion="mediana"):
    dataset = leeFichero(fichero)
    dataset1 = eliminaCaracteristicasMalas(dataset)
    dataset2 = convierteNumpy(dataset1)
    dataset3 = imputaDatos(dataset2, imputacion=imputacion)
    labels = dataset3[:,0]
    dataset = dataset3[:,1:]
    return dataset,labels
