# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

np.random.seed(123456789)

print("Leyendo el fichero...\n\n\n")
f = open('./dataset/aps_failure_training_set.csv', 'r')
dataset = []
with f as csvfile:
    file = csv.reader(csvfile, delimiter=',', quotechar='|')
    i=0
    for line in file:
        if i>=21:
            dataset.append(line)
        i+=1

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

instancias_malas = []
for i in range(len(dataset1)):
    num_nas = 0
    for v in dataset1[i]:
        if "na" in v:
            num_nas+=1
    if num_nas/len(dataset1[i])>0.15 and dataset[i][0]=="neg":
        instancias_malas.append(i)

print("\n\n\nTenemos " + str(len(instancias_malas)) + " instancias negativas con más de un 15% de NAs")

dataset2 = np.delete(dataset1,instancias_malas,0)
del dataset
print("Número de instancias del dataset1: " + str(len(dataset1)))
print("Número de instancias del dataset nuevo: " + str(len(dataset2)))
del dataset1

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

# Vamos a hacer dos tipos de imputación de valores
print("Hacemos imputación de valores con la media")
dataset_media = []
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset_media = imp.fit_transform(dataset2)

print("Hacemos imputación de valores con la mediana")
dataset_mediana = []
imp = SimpleImputer(missing_values=np.nan, strategy='median')
dataset_mediana = imp.fit_transform(dataset2)

print("Visualizamos en primer lugar el dataset obtenido con la media")
labels = dataset_media[:,0]

# Pinto el dataset de la media en dos dimensiones
'''
dataset_media_reduced = TSNE(n_components=2).fit_transform(dataset_media)
neg = np.array([dataset_media_reduced[i] for i in range(len(dataset_media_reduced)) if labels[i]==0])
pos = np.array([dataset_media_reduced[i] for i in range(len(dataset_media_reduced)) if labels[i]==1])
plt.scatter(neg[:,0], neg[:,1], c="green", label="Negativo")
plt.scatter(pos[:,0], pos[:,1], c="red", label="Positivo")
plt.legend()
plt.title("Conjunto de datos imputados con la media")
plt.show()
'''

# Pinto el dataset de la mediana en dos dimensiones
dataset_mediana_reduced = TSNE(n_components=2).fit_transform(dataset_mediana)
neg = np.array([dataset_mediana_reduced[i] for i in range(len(dataset_mediana_reduced)) if labels[i]==0])
pos = np.array([dataset_mediana_reduced[i] for i in range(len(dataset_mediana_reduced)) if labels[i]==1])
plt.scatter(neg[:,0], neg[:,1], c="green", label="Negativo")
plt.scatter(pos[:,0], pos[:,1], c="red", label="Positivo")
plt.legend()
plt.title("Conjunto de datos imputados con la mediana")
plt.show()
