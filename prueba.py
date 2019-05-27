# -*- coding: utf-8 -*-

import csv
import numpy as np

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
        
print("\n\n\nTenemos " + str(len(instancias_malas)) + " instancias con más de un 15% de NAs")

dataset2 = np.delete(dataset,instancias_malas,0)
print("Número de instancias del dataset1: " + str(len(dataset1)))
print("Número de instancias del dataset nuevo: " + str(len(dataset2)))