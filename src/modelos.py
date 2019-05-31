import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
from sklearn import metrics
# Modelos
from sklearn import svm

np.random.seed(123456789)

def obtenerDatosTrain(imputacion):
    '''
    @brief Función que lee los datos, los preprocesa e imputa valores perdidos
    según la estrategia pasada por el argumento imputacion
    @param imputacion Cadena de texto que puede ser <media> o <mediana>
    para indicar la estrategia de imputación de valores perdidos a usar
    @return Devuelve dos numpy arrays, el primero es el conjunto de datos y el
    segundo las etiquetas.
    '''
    # Abrimos el fichero
    f = open('../dataset/aps_failure_training_set.csv', 'r')
    dataset = []
    # Lo leemos como un fichero csv
    with f as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        for line in file:
            # Sabemos que el contenido está a partir de la línea 21
            if i>=21:
                dataset.append(line)
            i+=1
    # Lo convertimos en numpy array
    dataset = np.array(dataset)
    deberiamos_quitarla =[False]*len(dataset[0])
    numeros_na = [0]*len(dataset[0])
    # Estudiamos el número de NA's por característica
    for d in dataset:
        for i in range(len(d)):
            if "na" in d[i]:
                numeros_na[i]+=1

    # Si tiene más de un 70% de NA's entonces la marcamos para quitarla
    for i in range(len(numeros_na)):
        if (numeros_na[i]/len(dataset))>0.7:
            deberiamos_quitarla[i]=True


    indices =[]
    # Calculamos los índices a quitar
    for deberiamos,index in zip(deberiamos_quitarla,range(len(deberiamos_quitarla))):
        if deberiamos:
            indices.append(index)

    # Eliminamos los índices de las características deficientes (en cuanto a número de NA's)
    dataset1 = np.delete(dataset,indices,1)

    # Obtenemos las instancias con más de un 15% de los atributos como NA's
    instancias_malas = []
    for i in range(len(dataset1)):
        num_nas = 0
        for v in dataset1[i]:
            if "na" in v:
                num_nas+=1
        if num_nas/len(dataset1[i])>0.15 and dataset[i][0]=="neg":
            instancias_malas.append(i)

    # Quitamos las instancias que aportan poca información
    dataset2 = np.delete(dataset1,instancias_malas,0)

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

    # Se aplica la imputación de valores según la media
    if imputacion=="media":
        dataset_media = []
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataset_media = imp.fit_transform(dataset2)
        return dataset_media[:,1:], dataset_media[:,0]
    # Se aplica la imputación de valores según la mediana
    elif imputacion=="mediana":
        dataset_mediana = []
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        dataset_mediana = imp.fit_transform(dataset2)
        return dataset_mediana[:,1:], dataset_mediana[:,0]
    # No se aplica imputación de valores y se devuelve el conjunto de datos con NA's
    else:
        print("No se ha elegido un método de imputación, devolvemos el dataset bruto.")
        return dataset2[:,1:], dataset2[:,0]

def obtenerDatosTest(imputacion):
    '''
    @brief Función que lee los datos de test
    @param imputacion Cadena de texto que puede ser <media> o <mediana>
    para indicar la estrategia de imputación de valores perdidos a usar
    @return Devuelve dos numpy arrays, el primero el conjunto de datos y el segundo
    las etiquetas del mismo.
    '''
    # Abrimos el fichero
    f = open('../dataset/aps_failure_test_set.csv', 'r')
    dataset = []
    # Lo leemos como un fichero csv
    with f as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        for line in file:
            # Sabemos que el contenido está a partir de la línea 21
            if i>=21:
                dataset.append(line)
            i+=1
    # Lo convertimos en numpy array
    dataset = np.array(dataset)

    deberiamos_quitarla =[False]*len(dataset[0])
    numeros_na = [0]*len(dataset[0])
    # Estudiamos el número de NA's por característica
    for d in dataset:
        for i in range(len(d)):
            if "na" in d[i]:
                numeros_na[i]+=1

    # Si tiene más de un 70% de NA's entonces la marcamos para quitarla
    for i in range(len(numeros_na)):
        if (numeros_na[i]/len(dataset))>0.7:
            deberiamos_quitarla[i]=True


    indices =[]
    # Calculamos los índices a quitar
    for deberiamos,index in zip(deberiamos_quitarla,range(len(deberiamos_quitarla))):
        if deberiamos:
            indices.append(index)

    # Eliminamos los índices de las características deficientes (en cuanto a número de NA's)
    dataset1 = np.delete(dataset,indices,1)

    # Convertimos los datos a formato numpy
    for i in range(len(dataset1)):
        nueva_fila = []
        for v in dataset1[i]:
            if "na" in v:
                nueva_fila.append(np.nan)
            #CONVIERTO LAS ETIQUETAS A NÚMEROS
            elif 'neg' in v:
                nueva_fila.append(0)
            elif 'pos' in v:
                nueva_fila.append(1)
            else:
                nueva_fila.append(float(v))
        dataset1[i]=nueva_fila

    # Se aplica la imputación de valores según la media
    if imputacion=="media":
        dataset_media = []
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataset_media = imp.fit_transform(dataset1)
        return dataset_media[:,1:], dataset_media[:,0]
    # Se aplica la imputación de valores según la mediana
    elif imputacion=="mediana":
        dataset_mediana = []
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        dataset_mediana = imp.fit_transform(dataset1)
        return dataset_mediana[:,1:], dataset_mediana[:,0]
    # No se aplica imputación de valores y se devuelve el conjunto de datos con NA's
    else:
        print("No se ha elegido un método de imputación, devolvemos el dataset bruto.")
        return dataset1[:,1:], dataset1[:,0]

print("Leyendo los conjuntos de datos")
dataset_train, labels_train = obtenerDatosTrain(imputacion="mediana")
dataset_test, labels_test = obtenerDatosTest(imputacion="mediana")
clasificador_svm = svm.SVC(gamma="auto", verbose=True)
print("Ajustando SVM")
clasificador_svm.fit(dataset_train,labels_train)
print("Guardamos la SVM en un fichero (svm.txt)")
joblib.dump(clasificador_svm,"svm.txt")
print("Obteniendo el score")
pred = clasificador_svm.predict(dataset_test)
score = clasificador_svm.score(dataset_test, labels_test)
print("El score de SVM es: " + str(score))
print("Precision: " + str(metrics.precision_score(labels_test,pred)))
print("Recall: " + str(metrics.recall_score(labels_test,pred)))
print("F1 Score: " + str(metrics.f1_score(labels_test,pred,average="binary")))
