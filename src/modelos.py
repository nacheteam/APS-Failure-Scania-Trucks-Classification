import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
from sklearn import metrics
from imblearn.over_sampling import SMOTE
# Modelos
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

np.random.seed(123456789)

#
# plot_confusion_matrix
# @brief: Función encargada de computar y preparar la impresión de la matriz de confusión. Se puede extraer los resultados normalizados o sin normalizar. Basada en un ejemplo de scikit-learn
# @param: y_true. Etiquetas verdaderas
# @param: y_pred. Etiquetas predichas
# @param: classes. Distintas clases del problema (vector)
# @param: normalize. Booleano que indica si se normalizan los resultados o no
# @param: title. Título del gráfico
# @param: cmap. Paleta de colores para el gráfico
#

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión sin normalizar'

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    # Clases
    classes = [0,1,2,3,4,5,6,7,8,9]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalizar')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiquetas verdaderas',
           xlabel='Etiquetas predichas')

    # Rotar las etiquetas para su posible lectura
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Creación de anotaciones
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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
                nueva_fila.append(-1)
            elif 'pos' in v:
                nueva_fila.append(1)
            else:
                nueva_fila.append(float(v))
        dataset2[i]=nueva_fila

    sm = SMOTE(sampling_strategy=0.4,random_state = 123456789)
    # Se aplica la imputación de valores según la media
    if imputacion=="media":
        dataset_media = []
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataset_media = imp.fit_transform(dataset2)
        X_res, y_res = sm.fit_resample(dataset_media[:,1:],dataset_media[:,0])
        return X_res, y_res
    # Se aplica la imputación de valores según la mediana
    elif imputacion=="mediana":
        dataset_mediana = []
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        dataset_mediana = imp.fit_transform(dataset2)
        X_res, y_res = sm.fit_resample(dataset_mediana[:,1:],dataset_mediana[:,0])
        return X_res, y_res
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
                nueva_fila.append(-1)
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

def ajustaSVM(dataset_train,labels_train, ficherosvm=None, ficherosave=None):
    if ficherosvm==None:
        clasificador_svm = svm.SVC(gamma="auto", verbose=True)
        clasificador_svm.fit(dataset_train,labels_train)
        if ficherosave!=None:
            joblib.dump(clasificador_svm,ficherosave)
        return clasificador_svm
    else:
        clasificador_svm = joblib.load(ficherosvm)
        return clasificador_svm

def obtenScores(clasificador, dataset_test, labels_test):
    pred = clasificador.predict(dataset_test)
    score = clasificador.score(dataset_test, labels_test)
    print("El score es: " + str(score))
    print("Precision: " + str(metrics.precision_score(labels_test,pred,average='weighted')))
    print("Recall: " + str(metrics.recall_score(labels_test,pred,average='weighted')))
    print("F1 Score: " + str(metrics.f1_score(labels_test,pred,average="weighted")))
    print("Matriz de confusión")
    nombres = ["neg","pos"]
    plot_confusion_matrix(labels_test, pred, classes=nombres,normalize = False,title='Matriz de confusión para SVM')
    plt.show()

print("Leyendo los datasets")
dataset_train, labels_train = obtenerDatosTrain(imputacion="mediana")
dataset_test, labels_test = obtenerDatosTest(imputacion="mediana")
print("Ajustando SVM")
clasificador_svm = ajustaSVM(dataset_train, labels_train, ficherosave="svm_mediana.txt")
obtenScores(clasificador_svm,dataset_test, labels_test)
