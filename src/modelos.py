import csv
import numpy as np
import preprocesamiento
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
from sklearn import metrics
from imblearn.over_sampling import SMOTE
# Modelos
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn import linear_model

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

def obtenScores(clasificador, dataset_test, labels_test, nombre="SGD"):
    pred = clasificador.predict(dataset_test)
    score = clasificador.score(dataset_test, labels_test)
    print("El score es: " + str(score))
    print("Precision: " + str(metrics.precision_score(labels_test,pred,average='weighted')))
    print("Recall: " + str(metrics.recall_score(labels_test,pred,average='weighted')))
    print("F1 Score: " + str(metrics.f1_score(labels_test,pred,average="weighted")))
    print("Matriz de confusión")
    nombres = ["neg","pos"]
    plot_confusion_matrix(labels_test, pred, classes=nombres,normalize = False,title='Matriz de confusión para ' + nombre)
    plt.show()

print("Leyendo los datasets")
dataset_train, labels_train = preprocesamiento.obtenerDatosTrain(imputacion="mediana")
dataset_test, labels_test = preprocesamiento.obtenerDatosTest(imputacion="mediana")
print("########################################################################\n\n")

'''
clf = svm.LinearSVC(random_state=123456789, tol=1e-5, verbose=True)
clf.fit(dataset_train, labels_train)
obtenScores(clf,dataset_test, labels_test, nombre="LinearSVM")
'''


clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6, verbose=True)
clf.fit(dataset_train, labels_train)
obtenScores(clf,dataset_test, labels_test)


'''
print("Ajustando SVM")
clasificador_svm = ajustaSVM(dataset_train, labels_train, ficherosave="svm_mediana.txt")
obtenScores(clasificador_svm,dataset_test, labels_test)
'''
