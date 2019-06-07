import csv
import numpy as np
import preprocesamiento
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
# Modelos
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import preprocessing

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
        clasificador_svm = svm.SVC(gamma=0.01, C=0.95, verbose=True)
        clasificador_svm.fit(dataset_train,labels_train)
        if ficherosave!=None:
            joblib.dump(clasificador_svm,ficherosave)
        return clasificador_svm
    else:
        clasificador_svm = joblib.load(ficherosvm)
        return clasificador_svm

def ajustaRandomForest(dataset_train, labels_train):
    # Resultado del GridSearch
    clf = RandomForestClassifier(n_estimators=189, random_state=123456789, verbose=True)
    clf.fit(dataset_train, labels_train)
    return clf

def ajustaSGD(dataset_train, labels_train):
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6, verbose=True)
    clf.fit(dataset_train, labels_train)
    return clf

def ajustaBoosting(dataset_train, labels_train):
    clf = AdaBoostClassifier(n_estimators=100, random_state=123456789)
    clf.fit(dataset_train,labels_train)
    return clf

def  ajustaRedNeuronal(dataset_train, labels_train):
    clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=10000, random_state=123456789, verbose=True)
    clf.fit(dataset_train, labels_train)
    return clf

def obtenScores(clasificador, dataset_test, labels_test, dataset_train, labels_train, nombre="SGD"):
    pred = clasificador.predict(dataset_test)
    score = clasificador.score(dataset_test, labels_test)
    pred_in = clasificador.predict(dataset_train)
    score_in = clasificador.predict(dataset_train, labels_train)
    print("Con el fichero de TEST")
    print("El score es: " + str(score))
    print("Precision: " + str(metrics.precision_score(labels_test,pred,average='weighted')))
    print("Recall: " + str(metrics.recall_score(labels_test,pred,average='weighted')))
    print("F1 Score: " + str(metrics.f1_score(labels_test,pred,average="weighted")))
    print("#####################################################################")
    print("Ein: " + str(1-clasificador.accuracy_score(labels_train, pred_in, average="weighted" + "\n\n")))
    print("Eout: " + str(1-clasificador.accuracy_score(labels_test, pred, average="weighted" + "\n\n")))
    print("Matriz de confusión")
    nombres = ["neg","pos"]
    plot_confusion_matrix(labels_test, pred, classes=nombres,normalize = False,title='Matriz de confusión para ' + nombre)
    plt.show()
    print("Área bajo la curva ROC")
    fpr, tpr, threshold = metrics.roc_curve(labels_test,pred)
    roc_auc = metrics.auc(fpr,tpr)
    plt.title('Curva ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


dataset_train, labels_train, dataset_test, labels_test = preprocesamiento.obtenerDatos(imputacion="mediana")


# Random forest
clf = ajustaRandomForest(dataset_train, labels_train)
obtenScores(clf,dataset_test, labels_test, dataset_train, labels_train,nombre="Random Forest")
'''
# Gradiente descendente estocástico
clf = ajustaSGD(dataset_train, labels_train)
obtenScores(clf,dataset_test, labels_test, dataset_train, labels_train, nombre = "SGD")

# SVM
clf = ajustaSVM(dataset_train, labels_train, ficherosave="svm_mediana_normalizacion.txt")
obtenScores(clf,dataset_test, labels_test, dataset_train, labels_train nombre="SVM")

# AdaBoost
clf = ajustaBoosting(dataset_train, labels_train)
obtenScores(clf,dataset_test,labels_test, dataset_train, labels_train nombre="AdaBoost")

# Red Neuronal
clf = ajustaRedNeuronal(dataset_train, labels_train)
obtenScores(clf, dataset_test, labels_test, dataset_train, labels_train nombre="Red Neuronal")


# Grid search para Random Forest
random_forest = RandomForestClassifier(random_state=123456789)
parametros = {'n_estimators': np.round(np.linspace(100,200,10)).astype(int)}
print("vAMOS AL GRID")
grid = GridSearchCV(random_forest, parametros, cv=5,n_jobs=4, scoring='recall', verbose=True)
print("Fin")
grid.fit(dataset_train, labels_train)
print(grid.best_params_)

# Grid search para svm no acaba. Pruebo con

C_range = [0.1]
#gamma_range = np.logspace(-2,3,13)
param_grid = dict(C=C_range)
scores = ['recall']

for score in scores:
    clf = GridSearchCV(svm.SVC(kernel='rbf',class_weight = 'balanced'),param_grid = param_grid, cv=5,scoring = '%s_macro' % score)
    clf.fit(dataset_train,labels_train)
    print(clf.best_params_)
'''
