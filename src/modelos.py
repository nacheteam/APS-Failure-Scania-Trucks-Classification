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

# Fijamos la semilla
np.random.seed(123456789)

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    '''
    @brief Función encargada de computar y preparar la impresión de la matriz de confusión. Se puede extraer los resultados normalizados o sin normalizar. Basada en un ejemplo de scikit-learn
    @param y_true Etiquetas verdaderas
    @param y_pred Etiquetas predichas
    @param classes Distintas clases del problema (vector)
    @param normalize Booleano que indica si se normalizan los resultados o no
    @param title Título del gráfico
    @param cmap Paleta de colores para el gráfico
    '''
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

def ajustaSVM(dataset_train,labels_train):
    '''
    @brief Función que ajusta un modelo SVM
    @param dataset_train Conjunto de datos de entrenamiento
    @para labels_train Etiquetas del conjunto de entrenamiento
    @return Devuelve el clasificador SVM ajustado
    '''
    # SVM de parámetros gamma=0.1 y C=0.95
    clasificador_svm = svm.SVC(gamma=0.01, C=0.95, verbose=True)
    clasificador_svm.fit(dataset_train,labels_train)
    return clasificador_svm

def ajustaRandomForest(dataset_train, labels_train):
    '''
    @brief Función que ajusta un modelo Random Forest
    @param dataset_train Conjunto de datos de entrenamiento
    @para labels_train Etiquetas del conjunto de entrenamiento
    @return Devuelve el clasificador Random Forest ajustado
    '''
    # Parámetros obtenidos por un GridSearch
    clf = RandomForestClassifier(n_estimators=189, random_state=123456789, verbose=True)
    clf.fit(dataset_train, labels_train)
    return clf

def ajustaSGD(dataset_train, labels_train):
    '''
    @brief Función que ajusta un modelo Gradiente Descendente Estocástico
    @param dataset_train Conjunto de datos de entrenamiento
    @para labels_train Etiquetas del conjunto de entrenamiento
    @return Devuelve el clasificador Gradiente Descendente Estocástico ajustado
    '''
    # Ajustamos el SGD con 10.000 iteraciones como máximo y una tolerancia de 10^-6
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6, verbose=True)
    clf.fit(dataset_train, labels_train)
    return clf

def ajustaBoosting(dataset_train, labels_train):
    '''
    @brief Función que ajusta un modelo AdaBoost
    @param dataset_train Conjunto de datos de entrenamiento
    @para labels_train Etiquetas del conjunto de entrenamiento
    @return Devuelve el clasificador AdaBoost ajustado
    '''
    # Ajustamos AdaBoost con 100 estimadores
    clf = AdaBoostClassifier(n_estimators=100, random_state=123456789)
    clf.fit(dataset_train,labels_train)
    return clf

def ajustaRedNeuronal(dataset_train, labels_train):
    '''
    @brief Función que ajusta un modelo Perceptrón multicapa
    @param dataset_train Conjunto de datos de entrenamiento
    @para labels_train Etiquetas del conjunto de entrenamiento
    @return Devuelve el clasificador Perceptrón multicapa ajustado
    '''
    # Perceptrón multicapa con 3 capas ocultas y 100 unidades por capa. Máximo de iteraciones 10.000
    clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=10000, random_state=123456789, verbose=True)
    clf.fit(dataset_train, labels_train)
    return clf

def cota_eout_test(etest,N,tol):
    '''
    @brief Función que calcula la cota de Eout en función de etest
    @param etest Error obtenido en el conjunto de test
    @param N número de datos
    @param tol tolerancia o delta
    @return Devuelve el valor de la cota
    '''
    return etest + np.sqrt((1/(2*N))*np.log(2/tol))

def cota_eout_ein(ein,N,tol,vc):
    '''
    @brief Función que calcula la cota de Eout en función de ein
    @param ein Error cometido en el conjunto de train
    @param N número de datos
    @param tol tolerancia o delta
    @param vc Dimensión de vapnik chervonenkis
    @return Devuelve el valor de la cota
    '''
    return ein + np.sqrt((8/N)*np.log((4*((2*N)**(vc)+1))/tol))

def obtenScores(clasificador, dataset_test, labels_test, dataset_train, labels_train, nombre="SGD"):
    '''
    @brief Función que toma un clasificador y obtiene las distintas métricas y gráficas que
    evalúan su resultado en la predicción sobre el conjunto de test
    @param clasificador Clasificador que queremos evaluar
    @param dataset_test Conjunto de test
    @param labels_test Etiquetas del conjunto de test
    @param dataset_train Conjunto de entrenamiento
    @param labels_train Etiquetas del conjunto de entrenamiento
    @param nombre Nombre que saldrá en las gráficas
    '''
    # Obtenemos la predicción sobre el conjunto de test con el clasificador
    pred = clasificador.predict(dataset_test)
    # Obtenemos el score de la predicción anterior
    score = clasificador.score(dataset_test, labels_test)
    # Obtenemos la predicción sobre el conjunto de train
    pred_in = clasificador.predict(dataset_train)

    # Prints de resultados
    print("\n\nCon el fichero de TEST")
    print("El score es: " + str(score))
    print("Precision: " + str(metrics.precision_score(labels_test,pred,average='weighted')))
    print("Recall: " + str(metrics.recall_score(labels_test,pred,average='weighted')))
    print("F1 Score: " + str(metrics.f1_score(labels_test,pred,average="weighted")))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Calculamos la estimación de eout y ein
    ein = 1-metrics.accuracy_score(labels_train, pred_in)
    etest = 1-metrics.accuracy_score(labels_test, pred)
    print("Ein: " + str(ein))
    print("Etest: " + str(etest) + "\n")

    # Si el modelo es SGD o SVM entonces calculamos las cotas
    if nombre == "SGD":
        print("Cota Eout (con Ein): " + str(cota_eout_ein(ein,len(dataset_train),0.05,3)))
    elif nombre == "SVM":
        print("Cota Eout (con Ein): " + str(cota_eout_ein(ein,len(dataset_train),0.05,len(dataset_train[0]))))
    print("Cota Eout (con Etest): " + str(cota_eout_test(etest,len(dataset_train),0.05)))
    print("Matriz de confusión")

    # Pintamos la matriz de confusión
    nombres = ["neg","pos"]
    plot_confusion_matrix(labels_test, pred, classes=nombres,normalize = False,title='Matriz de confusión para ' + nombre)
    plt.show()

    # Pintamos la curva ROC
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


# Leemos los conjuntos de datos
dataset_train, labels_train, dataset_test, labels_test = preprocesamiento.obtenerDatos(imputacion="mediana")

# Random forest
print("########################################################################")
print("Ajustamos el modelo Random Forest")
print("########################################################################\n")
clf = ajustaRandomForest(dataset_train, labels_train)
obtenScores(clf,dataset_test, labels_test, dataset_train, labels_train,nombre="Random Forest")
input("Presione ENTER para continuar...")
print("\n\n")

# Gradiente descendente estocástico
print("########################################################################")
print("Ajustamos el modelo Gradiente Descendente Estocastico")
print("########################################################################\n")
clf = ajustaSGD(dataset_train, labels_train)
obtenScores(clf,dataset_test, labels_test, dataset_train, labels_train, nombre = "SGD")
input("Presione ENTER para continuar...")
print("\n\n")

# SVM
print("########################################################################")
print("Ajustamos el modelo SVM")
print("########################################################################\n")
clf = ajustaSVM(dataset_train, labels_train, ficherosave="svm_mediana_normalizacion.txt")
obtenScores(clf,dataset_test, labels_test, dataset_train, labels_train, nombre="SVM")
input("Presione ENTER para continuar...")
print("\n\n")

# AdaBoost
print("########################################################################")
print("Ajustamos el modelo AdaBoost")
print("########################################################################\n")
clf = ajustaBoosting(dataset_train, labels_train)
obtenScores(clf,dataset_test,labels_test, dataset_train, labels_train, nombre="AdaBoost")
input("Presione ENTER para continuar...")
print("\n\n")

# Red Neuronal
print("########################################################################")
print("Ajustamos el modelo Perceptrón Multicapa")
print("########################################################################\n")
clf = ajustaRedNeuronal(dataset_train, labels_train)
obtenScores(clf, dataset_test, labels_test, dataset_train, labels_train, nombre="Red Neuronal")
input("Presione ENTER para continuar...")
print("\n\n")


'''
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
