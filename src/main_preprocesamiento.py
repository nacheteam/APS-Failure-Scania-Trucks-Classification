import preprocesamiento

# Ponemos para visualizar los prints
preprocesamiento.DEBUG=True
# Obtenemos los datos
dataset_train, labels_train, dataset_test, labels_test = preprocesamiento.obtenerDatos()

# Los pintamos
print("Visualizamos los datos de train")
preprocesamiento.visualizaDatos(dataset_train, labels_train)
input("Presione ENTER para continuar...")

print("Visualizamos los datos de test")
preprocesamiento.visualizaDatos(dataset_test, labels_test)
input("Presione ENTER para continuar...")
