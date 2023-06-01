"""
Práctica 2. DIAGRAMA DE VORONOI Y CLUSTERING
Rubén Gómez Blanco y Adrián Sanjuán Espejo
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore")


# #############################################################################
# Aquí tenemos definido el sistema X de 1500 elementos (personas) con dos estados
archivo1 = "R:/QUINTO/GCOMP/Practica 2/Personas_en_la_facultad_matematicas.txt"
archivo2 = "R:/QUINTO/GCOMP/Practica 2/Grados_en_la_facultad_matematicas.txt"
X = np.loadtxt(archivo1)
Y = np.loadtxt(archivo2)
labels_true = Y[:,0]

header = open(archivo1).readline()
print("Set of environment variable values (1500):")
print(header)
print(X)

#Si quisieramos estandarizar los valores del sistema, haríamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

#Envolvente convexa, envoltura convexa o cápsula convexa 
hull = ConvexHull(X)
convex_hull_plot_2d(hull)

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.xlabel('Estrés')
plt.ylabel('Rock')
plt.title('Envolvente convexa de los datos')
fig1 = plt.gcf()
plt.show()
fig1.savefig('envconvx.png')

#EJERCICIO 1
print('\n----EJERCICIO 1----\n')
# Obtener el coeficiente de Silhouette para distinto número de vecinades (de 2 a 15), 
# graficar estos valores en funcion de la vecindad y decidir el numero de clusters óptimo.
#Mostrar clasificación y diagrama de Voronoi gráficamente

#Maximum number of clusters
n=15

'''
Funcion que encuentra el numero de clusters (entre 2 y 15) con el que se obtiene el mejor
valor del coeficiente de Silhouette
input:
    X: dataset a evaluar
output:
    max_k: numero de clusters optimo
    max_silhouette: maximo valor del coeficiente de Silhouette
    silhouette_values: vector con todos los valores de Silhouette obtenidos para todos los epsilon
'''
def encontrar_valores_kmeans(X):
    silhouette_values = []
    max_silhouette = 0
    max_k = 0
    #Usamos la inicialización aleatoria "random_state=0"
    for n_clusters in range(2,n+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        #No especificamos la métrica porque por defecto se utiliza la métrica euclidiana
        labels = kmeans.labels_
        sil = metrics.silhouette_score(X, labels)
        silhouette_values.append(sil)
        if max_silhouette < sil:
            max_silhouette = sil
            max_k = n_clusters
    return max_k, max_silhouette, silhouette_values

max_k, max_silhouette, silhouette_values=encontrar_valores_kmeans(X)

#Observamos los valores obtenidos para cada n_clusters entre 2 y 15
#print(silhouette_values)

plt.plot(range(2,16),silhouette_values)
plt.title('Silhouette values in terms of number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette value')
fig2 = plt.gcf()
plt.show()
fig2.savefig('svkmeans.png')

#Volvemos a ejecutar el algoritmo k_means con el valor de n_clusters que
#maximiza el coeficiente de Silhouette, en este caso es 3
kmeans = KMeans(n_clusters=max_k, random_state=0).fit(X)
labels = kmeans.labels_
silhouette_values.append(metrics.silhouette_score(X, labels))


fig3,ax = plt.subplots(1,1)
plt.title('Fixed number of KMeans clusters: %d' % max_k )
plt.xlabel('Estrés')
plt.ylabel('Rock')
def graficar_clusters(labels,X):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(8,4))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)
    ax.set_xlim([-2.3,2.3])
    ax.set_ylim([-2.3,1.7])

diagram=Voronoi(kmeans.cluster_centers_)
voronoi_plot_2d(diagram,ax)
graficar_clusters(labels,X)
plt.show()
fig3.savefig('kmeansclusters.png')

#Coeficiente de Silhouette
print("Maximum Silhouette Coefficient: %0.3f" % max_silhouette)
#Número de clusters óptimo
print("Number of clusters that maximize Silhouette coefficient (max_k): ", max_k)
# Índice de los centros de vencindades o regiones de Voronoi para cada elemento (punto) 
print("Voronoi Centers for max_k clusters: ")
print(kmeans.cluster_centers_)
# Etiqueta de cada elemento (punto)
print("Labels for each data point for max_k clusters: ")
print(labels)
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))


#EJERCICIO 2
print('\n----EJERCICIO 2----\n')
# Obtener el coeficiente de Silhouette para distinto número de el parametro epsilon, 
# graficar estos valores en funcion del epsilon y decidir el numero de clusters óptimo.
#Mostrar clasificación y comparar gráficamente con el apartado 1
#Todo esto para las distancias euclideana y Manhattan

'''
Funcion que encuentra el epsilon con el que se obtiene el mejor
valor del coeficiente de Silhouette
input:
    metrica: metrica a utilizar por el algoritmo DBSCAN
    pasos: array con todos los valores de epsilon que se van a comprobar 
    min_s: número de elementos mínimo para el algoritmo
    X: dataset a evaluar
output:
    max_k: numero de clusters optimo
    max_silhouette: maximo valor del coeficiente de Silhouette
    silhouette_values: vector con todos los valores de Silhouette obtenidos para todos los epsilon
'''
def encontrar_valores_dbscan(metrica,pasos,min_s,X):
    silhouette_values = []
    max_silhouette = -1
    max_e = 0
    for epsilon in pasos:
        db = DBSCAN(eps=epsilon, min_samples=min_s, metric=metrica).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        dblabels = db.labels_
        sil = metrics.silhouette_score(X, dblabels)
        silhouette_values.append(sil)
        if max_silhouette < sil:
            max_silhouette = sil
            max_e = epsilon
    return silhouette_values,max_silhouette,max_e

#PARAMETROS A UTILIZAR EN EL DBSCAN
pasos = np.arange(0.1,0.4,0.01)
#min_s = 15 #UTILIZAR MIN_S=15 PARA QUE TENGA SENTIDO EL PROBLEMA
min_s = 10

print('-------------------')
print('DISTANCIA EUCLIDEA:')
print('-------------------')

metrica='euclidean'
silhouette_values,max_silhouette,max_e=encontrar_valores_dbscan(metrica,pasos,min_s,X)

#Observamos los valores obtenidos para cada epsilon
#print(silhouette_values,max_silhouette,max_e )
plt.plot(pasos,silhouette_values)
plt.title('Silhouette values in terms of epsilon value (euclidean)')
plt.xlabel('Epsilon value')
plt.ylabel('Silhouette value')
fig6 = plt.gcf()
plt.show()
fig6.savefig('sveucl.png')


#Volvemos a ejecutar el algoritmo DBSCAN con el valor de eps que
#maximiza el coeficiente de Silhouette
db = DBSCAN(eps=max_e, min_samples=min_s, metric=metrica).fit(X)
dblabels_e = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(dblabels_e)) - (1 if -1 in dblabels_e else 0)
n_noise_ = list(dblabels_e).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, dblabels_e))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, dblabels_e))
#Eps óptimo
print("Epsilon that maximize Silhouette coefficient (max_e): ", max_e)

# Representamos el resultado con un plot
fig7,ax = plt.subplots(1,1)
plt.xlabel('Estrés')
plt.ylabel('Rock')
plt.title('Estimated number of DBSCAN clusters with euclidean metric: %d' % n_clusters_)
graficar_clusters(dblabels_e,X)
plt.show()
fig7.savefig('clusteucl.png')
print('-------------------')
print('DISTANCIA MANHATTAN:')
print('-------------------')    

metrica='manhattan'
silhouette_values,max_silhouette,max_e=encontrar_valores_dbscan(metrica,pasos,min_s,X)

#Observamos los valores obtenidos para cada epsilon
#print(silhouette_values,max_silhouette,max_e )
plt.plot(pasos,silhouette_values)
plt.title('Silhouette values in terms of epsilon value (manhattan)')
plt.xlabel('Epsilon value')
plt.ylabel('Silhouette value')
fig4 = plt.gcf()
plt.show()
fig4.savefig('svmanh.png')


#Volvemos a ejecutar el algoritmo DBSCAN con el valor de eps que
#maximiza el coeficiente de Silhouette
db = DBSCAN(eps=max_e, min_samples=min_s, metric='manhattan').fit(X)
dblabels_m = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(dblabels_m)) - (1 if -1 in dblabels_m else 0)
n_noise_ = list(dblabels_m).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, dblabels_m))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, dblabels_m))
#Eps óptimo
print("Epsilon that maximize Silhouette coefficient (max_e): ", max_e)

# Representamos el resultado con un plot
fig5,ax = plt.subplots(1,1)
plt.xlabel('Estrés')
plt.ylabel('Rock')
plt.title('Estimated number of DBSCAN clusters with manhattan metric: %d' % n_clusters_)
graficar_clusters(dblabels_m,X)
plt.show()
fig5.savefig('clustmanh.png')


#EJERCICIO 3
print('\n----EJERCICIO 3----\n')
#Predecir el Grado de las personas con coordenadas a:=(0, 0) y b:=(0, -1)

'''
predice: Funcion que predice la clase a la que pertenece cada punto de la lista pasada por parámetro
input:
    lista: array de puntos a predecir
output:
    clases_pred: array de prediccciones para cada punto de la lista
'''
def predice(lista):
    clases_pred = kmeans.predict(problem)
    for i in range(len(lista)):
        print("El alumno de coordenadas",problem[i],"pertenece al Grado",clases_pred[i])
    return clases_pred


problem = np.array([[0, 0], [0, -1]])
clases_pred = predice(problem)

# Representamos el resultado graficamente
fig8,ax = plt.subplots(1,1)
plt.xlabel('Estrés')
plt.ylabel('Rock')
plt.title('Representación gráfica del problema a predecir con KMeans')

graficar_clusters(labels,X)
ax.plot(problem[:,0],problem[:,1],'o', markersize=12, markerfacecolor="red")

plt.show()
fig8.savefig('results.png')