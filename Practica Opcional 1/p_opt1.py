# -*- coding: utf-8 -*-
"""
Rubén Gómez Blanco
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString, Point

# ################################ PARTE 1 #####################################

#Generamos 1000 segmentos aleatorios, pero siempre serán los mismos

#Usaremos primero el concepto de coordenadas
X = []
Y = []

#Fijamos el modo aleatorio con una versión prefijada. NO MODIFICAR!!
random.seed(a=1, version=2)

#Generamos subconjuntos cuadrados del plano R2 para determinar los rangos de X e Y
xrango1 = random.sample(range(100, 1000), 200)
xrango2 = list(np.add(xrango1, random.sample(range(10, 230), 200)))
yrango1 = random.sample(range(100, 950), 200)
yrango2 = list(np.add(yrango1, random.sample(range(10, 275), 200)))
        
for j in range(len(xrango1)):
    for i in range(5):
        random.seed(a=i, version=2)
        xrandomlist = random.sample(range(xrango1[j], xrango2[j]), 4)
        yrandomlist = random.sample(range(yrango1[j], yrango2[j]), 4)
        X.append(xrandomlist[0:2])
        Y.append(yrandomlist[2:4])

'''
APARTADO 1

Diseñar primero un algoritmo, y luego programarlo en Python, para calcular el numero
de componentes conexas de la union de N segmentos aleatorios cualesquiera del plano

Para ello, seguremos estos pasos:
    
-Crear una lista vacía para almacenar los segmentos que formarán la unión.
-Leer N segmentos aleatorios del plano y añadirlos a la lista creada en el paso anterior.
-Crear un grafo vacío.
-Para cada segmento en la lista, crear un vértice en el grafo.
-Para cada par de segmentos que se intersequen, añadir una arista entre los vértices correspondientes en el grafo.
-Calcular el número de componentes conexas del grafo.
-Devolver el número de componentes conexas.
'''

def comp_con(N,X,Y):
    """
    Parameters
    ----------
    N : numero de segmentos a crear
    X : conjunto de primeras coordenadas de puntos
    Y : conjunto de segundas coordenadas de puntos

    Returns
    -------
    union_segmentos : espacio formado por union de segmentos
    """
    union_segmentos=[]
    for i in range(N):
        random_index1 = random.randint(0,len(X)-1)
        random_index2 = random.randint(0,1)
        random_index3 = random.randint(0,len(X)-1)
        random_index4 = random.randint(0,1)
        PointA1 = Point(X[random_index1][random_index2], Y[random_index1][random_index2])
        PointA2 = Point(X[random_index3][random_index4], Y[random_index3][random_index4])
        SegmentA = LineString([PointA1, PointA2])
        union_segmentos.append(SegmentA)
    return union_segmentos

'''
union_segmentos=comp_con(15,X,Y)

for segm in union_segmentos:
    plt.plot(*segm.xy, color="blue")
plt.xlim(0,1300)
plt.ylim(0,1300)
plt.title("Espacio formado por la unión de 15 segmentos aleatorios")
plt.savefig("Espacio 1")
plt.show()
'''
def crea_grafo(union_segmentos):
    """
    Funcion que genera un grafo a partir de un conjunto de segmentos, cada segmento un nodo
    y dos nodos se unen por una arista si los segmentos se intersecan.
    
    Parameters
    ----------
    union_segmentos : espacio formado por union de segmentos
    Returns
    -------
    grafo : grafo formado por los nodos y aristas
    """
    grafo=nx.Graph()
    for i in range(len(union_segmentos)):
        grafo.add_node(Point(union_segmentos[i].coords[0]))
    
    for i in range(len(union_segmentos)):
            for j in range(len(union_segmentos)):
                if i!=j:
                    if union_segmentos[i].intersects(union_segmentos[j]):
                        grafo.add_edge(Point(union_segmentos[i].coords[0]), Point(union_segmentos[j].coords[0]))
                        
    return grafo
                    
#g=crea_grafo(union_segmentos)
                        
def dibuja_grafo(grafo):
    """
    Funcion que dibuja el grafo (no respeta las distancios de los puntos, simplemente distribuye 
    los nodos y aristas para una corrrecta visualizacion)

    Parameters
    ----------
    grafo : grafo que representa la union de segmentos
    """
    nx.draw(g)
'''
dibuja_grafo(g)
plt.title("Grafo que representa el espacio de 15 segmentos")
plt.savefig("Node map")  
plt.show() 
'''
def dfs(visitados,g,nodo):
    """
    Funcion que marca un nodo como vsiitado y explora en profundidad todos sus vecinos
    
    Parameters
    ----------
    visitados : diccionario nodo:visitado, a cada nodo le asigna True si ya ha sido visitado
    por el algoritmo de busqueda en profundidad
    g : grafo de la union de segmentos
    nodo : nodo que se visita y el cual se exploran sus vecinos
    """
    visitados[nodo] = True;
    for elem in g.adj[nodo]:
        if visitados[elem]==False:
            dfs(visitados, g,elem);

def num_comp_con(grafo,union_segmentos):
    """
    Funcion que inicializa el diccionario nodo:visitado y llama a dfs para explorar cada nodo del grafo
    Con esto podemos contar las componentes conexas del grafo

    Parameters
    ----------
    grafo : grafo de la union de segmentos 
    union_segmentos : espacio formado por union de segmentos

    Returns
    -------
    count : numero de componentes conexas del grafo
    """
    visitados={}
    for i in range(len(union_segmentos)):
        visitados[Point(union_segmentos[i].coords[0])]=False
 
    count = 0;
 
    for elem in visitados.keys():
        if visitados[elem] == False :
            dfs(visitados,grafo,elem)
            count += 1
        
    return count;

#print(num_comp_con(g,union_segmentos))   

'''
APARTADO 2

Representar graficamente el espacio A y usar el apartado anterior para calcular el numero
de componentes conexas. ¿Cuantos carriles bici adicionales se requieren para conseguir conectar
correctamente todos los anterior? 
'''

#Representamos el Espacio topológico representado por los 1000 segmentos
        
for i in range(len(X)):
    plt.plot(X[i], Y[i], 'b')
plt.title("Espacio formado por la unión de segmentos A")
plt.savefig("Espacio A")
plt.show()

#Creamos el conjunto union de segmentos para el espacio total
def crea_espacio(X,Y):
    """
    Funcion que genera los segmentes del espacio A de la plantilla
    
    Parameters
    ----------
    X : conjunto de primeras coordenadas de puntos
    Y : conjunto de segundas coordenadas de puntos

    Returns
    -------
    union_segmentos : espacio formado por union de segmentos
    """
    union_segmentos=[]
    
    for i in range(len(X)):
        PointA1 = Point(X[i][0], Y[i][0])
        PointA2 = Point(X[i][1], Y[i][1])
        SegmentA = LineString([PointA1, PointA2])
        union_segmentos.append(SegmentA)
    return union_segmentos

espacio=crea_espacio(X,Y)

g=crea_grafo(espacio)

#En este caso no dibujamos el grafo ya que son tantos nodos que no se visualiza correctamente
#dibuja_grafo(g)     

num_cc=num_comp_con(g,espacio)
print("Hay un total de",num_cc,"componentes conexas")

#Trazamos una vertical para cada segmento que se encuentre solo en una componente conexa
cont_nuevos_carriles=0
g_aux=g

for elem in g.adj.keys():
    if len(g_aux.adj[elem])==0:
        PointA1 = Point(elem.x,100)
        PointA2 = Point(elem.x,1200)
        SegmentA = LineString([PointA1, PointA2])
        espacio.append(SegmentA)
        cont_nuevos_carriles+=1
        g_aux=crea_grafo(espacio)
        if len(g_aux.adj[PointA1])==1:
            PointA1 = Point(100,elem.y)
            PointA2 = Point(1200,elem.y)
            SegmentA = LineString([PointA1, PointA2])
            espacio.append(SegmentA)
            cont_nuevos_carriles+=1
            g_aux=crea_grafo(espacio)
        
        
num_cc=num_comp_con(g_aux,espacio)
print('Componentes',num_cc)        
print("Se necesitan un total de",cont_nuevos_carriles,"carriles") 
