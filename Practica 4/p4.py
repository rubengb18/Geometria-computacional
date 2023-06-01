# -*- coding: utf-8 -*-
"""
Practica 4-Transformaciones isometricas afines
Rubén Gómez Blanco y Adrián Sanjuan Espejo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color
import math
from scipy.spatial import ConvexHull
from apng import APNG
from scipy.spatial.distance import cdist
os.getcwd()
#os.chdir()


########################################################################
#APARTADO 1
########################################################################

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
ax.clabel(cset, fontsize=9, inline=1)
plt.title('Figura original en 3D')
plt.savefig('figura1')
plt.show()


'''
Funcion que calcula el centroide de una figura en 3D
'''
def centroide3D(xt,yt,zt):
    '''
    Parameters
    ----------
    xt, yt, zt: coordenadas de los puntos del sistema
    
    Return
    ----------
    coordenadas del centroide
    '''
    x=0
    y=0
    z=0
    for i in range(len(xt)):
        x+=xt[i]
        y+=yt[i]
        z+=zt[i]
    return x/len(xt),y/len(xt),z/len(xt)

'''
Funcion que calcula el diametro de una figura en 2D
'''
def diam3D(x0,y0,z0):
    '''
    Parameters
    ----------
    x0, y0,z0: coordenadas de los puntos del sistema
    '''
    #Calculamos los puntos de la envolvente convexa O(N log N)
    points = np.array([x0,y0,z0]).transpose()
    hull = ConvexHull(points)
    
    #Sacamos los puntos de la envolvente
    hullpoints = points[hull.vertices,:]
    
    #En caso de querer dibujar la envolvente convexa
    '''
    ax = plt.axes(xlim=(-50,50), ylim=(-50,50), projection='3d')
    ax.set_zlim(-80,80)
    cset = ax.scatter3D(hullpoints[:,0], hullpoints[:,1], hullpoints[:,2], color='r',edgecolors='w',zorder=3)
    ax.clabel(cset, fontsize=9, inline=1)

    cset = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap = plt.cm.get_cmap('viridis'), edgecolors='w')
    ax.clabel(cset, fontsize=9, inline=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    '''
    
    # El diametro lo calculamos como la mayor distancia entre los puntos de la envolvente
    
    #hdist es una matriz simetrica (diagonal 0) de distancias de todos los puntos entre ellos
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    #bestpair son los indices de los puntos con mayor distancia entre sí
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    
    #Obtenemos los puntos a mayor distancia
    p1 = hullpoints[bestpair[0]]
    p2 = hullpoints[bestpair[1]]
    
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

'''
Funcion que calcula la transformacion isometrica asociada de un sistema de coordenadas X,Y,Z respecto a un centroide
'''
def transf_isom_afin(x,y,z,M, v,c):
    '''
    Parameters
    ----------
    X,Y,Z: coordenadas de los puntos del sistema a transformar
    M : matriz de rotacion
    v : vector de desplazamiento
    c : centroide de la figura
    '''
    
    xt = x*0
    yt = x*0
    zt = x*0
    for i in range(len(x)):
        if len(c)>2:
            q = np.array([x[i]-c[0],y[i]-c[1],z[i]-c[2]])
        else:
            q = np.array([x[i]-c[0],y[i]-c[1],z[i]])
        xt[i], yt[i], zt[i] = np.matmul(M, q)
        xt[i] += v[0]+c[0]
        yt[i] += v[1]+c[1]
        zt[i] += v[2]
        if len(c)>2:
            zt[i]+=c[2]
    return xt, yt, zt

'''
Funcion que calcula la tranformacion isometrica afin asociada a un tiempo t y la dibuja con tamaños
asociados al problema del sistema en 3D de la plantilla
'''
def animate(t,X,Y,Z,M,v,c):
    """
    Parameters
    ----------
    t : tiempo
    X,Y,Z: coordenadas de los puntos del sistema a transformar
    M : matriz de rotacion
    v : vector de desplazamiento
    c : centroide de la figura
    """
    ax = plt.axes(xlim=(-80,80), ylim=(-80,80),zlim=(-100,300), projection='3d')
    #ax.view_init(60, 30)
    M = np.array([[np.cos(t*theta),-np.sin(t*theta),0],[np.sin(t*theta),np.cos(t*theta),0],[0,0,1]])
    v=v*t
    X,Y,Z = transf_isom_afin(X,Y,Z, M=M, v=v, c=c)
    ax = plt.axes(projection='3d')
    
    X = X.reshape(120,120)
    Y = Y.reshape(120,120)
    Z = Z.reshape(120,120)
    
    cset = ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
    ax.clabel(cset, fontsize=9, inline=1)
    ax.set_xlim(-80,80)
    ax.set_ylim(-80,80)
    ax.set_zlim(-100,300)
    plt.savefig('{name}.jpg'.format(name=t))
    plt.show()
    return ax,

X, Y, Z = axes3d.get_test_data(0.05)

#Trabsformamos X, Y y Z en vectores unidimensionales para manejarlos mejor, luego se deshace el cambio
X = X.flatten()
Y= Y.flatten()
Z = Z.flatten()

d=diam3D(X,Y,Z)
v=np.array([0,0,d])
c1=centroide3D(X, Y, Z)

theta=math.pi * 3
M=[[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]]

#Codigo para ver el resultado final
    
'''
X, Y, Z = transf_isom_afin(X,Y,Z,M, v,c1)
fig = plt.figure()
ax = plt.axes(projection='3d')
    
X = X.reshape(120,120)
Y = Y.reshape(120,120)
Z = Z.reshape(120,120)

cset = ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
ax.clabel(cset, fontsize=9, inline=1)
ax.set_xlim(-60,10)
ax.set_ylim(-20,30)
ax.set_zlim(0,200)
plt.title('Figura transformada en 3D')
plt.savefig('figura1_t')
plt.show()
'''

print('Diámetro:',d)
print('El centroide se sitúa en las coordenadas',c1)

#Codigo para generar el GIF
'''
for t in np.arange(0,1,0.01):
    animate(t,X, Y, Z,M,v,c1)
    
APNG.from_files(['{name}.jpg'.format(name=t) for t in np.arange(0,1,0.01)],
delay=100).save('ejer1p4.gif')
'''


########################################################################
#APARTADO 2
########################################################################

img = io.imread('arbol.png')
xyz = img.shape

x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,1]
zz = np.asarray(z).reshape(-1)

'''
Funcion que calcula el centroide de una figura en 2D
'''
def centroide2D(xt,yt):
    '''
    Parameters
    ----------
    xt, yt: coordenadas de los puntos del sistema
    
    Return
    ----------
    coordenadas del centroide
    '''
    x=0
    y=0
    for i in range(len(xt)):
        x+=xt[i]
        y+=yt[i]
    return x/len(xt),y/len(xt)
        

'''
Funcion que calcula el diametro de una figura en 2D
'''
def diam2D(x0,y0):
    '''
    Parameters
    ----------
    x0, y0: coordenadas de los puntos del sistema
    '''
    #Calculamos los puntos de la envolvente convexa
    points = np.array([x0,y0]).transpose()
    hull = ConvexHull(points)
    
    #Sacamos los puntos de la envolvente
    hullpoints = points[hull.vertices,:]
    #En caso de querer dibujar la envolvente convexa
    '''
    fig = plt.figure()
    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')
    ax.set_zlim(0,50)
    cset = ax.scatter(hullpoints[:,0], hullpoints[:,1], color='r')
    ax.clabel(cset, fontsize=9, inline=1)
    cset = ax.scatter(x0,y0,c=col,s=0.1,animated=True)
    ax.clabel(cset, fontsize=9, inline=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    '''
    
    # El diametro lo calculamos como la mayor distancia entre los puntos de la envolvente
    
    #hdist es una matriz simetrica (diagonal 0) de distancias de todos los puntos entre ellos
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    #bestpair son los indices de los puntos con mayor distancia entre sí
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    
    #Obtenemos los puntos a mayor distancia
    p1 = hullpoints[bestpair[0]]
    p2 = hullpoints[bestpair[1]]
    
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

#Consideraremos sólo los elementos con zz < 240 
#Variables de estado coordenadas
x0 = xx[zz<240]
y0 = yy[zz<240]
z0 = zz[zz<240]/256.
#Variable de estado: color
col = plt.get_cmap("viridis")(np.array(0.1+z0))

d=diam2D(x0,y0)
v=np.array([d,d,0])
c=centroide2D(x0,y0)
print('-----------------')
print('Diámetro:',d)
print('El centroide se sitúa en las coordenadas',c)

'''
Funcion que calcula la tranformacion isometrica afin asociada a un tiempo t y la dibuja con tamaños
 asociados al problema del arbol
'''
def animate_arbol(t,X,Y,Z,M,v,c):
    """
    Parameters
    ----------
    t : tiempo
    X,Y,Z: coordenadas de los puntos del sistema a transformar
    M : matriz de rotacion
    v : vector de desplazamiento
    c : centroide de la figura
    """
    ax = plt.axes(xlim=(0,700), ylim=(0,700),zlim=(0,1), projection='3d')
    M = np.array([[np.cos(t*theta),-np.sin(t*theta),0],[np.sin(t*theta),np.cos(t*theta),0],[0,0,1]])
    v=v*t
    X,Y,Z = transf_isom_afin(X,Y,Z, M=M, v=v,c=c)
    ax.scatter(X,Y,Z,c=col,s=0.1,animated=True)
    plt.savefig('2_{name}.jpg'.format(name=t))
    return ax,
    
#Codigo para generar el GIF
'''
for t in np.arange(0,1,0.05):
    animate_arbol(t,x0, y0, z0,M,v,c)
    
APNG.from_files(['2_{name}.jpg'.format(name=t) for t in np.arange(0,1,0.05)],
delay=100).save('ejer2p4.gif')
'''

#Codigo para ver la figura inicial
'''
ax = plt.axes(xlim=(0,350), ylim=(25,375),zlim=(0,1), projection='3d')
cset = ax.scatter(x0,y0,c=col,s=0.1,animated=True)
ax.clabel(cset, fontsize=9, inline=1)
plt.title('Figura original del árbol')
plt.savefig('figura2')
plt.show()
'''
#Codigo para ver el resultado final
'''
ax = plt.axes(xlim=(400,700), ylim=(350,700),zlim=(0,1), projection='3d')
x0,y0,z0 = transf_isom_afin(x0,y0,z0, M=M, v=v,c=c)
ax.scatter(x0,y0,z0,c=col,s=0.1,animated=True)
plt.title('Figura transformada del árbol')
plt.savefig('figura2_t')
plt.show()
'''

