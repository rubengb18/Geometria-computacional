# -*- coding: utf-8 -*-
"""
Practica 3 - Espacio Fasico
Ruben Gomez y Adrian Sanjuan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from apng import APNG
from numpy import trapz


#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

os.getcwd()

#Calcula la derivada como limite como cociente de diferencias
def deriv(q,dq0,d):
    '''
    Parameters
    ----------
    q : variable de posición (vector)
    dq0 : valor inicial de la derivada
    d : granularidad del parametro temporal

    Returns
    -------
    dq :  vector de derivadas

    '''
    #dq = np.empty([len(q)])
    dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
    dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
    return dq

#Ecuaciones de Hamilton-Jacobi para un oscilador no lineal
a=3
b=1/2
def F(q):
    '''    
    Parameters
    ----------
    q : variable de posición (vector)
    
    Returns
    -------
    vector de derivadas segundas

    '''
    return -(8/a)*q*(q**2-b)

#Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
def orb(n,q0,dq0,F, args=None, d=0.001):
    '''
    Parameters
    ----------
    n : numero de puntos de la orbita
    q0 : posición inicial
    dq0 : valor inicial de la derivada
    F : función del sistema
    args : The default is None.
    d : granularidad del parametro temporal

    Returns
    -------
    q : vector de posiciones calculado

    '''
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q #np.array(q),

def periodos(q,d,max=True):
    #Si max = True, tomamos las ondas a partir de los máximos/picos
    #Si max == False, tomamos los las ondas a partir de los mínimos/valles
    epsilon = 5*d
    dq = deriv(q,dq0=None,d=d) #La primera derivada es irrelevante
    if max == True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q >0))
    if max != True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q <0))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0]>1]
    pers = diff_waves[diff_waves>1]*d
    return pers, waves

#################################################################    
#  CÁLCULO DE ÓRBITAS
#################################################################

#Ejemplo gráfico del oscilador no lineal
q0 = 0.
dq0 = 1.
fig, ax = plt.subplots(figsize=(12,5))
plt.ylim(-1.5, 1.5)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("q(t)", fontsize=12)
iseq = np.array([3,3.25,3.50,3.75,4])
horiz = 32
for i in iseq:
    d = 10**(-i)
    n = int(horiz/d)
    t = np.arange(n+1)*d
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    plt.plot(t, q, 'ro', markersize=0.5/i,label='$\delta$ ='+str(np.around(d,3)),
    c=plt.get_cmap("winter")(i/np.max(iseq)))
    ax.legend(loc=3, frameon=False, fontsize=12)
    ax.set_title("Evolución de q(t) para varios deltas")
plt.savefig('time_granularity.png', dpi=250)

##Nos quedamos con d = 10**-4. Calculamos la orbita y la coordenada canónica 'p'
q0 = 0.
dq0 = 1.
horiz = 32
d = 10**(-4)
n = int(horiz/d)
t = np.arange(n+1)*d
q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
dq = deriv(q,dq0=dq0,d=d)
p = dq/2

#Ejemplo gráfico de la derivada de q(t)
def graf_dq(dq,t):
    '''
    Parameters
    ----------
    dq : vector de derivadas
    t : vector de tiempos
    '''
    fig, ax = plt.subplots(figsize=(12,5))
    plt.ylim(-1.5, 1.5)  
    plt.rcParams["legend.markerscale"] = 6
    ax.set_xlabel("t = n $\delta$", fontsize=12)
    ax.set_ylabel("dq(t)", fontsize=12)
    ax.set_title("Evolución de dq(t) para varios deltas")
    plt.plot(t, dq, '-')
    plt.savefig('dqt.png', dpi=250)
    plt.show()
    
graf_dq(dq,t)


#Ejemplo de diagrama de fases (q, p) para una órbita completa

def diag_fases_orb(q,p):
    '''
    Parameters
    ----------
    q : vector de valores q (1era coordenada)
    p : vector de valores p (2da coordenada)
    '''
    fig, ax = plt.subplots(figsize=(10,5))
    plt.xlim(-2,2)  
    plt.ylim(-1, 1) 
    plt.rcParams["legend.markerscale"] = 6
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    ax.set_title("Diagrama de fases para una órbita completa")
    plt.plot(q, p, '-')
    plt.show()

diag_fases_orb(q,p)
plt.savefig('orbita_completa.png', dpi=250)

#Ejemplo de diagrama de fases (q, p) para un tiempo determinado

#Funcion que calcula el diagrama de fases para un tiempo determinado y lo muestra
def diag_fases_t(t, num_puntos,d,mostrar=False):
    '''
    Parameters
    ----------
    t : tiempo
    num_puntos : numero de puntos (q,p) en la muestra

    Returns
    -------
    q2 : primera coordenada de la órbita calcualda
    p2 : segunda coordenada de la órbita calculada
    '''
    horiz = t
    if mostrar==True:
        ax = fig.add_subplot(1,1, 1)
        ax.set_xlabel("q(t)", fontsize=12)
        ax.set_ylabel("p(t)", fontsize=12)
    seq_q0 = np.linspace(0.,1.,num=num_puntos)
    seq_dq0 = np.linspace(0.,2,num=num_puntos)
    q2 = np.array([])
    p2 = np.array([])
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            n = int(horiz/d)
            t = np.arange(n+1)*d
            q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
            dq = deriv(q,dq0=dq0,d=d)
            p = dq/2
            q2 = np.append(q2,q[-1])
            p2 = np.append(p2,p[-1])
            if mostrar==True:
                plt.xlim(-2.2, 2.2)
                plt.ylim(-1.2, 1.2)
                plt.rcParams["legend.markerscale"] = 6
                plt.plot(q[-1], p[-1], marker="o", markersize= 1, markeredgecolor="red",markerfacecolor="red")
    if mostrar==True:
        plt.xlabel("q(t)", fontsize=12)
        plt.ylabel("p(t)", fontsize=12)
        plt.title("Diagrama de fases para t = 1/4",fontsize = 14)
        plt.savefig('diagrama_fases_un_cuarto.png', dpi=250)
        plt.show()
    return q2,p2

#Diagrama de fases para t=1/4, 20 puntos y d=10**-4
q2,p2=diag_fases_t(0.25, 20, d,mostrar=True)

# Función que dibuja orbita del espacio fásico
def simplectica(q0,dq0,F,col=0,d = 10**(-4),n = int(16/d),marker='-'): 
    '''
    Parameters
    ----------
    q0 : posicion inicial
    dq0 : valor inicial de la derivada
    F : funcion del sistema
    col : color. El por defecto es 0.
    d : granularidad del parametro temporal. El por defecto es 10**(-4).
    n : numero de puntos de la orbita. El por defecto es int(16/d).
    marker : forma de representación de la órbita
    '''
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker,c=plt.get_cmap("winter")(col), markersize = 4)
    
Horiz = 20

#EJERCICIO 1
print('\n----EJERCICIO 1----\n')
#Representar el espacio fásico de al menos 10 órbitas a partir de las condiciones iniciales D0

#Función para dibujar el espacio fásico
def esp_fasico(num_puntos):
    '''
    Parameters
    ----------
    num_puntos : número de órbitas
    '''
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    ax = fig.add_subplot(1,1, 1)
    #Condiciones iniciales:
    seq_q0 = np.linspace(0.,1.,num=num_puntos)
    seq_dq0 = np.linspace(0.,2,num=num_puntos)
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
            #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
            simplectica(q0=q0,dq0=dq0,F=F,col=col,marker='ro',d= 10**(-4),n = int(Horiz/d))
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    ax.set_title("Espacio fásico")
    fig.savefig('esp_fasico.png', dpi=250)
    plt.show()

#esp_fasico(15)

#EJERCICIO 2
print('\n----EJERCICIO 2----\n')
#Obtener el área de D(1/4) y una estimación del su intervalo de error.
#Verficar el Tª Liouville entre D0 y D(1/4), y D0 y D(0,inf)

#Funcion que calcula el area de la envolvente conexa de un diagrama de fases
def area(q2,p2,mostrar=True):
    '''
    Parametros:
    ---------------
    q2 : primera coordenada de los puntos del diagrama de fases
    p2 : segunda coordenada de los puntos del diagrama de fases
    
    Return:
    hull.volume:area de la envolvente conexa del diagrama de fases
    '''
    
    X = np.array([q2,p2]).T
    hull = ConvexHull(X)
    if mostrar==True:
        convex_hull_plot_2d(hull)
    return hull.volume
    
#Funcion que calcula el area de la envolvente conexa de un diagrama de fases y le resta los sobrantes
def area_prec(q2,p2,mostrar=True):
    '''
    Parameters
    ----------
    q2 : primera coordenada de los puntos del diagrama de fases
    p2 : segunda coordenada de los puntos del diagrama de fases
    mostrar : si True muestra gráfica
    Return:
    ----------
    area 
    '''
    #Caclulo del área del diagrama de fases para t = 1/4 mediante envolventes convexas
    vol_env=area(q2,p2,mostrar)
    if mostrar==True:
        plt.rcParams["legend.markerscale"] = 6
        plt.xlabel("q(t)", fontsize=12)
        plt.ylabel("p(t)", fontsize=12)
        plt.title("Envolvente convexa del diagrama de fases para t = 1/4", fontsize = 14)
        plt.savefig('envolvente_convexa.png', dpi=250)
        plt.show()

    #Calculo del área "sobrante" en la parte derecha al hacer la envolvente conexa
    vol1=area(q2[380:400],p2[380:400],mostrar)
    if mostrar==True:
        plt.xlabel("q(t)", fontsize=12)
        plt.ylabel("p(t)", fontsize=12)
        plt.title("Envolvente convexa del borde derecho",fontsize = 14)
        plt.savefig('envolvente_convexa_dcha.png', dpi=250)
        plt.show()


    #Calculo del área "sobrante" en la parte inferior al hacer la envolvente conexa
    q_aux=[q2[i] for i in range(0,400,20)]
    p_aux=[p2[i] for i in range(0,400,20)]
    vol2=area(q_aux,p_aux,mostrar)
    if mostrar==True:
        plt.xlabel("q(t)", fontsize=12)
        plt.ylabel("p(t)", fontsize=12)
        plt.title("Envolvente convexa del borde inferior", fontsize = 14)
        plt.savefig('envolvente_convexa_inf.png', dpi=250)
        plt.show()
        
    return vol_env-vol1-vol2
 

# Veamos si se cumple el Teorema de Liouville entre D0 y D(1/4)
def teoremaLiouville(t,A):
    '''
    Parameters
    ----------
    t : tiempo del diagrama de fases en el que se calcula el area
    A : area con la que se compara
    '''
    deltas = np.linspace(3, 4, num=5)
    areas=[]
    for d in deltas:
        q_aux,p_aux=diag_fases_t(t, 20, 10**((-1)*d),mostrar=False)
        areas.append(area_prec(q_aux,p_aux,mostrar=False))
    resta_areas = [abs(areas[i]-areas[4]) for i in range(len(areas)-1)]
    error = max(resta_areas)
    print("Área para D({tiempo}) y d = 10^-4: {area}".format(tiempo = t, area = areas[4]))
    print("Se calcula un error en el cálculo del área de", error)
    if abs(areas[4]-A) < error:
        print("Se cumple el Teorema de Liouville para D_{a}".format(a = t))
    else:
        print("NO se cumple el Teorema de Liouville para D_{a}".format(a = t))

teoremaLiouville(0.25,1)

fig, ax = plt.subplots(figsize=(5,5)) 

#Funcion que calcula el area encerrada entre un conjunto de puntos con la Regla del Trapezoide
def area_2(q0,dq0,d,n, F,mostrar):
    '''
    Parameters
    ----------
    (q0,dq0) : condiciones iniciales, posicion y derivada.
    d : granularidad
    n : numero de puntos de la orbita
    F : función del sistema
    mostrar : si True se muestra gráfica

    Returns
    -------
    areaT :  area encerrada entre los puntos
    '''
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    if mostrar==True:
        plt.rcParams["legend.markerscale"] = 6
        ax.set_xlabel("q(t)", fontsize=12)
        ax.set_ylabel("p(t)", fontsize=12)
        plt.plot(q, p, '-')
        plt.axis('equal')
        plt.xlim(-2.2, 2.2)
        plt.ylim(-1.2, 1.2)
    # Regla del trapezoide
    areaT = trapz(p,q)
    return areaT

#Funcion que calcula el area para dos granularidades (extremas) y el error asociado al calculo
def area_enc_orb(horiz,q0,dq0):
    '''
    Parameters
    ----------
    horiz : medida de tiempo en el que recorre las órbitas
    (q0,dq0) : condiciones iniciales

    Returns
    -------
    a1 : área del espacio de fases con granularidad más fina
    error_1 : error asociado al calculo

    '''
    d = 10**(-4)
    n = int(horiz/d)
    a1 = area_2(q0,dq0,d,n, F,True)
    d = 10**(-3)
    n = int(horiz/d)
    a1_e = area_2(q0,dq0,d,n, F,False)
    print('Área encerrada por órbita con condiciones iniciales q = 0 y dq =',dq0,':', a1)
    error_1 = abs(a1-a1_e)
    print('Error:',error_1)
    return a1,error_1

a1,error_1=area_enc_orb(35,0.,10**(-10))
a2,error_2=area_enc_orb(3.625,0.,2)

plt.title('Area encerrada entre órbita máxima y mínima')
plt.savefig('areas.png')
plt.show()

print('Área estimada espacio fásico:', a2-a1/2)
print('Error:', error_1/2+error_2)

#EJERCICIO 3
print('\n----EJERCICIO 3----\n')
#Realiza una animación GIF con la evolución del diagrama de fases Dt para t en (0,5)

def animate(t):
    seq_q0 = np.linspace(0.,1.,num=50)
    seq_dq0 = np.linspace(0.,2,num=50)
    q2 = np.array([])
    p2 = np.array([])
    fig, ax = plt.subplots(figsize=(5,5))
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            if (i==0 or i==len(seq_q0)-1 or j==0 or j==len(seq_dq0)-1):
                q0 = seq_q0[i]
                dq0 = seq_dq0[j]
                d = 10**(-4)
                n = int(t/d)
                #t = np.arange(n+1)*d
                q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
                dq = deriv(q,dq0=dq0,d=d)
                p = dq/2
                q2 = np.append(q2,q[-1])
                p2 = np.append(p2,p[-1])
                ax.plot(q[-1], p[-1], marker="o", markersize= 1, markeredgecolor="red",markerfacecolor="red")
    plt.xlim(-2.2, 2.2)
    plt.ylim(-1.2, 1.2)
    plt.savefig('{name}.jpg'.format(name=t))
    return ax,

def init():
    return animate(0),

fig = plt.figure(figsize=(6,6))

#El siguiente fragmento de codigo realiza un GIF a partir de todas las imagenes de 
#los diagramas de fases para distintos tiempos
for t in np.arange(0.0001, 5,0.2):
    animate(t)
    
APNG.from_files(['{name}.jpg'.format(name=t) for t in np.arange(0.0001, 5,0.2)],
delay=100).save('ejemplop3_bordes.gif')