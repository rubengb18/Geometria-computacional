# -*- coding: utf-8 -*-
"""
Practica Opcional 2-Redes neuronales
Rubén Gómez Blanco
"""

import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Datos de entrada y salidas esperadas
X = np.array([[1, 1], [1, 0], [0, 1]])
y = np.array([1, 1, 0])

# Inicialización de los pesos sinápticos con valores aleatorios pequeños
d=100

w1 = random.uniform(-1/d,1/d)
w2 = random.uniform(-1/d,1/d)

e=0.5
# Función de activación sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Definición de la derivada de la función sigmoide
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Función de clasificación
def classify(x, y):
    z = w1 * x + w2 * y
    y_hat = sigmoid(z)
    return 1 if y_hat > 0.55 else 0

epochs=1
# Realizar una iteración para cada patrón de entrada
for r in range(epochs):
    for i in range(len(X)):
        x1 = X[i][0]
        x2 = X[i][1]
        t = y[i]
        
        z = w1 * x1 + w2 * x2
    
        # Clasificar el patrón de entrada
        y_pred = classify(x1,x2)
        
        # Actualizar los pesos si la clasificación es incorrecta
        if y_pred != t:
            delta = (t - y_pred) * sigmoid_prime(z) * e
            w1 += delta * x1
            w2 += delta * x2
                
# Comprobar el resultado con la librería Keras
# Crear un modelo de perceptrón simple con Keras
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid', use_bias=True, 
                kernel_initializer='random_normal', bias_initializer='random_normal'))

# Compilar el modelo
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Ajustar el modelo con los datos de entrada y salidas esperadas
model.fit(X, y, epochs=500, verbose=0)

# Obtener las predicciones del modelo
y_pred_keras = model.predict(X)
lista=[]
for obj in y_pred_keras:
    if obj >=0.5:
        lista.append(1)
    else:
        lista.append(0)
        
# Imprimir las clasificaciones obtenidas por ambos métodos
print("Clasificación obtenida manualmente: ", [classify(X[i][0], X[i][1]) for i in range(len(X))])
print("Clasificación obtenida con Keras: ", lista)
'''
#Para ver cuantas veces acierta y cuantas falla
lista_en_binario=[0,0,0,0,0,0,0,0]
for i in range(10000):
    w1 = random.uniform(-1/d,1/d)
    w2 = random.uniform(-1/d,1/d)
    for r in range(epochs):
        for i in range(len(X)):
            x1 = X[i][0]
            x2 = X[i][1]
            t = y[i]
            
            z = w1 * x1 + w2 * x2
        
            # Clasificar el patrón de entrada
            y_pred = classify(x1,x2)
            
            # Actualizar los pesos si la clasificación es incorrecta
            if y_pred != t:
                delta = (t - y_pred) * sigmoid_prime(z) * e
                w1 += delta * x1
                w2 += delta * x2
    resultado=[classify(X[i][0], X[i][1]) for i in range(len(X))]
    lista_en_binario[resultado[0]*4+resultado[1]*2+resultado[2]*1]+=1
'''
#Comrpobar si clasifica bien otros ejemplos
prueba=np.array([[2, 1], [0, 0], [0, 2], [0, -1], [2, 2], [0, 1/2], [1/2, 1/2], [1/3, 1/2]])
print("Clasificación obtenida manualmente: ", [classify(prueba[i][0],prueba[i][1]) for i in range(len(prueba))])
