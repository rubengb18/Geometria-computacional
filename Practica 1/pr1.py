"""
Práctica 1. Código de Huffmann y Teorema de Shannon
Rubén Gómez Blanco y Adrián Sanjuán Espejo
"""
import os
import numpy as np
import pandas as pd
import math
from collections import Counter

#### Vamos al directorio de trabajo
os.getcwd()
#os.chdir("C:/PR1")
os.chdir("R:/QUINTO/GCOMP/Practica 1")
#files = os.listdir(ruta)

###Funcion que lee un fichero dada una ruta por defecto
def leer_fichero(fichero):
    with open(fichero, 'r',encoding="utf8") as file:
      txt = file.read()
      return txt
  
en=leer_fichero('GCOM2023_pract1_auxiliar_eng.txt')
es=leer_fichero('GCOM2023_pract1_auxiliar_esp.txt')

#### Después del calculo del error:
error=0.0001

#### Contamos cuantos caracteres hay en cada texto
tab_en = Counter(en)
tab_es = Counter(es)

#### Transformamos en formato array de los carácteres (states) y su frecuencia
#### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
def calcular_distr(tab):
    tab_states = np.array(list(tab))
    tab_weights = np.array(list(tab.values()))
    tab_probab = tab_weights/float(np.sum(tab_weights))
    distr = pd.DataFrame({'states': tab_states, 'probab': tab_probab, 'position': range(len(tab_states))})
    #En caso de igualdad en las probabilidades, tenemos en cuenta la posición, es decir, más prioridad el 
    #que se encuentra antes en el texto
    distr = distr.sort_values(by=['probab','position'], ascending=True)
    distr.index=np.arange(0,len(tab_states))
    distr.drop(['position'], axis=1, inplace = True)
    return distr

#### Calculo las distribuciones a partir de tab_en y tab_es
distr_en=calcular_distr(tab_en)
distr_es=calcular_distr(tab_es)

#### Obtenemos una rama del arbol fusionando los dos states con menor frecuencia y 
#### seguimos con el algoritmo ordenando el estado resultante en el conjunto de estados
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab})
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

#### Funcion que obtiene todo el arbol de Huffman rama por rama
def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
#### tree_en es el arbol para el texto en ingles
tree_en = huffman_tree(distr_en)
#### tree_es es el arbol para el texto en español
tree_es = huffman_tree(distr_es)

#EJERCICIO 1
print('----EJERCICIO 1----')
# Hallar código Huffman de los textos (almacenados en 'en' y 'es'), longitudes medias
# y comprobar Primer Teorema de Shannon

### Para que ambos diccionarios de distribuciones indexen por los caracteres
distr_es.set_index('states',inplace = True)
distr_en.set_index('states',inplace = True)
'''
codificar_estados: Funcion que genera un diccionario con pares caracter:codigo a partir
 de un arbol de Huffman.
input:
    tree: arbol de Huffman
output:
    Diccionario states donde para cada caracter del idioma se le asocie su código de Huffman
'''
### Recorrido por ramas de hojas a raiz del árbol de Huffman, y se va concatenando 0 o 1 
### en el valor de la clave del caracter
def codificar_estados(tree):
    states = dict()
    for i in range(len(tree)-1,-1,-1):
        for key in tree[i].keys():
            for state in key:
                if state not in states:
                    states[state] = ""
                states[state] += str(tree[i][key])
    return states

'''
long_media: Funcion que calcula la longitud media de una codificación de Huffman
input:
    distr: distribucion de caracteres en el texto segun el idioma
    codificaciones: diccionario donde para cada caracter del idioma se le asocie su código de Huffman
output:
    Longitud media sum_pond del código de Huffman del texto
'''

def long_media(distr, codificaciones): 
    sum_pond = 0
    for state in codificaciones:
        sum_pond += len(codificaciones[state]) *  distr.loc[state,"probab"]
    sum_pond = sum_pond / sum(distr["probab"]) #En nuestro caso será 1 al ser frecuencias relativas
    return sum_pond

#### codif_en es el diccionario caracter:codigo del texto en ingles
codif_en = codificar_estados(tree_en)
#### codif_es es el diccionario caracter:codigo del texto en español
codif_es = codificar_estados(tree_es)
#print("Diccionario con la codificación en inglés:", codif_en)
#print("Diccionario con la codificación en español:",codif_es)

#### L_en es la longitud media del codigo de Huffman del texto en ingles
L_en = long_media(distr_en, codif_en)
#### L_es es la longitud media del codigo de Huffman del texto en español
L_es = long_media(distr_es, codif_es)
print("La longitud media de S_en es", L_en, "+/-", error)
print("La longitud media de S_es es", L_es, "+/-", error)

'''Para ver que se cumple el Primer Teorema de Shannon H(C) <= L(C) < H(C) + 1'''
'''
H: Funcion que calcula la entropia del sistema
input:
    distr: distribucion de caracteres en el texto segun el idioma (diccionario caracter:distribucion)
output:
    Entropia suma del sistema
'''
def H(distr):
    suma = 0
    for prob in distr["probab"]:
        suma -= prob * math.log(prob,2)
    return suma

'''
comprobar_shannon: Funcion que comprueba el Primer Teorema de Shannon
input:
    long: longitud media del código de Huffman
    distr: distribucion de caracteres en el texto segun el idioma (diccionario caracter:distribucion)
output:
    True/False si cumple/no cumple el Primer Teorema de Shannon
'''
def comprobar_shannon(long,distr):
    h_c=H(distr)
    if h_c <=long and long < h_c + 1:
        return True
    else:
        return False

print("Entropia del inglés =>",H(distr_en), "+/-", error)
print("Entropia del español =>",H(distr_es), "+/-", error)    
    
print("¿Se cumple el Primer Teorema de Shannon (en)? (T/F) =>",comprobar_shannon(L_en,distr_en))
print("¿Se cumple el Primer Teorema de Shannon (es)? (T/F) =>", comprobar_shannon(L_es,distr_es))
    
#EJERCICIO 2
print('\n----EJERCICIO 2----')
#Codificar 'dimension' en inglés y español y comparar con codificación binaria usual
'''
codificar: Funcion que codifica un texto a codigo de Huffman para cierto idioma (determinado por la
 codificacion)
input:
    txt: texto a codificar
    codificaciones: diccionario donde para cada caracter del idioma se le asocie su código de Huffman
output:
    Cadena de caracteres traduccion con el código de Huffman de la palabra a codificar
'''
def codificar(txt, codificaciones):
    traduccion=""
    for char in txt:
        traduccion += codificaciones[char]
    return traduccion

#print('----CODIFICACIÓN DEL TEXTO COMPLETO EN INGLÉS----')
#print(codificar(en,codif_en))
#print('----CODIFICACIÓN DEL TEXTO COMPLETO EN ESPAÑOL----')
#print(codificar(es,codif_es))

palabra="dimension"
print("Codificacion en ingles de '",palabra,"' =>", codificar(palabra, codif_en))
print("Codificacion en español de '",palabra,"' =>", codificar(palabra, codif_es))

'''Comparación con codificación binaria, suponiendo que todas las letras tiene una longitud binaria de
 log en base 2 del numero total de letras'''
'''
long_binaria: Funcion que calcula la longitud de una cadena de caracteres codificados en binario usual
input:
    distr: distribucion de caracteres en el texto segun el idioma (diccionario caracter:distribucion)
    txt: texto cuya longitud hay que calcular
output:
    Longitud como producto de la longitud de cada caracter en binario usual por el numero de caracteres
    (tam_cod_car*tam_txt)
'''
def long_binaria(distr,txt):
    tam_cod_car=int(math.log(len(distr),2))
    tam_txt=len(txt)
    return tam_cod_car*tam_txt

'''
long_huffman: Funcion que calcula la longitud de una cadena de caracteres codificados en binario usual
input:
    codificaciones: diccionario donde para cada caracter del idioma se le asocie su código de Huffman
    tab: diccionario donde para cada caracter se le asocia el numero de veces que aparece en el texto
output:
    Longitud suma del codigo de Huffman
'''
def long_huffman(codificaciones, tab):
    suma=0
    for state in codificaciones:
        suma+=tab[state]*len(codificaciones[state])
    return suma

### Tamaños del codigo binario usual y de Huffman de 'dimension' en ingles
print("Tamaño del código binario usual de '",palabra,"' en inglés:",long_binaria(distr_en,palabra))
print("Tamaño del código de Huffman de '",palabra,"' en inglés:",long_huffman(codif_en,Counter(palabra)))

### Tamaños del codigo binario usual y de Huffman de 'dimension' en español
print("Tamaño del código binario usual de '",palabra,"' en español: ",long_binaria(distr_en,palabra))
print("Tamaño del código de Huffman de '",palabra,"' en español:",long_huffman(codif_en,Counter(palabra)))

### Tamaños del codigo binario usual y de Huffman del texto completo en ingles
print('Tamaño del código binario usual en inglés:',long_binaria(distr_en,en))
print('Tamaño del código de Huffman en inglés:',long_huffman(codif_en,tab_en))

### Tamaños del codigo binario usual y de Huffman del texto completo en español
print('Tamaño del código binario usual en español:',long_binaria(distr_es,es))
print('Tamaño del código de Huffman en español:',long_huffman(codif_es,tab_es))


#EJERCICIO 3
#Decodificar un código de Huffman
print('\n----EJERCICIO 3----')

'''
decodificar: Funcion que decodifca un codigo de Huffman a un idioma
input:
    codigo: codigo de Huffman a decodificar
    codificaciones: diccionario donde para cada caracter del idioma se le asocie su código de Huffman
output:
    Cadena de caracteres en un idioma concreto de la decodificacion del codigo dado
'''

def decodificar(codigo,codificaciones):
    cadena=''
    decodif=''   
    dic_inv= {valor:clave for clave, valor in codificaciones.items()} 
    #Se puede hacer porque los codigos no se repiten y valen como claves
    for num in codigo:
        cadena+=num
        if cadena in dic_inv:
             decodif += dic_inv[cadena]
             cadena = ""
    return decodif

decode="0101010001100111001101111000101111110101010001110"
#### Decodificacion de decode en ingles
print(decode, "se traduce en ingles como ", decodificar(decode,codif_en))
#### Decodificacion de decode en español
#print(decode, "se traduce en español como ", decodificar(decode,codif_es))

### ADICIONAL: Probamos a codificar y decodificar una palabra
palabra="geométrico"
cod=codificar(palabra,codif_es)
print("La codificación de",palabra,"en español es",cod)
dec=decodificar(cod,codif_es)
print("La decodificación de",cod,"en español es",dec)