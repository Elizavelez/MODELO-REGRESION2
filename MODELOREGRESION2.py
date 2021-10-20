# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 23:18:03 2021

@author: User
"""

# 'intubed'

# , 'icu'


# Paso 1: importamos la librería numérica NumPy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import *


# CARGAR DATASET DE PACIENTES
#-----------------------------------------------------------------
datos = pd.read_excel('D:/Uni/SEMESTRE 9/TRABAJO DE GRADO 1 (todo)/BD/COVID-BD.xlsx')
sns.pairplot(datos)
datos = datos.drop(columns=['id', 'patient_type', 'entry_date', 'date_symptoms', 'pregnancy', 'contact_other_covid', 'covid_res'])

#Cambiamos la variable fecha de muerte por valores 1 (si fallecio) y 2 (No Fallecio)
Datos1=datos['Murio']

for i in range (len(Datos1)):
    if Datos1[i]=='9999-99-99':
        Datos1[i]=2
    else: 
        Datos1[i]=1

datos['Murio']=Datos1

datos = datos.drop(columns=['intubed','icu'])

#Estandarizar el Dataframe
from sklearn.preprocessing import MinMaxScaler #Permite que las caracteristicas esten en un rango (0 y 1)
scaler = MinMaxScaler()

#sns.countplot(x='Murio',data=datos)
#sns.countplot(x='Murio',hue='sex',data=datos)

# datos.isnull()

#Escalamos y tranformamos todas las columnas del Dataframe
datos_transfor = pd.DataFrame(scaler.fit_transform(datos), columns = datos.columns)

# SELECCIÓN DE LAS VARIABLES DEPENDIENTES E INDEPENDIENTES
#-----------------------------------------------------------------
# x contendra las variables de comorbilidades, años y sexo
X = datos.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13]].values
# y contendra la variable predictora es decir la mortalidad
Y = datos.iloc[:, 0].values
Y.astype('int')
# SELECCIÓN DEL TAMAÑO DEL TEST
#-----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y.astype('int'), test_size=0.30)
# TRANSFORMACIÓN DE LOS DATOS
#-----------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Paso 4: Creamos una instancia de la Regresión Logística
regresion_logistica = LogisticRegression(C=20, tol= 0.01, max_iter= 60, penalty= 'l2')

# Paso 5: Entrena la regresión logística con los datos de entrenamiento
regresion_logistica.fit(X_train, y_train)

#X_nuevo = np.array([2,2,1,47,1,2,2,2,2,2,2,2,2,2,1]).reshape(15)
y_pred = regresion_logistica.predict(X_test)
prediccion = regresion_logistica.predict([[2,1,47,1,2,2,2,2,2,2,2,2,2]])
print(prediccion)

if prediccion == 1:
    
    print ("El paciente tiene riesgo de morir")
    
else: 
    print("El paciente está fuera de riesgo")
    


# # print(confusion_matrix(y_test, y_pred))
# # print(classification_report(y_test, y_pred))
# print ("Resultado:" )
# print (resultado)
