from doctest import DocFileSuite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
# from pandas_datareader import data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from sklearn.metrics import plot_roc_curve

from sklearn import metrics
import streamlit as st


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de acciones')
    user_input = st.text_input('Introducir TICKER de la empresa' , 'MSFT')
    st.title('Model - SUPPORT VECTOR REGRESION')
    ticker = user_input

    tiempo = st.date_input('Introduzca la fecha')
    tiempo = str(tiempo)

    anio_presente = int(tiempo[:4])
    mes_presente = int(tiempo[5:7])
    dia_presente = int(tiempo[8:10])


    anio_anterior = int(tiempo[:4])
    mes_anterior = int(tiempo[5:7])
    dia_anterior = 30

    if (dia_presente - 30) < 0:
        dia_anterior = dia_presente
        mes_anterior = mes_anterior - 1
    else:
        dia_anterior = 1
    if mes_anterior == 0:
        mes_anterior = 12
        anio_anterior = int(tiempo[:4]) - 1

    period1 = int(time.mktime(datetime.datetime(anio_anterior, mes_anterior, dia_anterior, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(anio_presente, mes_presente, dia_presente, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df_svr = pd.read_csv(query_string)
    # Seleccion de datos
    # Obtenemos la data menos la ultima fila
    data = df_svr.head(len(df_svr)-1)
    data = data.dropna()

    days = list()
    adj_close_prices = list()
    # Obtenemos la fecha y precios de cierre ajustados
    df_days = data.loc[:, 'Date']
    df_adj_close = data.loc[:, 'Adj Close']

    # Describiendo los datos
    st.subheader(f'Datos del dia {dia_presente} del mes {mes_presente} del año {anio_presente}') 
    st.write(data)
    
    st.subheader('Informacion de la data') 
    st.write(data.describe())
    
    for day in df_days:
        days.append([int(day.split('-')[2])])

    for adj_close_price in df_adj_close:
        adj_close_prices.append( float(adj_close_price) )


    # Creamos 3 modelos SVR
    # Creamos y entrenamos un modelo SVR usando un kernel lineal
    lin_svr = SVR(kernel = 'linear', C=1000)
    lin_svr.fit(days, adj_close_prices)
    # Creamos y entrenamos un modelo SVR usando un kernel polinomial
    pol_svr = SVR(kernel = 'poly', C=1000, degree = 2)
    pol_svr.fit(days, adj_close_prices)
    # Creamos y entrenamos un modelo SVR usando un kernel rbf
    rbf_svr = SVR(kernel = 'rbf', C=1000, gamma = 0.15)
    rbf_svr.fit(days, adj_close_prices)

    # Graficamos los modelos cual fue el mejor modelo
    st.subheader('SVR - Score Modelos')
    plt.figure(figsize=(16,8))
    plt.scatter(days, adj_close_prices, color='red', label='Data')
    plt.plot(days, rbf_svr.predict(days), color='green', label='Modelo RBF')
    plt.plot(days, pol_svr.predict(days), color='orange', label='Modelo Polinomial')
    plt.plot(days, lin_svr.predict(days), color='blue', label='Modelo Lineal')
    plt.legend()
    st.pyplot()


    #modelo = rbf_svr
    #X_test = days
    #y_test = adj_close_prices

    #Visualizaciones 
    #pred_modelo = modelo.predict(X_test)
    #st.subheader('Classification Report')
    #st.text(classification_report(y_test,pred_modelo))
    # st.write(st.table(classification_report(y_test,pred_modelo)))
    #st.subheader('Confusion Matrix')
    #plot_confusion_matrix(modelo,X_test,y_test)
    #st.pyplot()


    #Mostrar el precio predecido para el dato dado
    daytest = [[dia_presente]]

    st.subheader('El modelo SVR RBF predijo: ')
    st.write(rbf_svr.predict(daytest))
    st.write('Score del modelo RBF') 
    st.success(rbf_svr.score(days,adj_close_prices))



    st.subheader('El modelo SVR Lineal predijo: ')
    st.write(lin_svr.predict(daytest))
    st.write('Score del modelo Lineal') 
    st.success(lin_svr.score(days,adj_close_prices))

    st.subheader('El modelo SVR Polinomial predijo: ')
    st.write(pol_svr.predict(daytest))   
    st.write('Score del modelo Polinomial') 
    st.success(pol_svr.score(days,adj_close_prices))

    # Mostrar el precio real para el dato dado
    st.subheader('El precio real es:')
    st.success(df_svr['Adj Close'][len(df_svr)-1])