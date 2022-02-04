import numpy as np
import pandas as pd
import time
import datetime
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de acciones')
    user_input = st.text_input('Introducir TICKER de la empresa' , 'TSLA')
    st.title('Model - LONG-SHORT TERM MEMORY')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 20, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    
    
    # Describiendo los datos
    st.subheader('Datos del 2010 al 2022') 
    st.write(df.describe())

    # Separacion de datos de entrenamiento y prueba
    set_entrenamiento = df[:2021].iloc[:,2:3]
    set_validacion = df[:2022].iloc[:,2:3]
    sc = MinMaxScaler(feature_range=(0,1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)
    time_step = 60
    X_train = []
    Y_train = []
    m = len(set_entrenamiento_escalado)
    for i in range(time_step,m):
        # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
        X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

        # Y: el siguiente dato
        Y_train.append(set_entrenamiento_escalado[i,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    # Reshape X_train para que se ajuste al modelo en Keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    dim_entrada = (X_train.shape[1],1)
    dim_salida = 1
    na = 50 #numero de neuronas
    modelo = Sequential()
    modelo.add(LSTM(units=na, input_shape=dim_entrada)) #se especifica el num de neuronas
    modelo.add(Dense(units=dim_salida))
    modelo.compile(optimizer='rmsprop', loss='mse')
    modelo.fit(X_train,Y_train,epochs=20,batch_size=32)
    x_test = set_validacion.values
    x_test = sc.transform(x_test)
    X_test = []
    for i in range(time_step,len(x_test)):
        X_test.append(x_test[i-time_step:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    prediccion = modelo.predict(X_test)
    prediccion = sc.inverse_transform(prediccion)
    st.subheader('Predicción de la acción VS valor real') 
    plt.plot(set_validacion.values[0:len(prediccion)],color='red', label='Valor real de la acción')
    plt.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    plt.legend()
    st.pyplot()
    #plt.show()
    # continuar con su codigo
