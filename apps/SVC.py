import numpy as np
import pandas as pd
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,plot_confusion_matrix,plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import streamlit as st

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de acciones')
    user_input = st.text_input('Introducir TICKER de la empresa' , 'MSFT')
    st.title('Model - SUPPORT VECTOR CLASSIFIER')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 10, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    data = pd.read_csv(query_string)
    
    # Seleccion de datos
    data['highest hight'] = data['High'].rolling(window=10).max()
    data['lowest low'] = data['Low'].rolling(window=10).min()
    data['trigger'] = np.where(data['High']==data['highest hight'],1,np.nan)
    data['trigger'] = np.where(data['Low']==data['lowest low'],0,data['trigger'])
    data['position'] = data['trigger'].ffill()
    data = data.drop(data.index[[0,1,2,3,4,5,6,7,8,9]])

    # Eleccion de datos
    df = data.drop(['Date','Adj Close','Volume','highest hight', 'lowest low', 'trigger'],axis=1)
    df = df.dropna()
    
    # Describiendo los datos
    st.subheader('Datos del 2010 al 2022') 
    st.write(df.describe())

    # Separacion de datos de entrenamiento y prueba
    X = df.drop('position',axis=1)
    y = df['position']
    X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=101)

    # Uso del modelo
    modelo = SVC()
    modelo.fit(X_train,y_train)
    st.subheader('Score del modelo') 
    st.success(modelo.score(X_test,y_test))

    #Visualizaciones 
    pred_modelo = modelo.predict(X_test)
    st.subheader('Classification Report')
    st.text(classification_report(y_test,pred_modelo))
    # st.write(st.table(classification_report(y_test,pred_modelo)))
    st.subheader('Confusion Matrix')
    plot_confusion_matrix(modelo,X_test,y_test)
    st.pyplot()

    st.subheader('ROC')
    plot_roc_curve(modelo, X_test, y_test, alpha = 0.8)
    st.pyplot()
    
    st.subheader('RECALL CURVE')
    plot_precision_recall_curve(modelo, X_test, y_test, alpha = 0.8)
    st.pyplot()
    
    # Prediccion del modelo
    st.subheader('Ingrese los datos para iniciar la prediccion')
    # Lecctura de datos
    open = st.text_input("Open:")
    high = st.text_input("High:")
    low = st.text_input("Low:")
    close = st.text_input("Close:")
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción"):
        predictS = modelo.predict([[np.float_(open.title()),np.float_(high.title()),np.float_(low.title()),np.float_(close.title())]])
        if predictS[0] == 1:
            st.success('SEÑAL DE COMPRA')
        else:
            st.success('SEÑAL DE VENTA')
