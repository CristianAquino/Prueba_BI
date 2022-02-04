import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import time
import datetime
import streamlit as st

# from fastai.tabular.core import add_datepart
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import neighbors
# from sklearn.model_selection import GridSearchCV

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de acciones')
    user_input = st.text_input('Introducir TICKER de la empresa' , 'MSFT')
    st.title('Model - KNN')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 10, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    
    df = df.dropna()
    df = df[['Open', 'High', 'Low', 'Close']]
    st.subheader('Datos del 2021 de Diciembre') 
    st.write(df.head())
    
    st.subheader('Detalles') 
    df.reset_index(inplace=True)
    st.write(df.describe())
    
    # Predictor variables
    st.subheader('Apertura-Cierre | Altos-Bajos') 
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
    df =df.dropna()
    X= df[['Open-Close', 'High-Low']]
    st.write(X.head())
    
    # Correlación de Pearson
    st.subheader('Coeficiente de correlación de Pearson') 
    corr = df.corr(method='pearson')
    st.write(corr)
    
    sb.heatmap(
        corr,
        xticklabels=corr.columns, 
        yticklabels=corr.columns,
        cmap='RdBu_r', 
        annot=True, 
        linewidth=0.5,
    )
    st.pyplot()
    
    # TRADING
    
    # Target variable
    Y= np.where(df['Close'].shift(-1) > df['Close'],1,-1)
    
    # Splitting the dataset
    split_percentage = 0.7
    split = int(split_percentage*len(df))

    X_train = X[:split]
    Y_train = Y[:split]

    X_test = X[split:]
    Y_test = Y[split:]
    
    # Instantiate KNN learning model(k=15)
    knn = KNeighborsClassifier(n_neighbors=15)

    # fit the model
    knn.fit(X_train, Y_train)

    # Accuracy Score
    accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
    accuracy_test = accuracy_score(Y_test, knn.predict(X_test))
    
    st.success('Precisión de datos de entrenamiento: %.2f' %accuracy_train)
    st.success('Precisión de datos de prueba: %.2f' %accuracy_test)

    # Predicted Signal
    df['Predicted_Signal'] = knn.predict(X)

    # SPY Cumulative Returns
    df['SPY_returns'] = np.log(df['Close']/df['Close'].shift(1))
    Cumulative_SPY_returns = df[split:]['SPY_returns'].cumsum() * 100

    # Cumulative Strategy Returns 
    df['Startegy_returns'] = df['SPY_returns']* df['Predicted_Signal'].shift(1)
    Cumulative_Strategy_returns = df[split:]['Startegy_returns'].cumsum() * 100

    # Plot the results to visualize the performance
    st.subheader('Rendimiento')
    plt.figure(figsize=(10,5))
    plt.plot(Cumulative_SPY_returns, color='r',label = 'Retorno de acciones por año')
    plt.plot(Cumulative_Strategy_returns, color='g', label = 'Retorno de estrategia')
    plt.legend()
    st.pyplot()
    
    Std = Cumulative_Strategy_returns.std()
    Sharpe = (Cumulative_Strategy_returns-Cumulative_SPY_returns)/Std
    Sharpe = Sharpe.mean()
    st.success('Porcentaje de retorno: %.2f'%Sharpe)
    