# O objetivo desse código é aprender com LSTM em janelas de 60 dias de 2013 até 2017
# Depois usar a nossa rede treinada para prever o valor de fechamento em 2018

# imports importantes necessários
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
# tensorflow (o mais importante)
from tensorflow.python.layers.core import dropout


# supress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import do csv ro dataframe para depois virar lista ou outro objeto python
data = pd.read_csv("GMAT3_STK_PRICE.csv")
print(data.head())
print(data.info())
print(data.describe())

#  Visualizar valores
# Plot 1 - valor de "open" e "close"
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Open'], label="Open", color="blue")
plt.plot(data['Date'], data['Price'], label="Price", color="red")
plt.title('Open-Close Price over Time')
plt.legend()

# Plot 2 - Volume de comercialização (procura valores distintos)
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Vol'], label="Volume", color="orange")
plt.title("Stock Volume over Time")

# seleciona apenas os valores numéricos do dataset
numeric_data = data.select_dtypes(include=["int64", "float64"])

# Plot 3 - Verifica correlações entre as estruturas

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Converte a table DATE em um frame DateTime

data['date'] = pd.to_datetime(data['date'])

# crava o espaço de tempo que será usado para predição
prediction = data.loc[
    (data['date'] >= datetime(2013, 1, 1)) &
    (data['date'] <= datetime(2018, 1, 1))
    ]

plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['close'], color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")

# filtra o preço de fechamento
stock_close = data.filter(["close"])
dataset = stock_close.values  # convert array para numpy

# (IMPORTANTE) define o tamanho do array de treinamento
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Transforma os valore em valores escalaveis [0,1]
# transforma [$1100, $1110, $1009, n...] em [0,0.5,1,0,n...] onde n ~= [0,1]
scaler = StandardScaler()
scale_data = scaler.fit_transform(dataset)

# cria o array de teste
training_data = scale_data[:training_data_len]  # 95% of all out data

X_train, Y_train = [], []

# Create a janela flutuante para nossa ação (60 dias)

for i in range(60, len(training_data)):
    X_train.append(training_data[i - 60:i, 0])
    Y_train.append(training_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# estabelece o modelo como sequencial
model = keras.models.Sequential()

# 1 Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# 2 Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3 Layer
model.add(keras.layers.Dense(128, activation="relu"))

# 4 Layer
model.add(keras.layers.Dropout(0.5))

# Final Layer
model.add(keras.layers.Dense(1))

model.summary()
# IMPORTANTE estabelece parâmetros
model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

# bora para rodar com 20 épocas
training = model.fit(X_train, Y_train, epochs=20, batch_size=32)

# cria array de valores de fechamento de 2018
test_data = scale_data[training_data_len - 60:]
X_test, Y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, shape=(X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)

# transforma [0,1,2,0,n...] em [$1100, $1110, $1009, n...]
predictions = scaler.inverse_transform(predictions)

# plotting
train = data[:training_data_len]
test = data[training_data_len:]
test = test.copy()

test['Predictions'] = predictions
plt.figure(figsize=(12,8))
plt.plot(train['date'], train['close'], label='Train', color='blue')
plt.plot(test['date'], test['close'], label='Test', color='red')
plt.plot(test['date'], test['Predictions'], label='Predictions(Actual)', color='Green')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
