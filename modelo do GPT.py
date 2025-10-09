# ============================================================
# Objetivo:
# Aprender com LSTM em janelas de 60 dias e prever o valor de fechamento (Price)
# a partir do histórico de ações do Grupo Mateus.
# ============================================================

# Imports principais
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Suprimir warnings do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# 1. Carregamento e inspeção dos dados
# ============================================================

data = pd.read_csv("Grupo Mateus Stock Price History.csv")

print(data.head())
print(data.info())

# ============================================================
# 2. Limpeza e conversão das colunas
# ============================================================

# Renomeia colunas para letras minúsculas mais simples
data.columns = ['date', 'close', 'open', 'high', 'low', 'volume', 'change']

# Converte a coluna de data
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# Converte "Vol." de texto (ex: '4.30M', '2.5K') para número
def convert_volume(vol):
    if isinstance(vol, str):
        vol = vol.strip()
        if vol.endswith('M'):
            return float(vol[:-1]) * 1_000_000
        elif vol.endswith('K'):
            return float(vol[:-1]) * 1_000
        else:
            try:
                return float(vol)
            except:
                return np.nan
    return vol

data['volume'] = data['volume'].apply(convert_volume)

# Remove o símbolo '%' e converte "Change %" para float
data['change'] = data['change'].str.replace('%', '', regex=False).astype(float)

# Ordena por data crescente (geralmente o arquivo vem invertido)
data = data.sort_values(by='date')

print("\nDados após limpeza:")
print(data.head())

# ============================================================
# 3. Visualizações iniciais
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['open'], label="Open", color="blue")
plt.plot(data['date'], data['close'], label="Price", color="red")
plt.title('Open-Close Price over Time')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['volume'], label="Volume", color="orange")
plt.title("Stock Volume over Time")
plt.legend()
plt.show()

# Seleciona apenas valores numéricos
numeric_data = data.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ============================================================
# 4. Preparação dos dados para o modelo
# ============================================================

stock_close = data.filter(["close"])
dataset = stock_close.values

# Define o tamanho do conjunto de treinamento (95%)
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Escala os valores (para LSTM)
scaler = StandardScaler()
scale_data = scaler.fit_transform(dataset)

# Cria o conjunto de treino
training_data = scale_data[:training_data_len]

X_train, Y_train = [], []

# Janelas de 60 dias
for i in range(60, len(training_data)):
    X_train.append(training_data[i - 60:i, 0])
    Y_train.append(training_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# ============================================================
# 5. Construção do modelo LSTM
# ============================================================

model = keras.models.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model.summary()

model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

# ============================================================
# 6. Treinamento do modelo
# ============================================================

training = model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)

# ============================================================
# 7. Preparação dos dados de teste
# ============================================================

test_data = scale_data[training_data_len - 60:]
X_test, Y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ============================================================
# 8. Previsões
# ============================================================

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# ============================================================
# 9. Visualização dos resultados
# ============================================================

train = data[:training_data_len]
test = data[training_data_len:].copy()
test['predictions'] = predictions

plt.figure(figsize=(12, 8))
plt.plot(train['date'], train['close'], label='Train', color='blue')
plt.plot(test['date'], test['close'], label='Test', color='red')
plt.plot(test['date'], test['predictions'], label='Predictions', color='green')
plt.title("Grupo Mateus Stock Price Predictions (LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (R$)")
plt.legend()
plt.show()
