# --- ETAPA 1: CARREGAMENTO E PRÉ-PROCESSAMENTO (Idêntico ao original) ---
import pandas as pd
import numpy as np
import os

# Pega o diretório do seu script atual
current_directory = os.path.dirname(__file__)

# Constrói o caminho para o arquivo de entrada
input_file_path = os.path.join(current_directory, 'banco_de_dados', 'database_reduced.csv')

# Carregar seus dados
data = pd.read_csv(input_file_path)

# Função para converter dB para linear
def db_to_linear(s11_db):
    return 10 ** (s11_db / 20)

# Converter cada coluna S11 (dB) para absorção
for i in [1, 2, 3]:
    s11_db_col = f'S11_{i}'
    abs_col = f'Absorption_{i}'
    s11_linear = db_to_linear(data[s11_db_col])
    data[abs_col] = 1 - np.abs(s11_linear) ** 2

# Constrói o caminho para o arquivo de saída
output_file_path = os.path.join(current_directory, 'banco_de_dados', 'absorcao_resultados.csv')

# Salvar os resultados intermediários
data.to_csv(output_file_path, index=False)

print("Processo concluído e arquivo 'absorcao_resultados.csv' salvo com sucesso!")

# --- ETAPA 2: PREPARAÇÃO PARA MODELAGEM ---

# Bibliotecas de manipulação e análise de dados
import random
import pandas as pd
import numpy as np

# Bibliotecas de aprendizado de máquina
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Bibliotecas para redes neurais
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
# O scikeras é o wrapper moderno e recomendado para usar Keras com Scikit-learn
from scikeras.wrappers import KerasRegressor

# Definir a semente para reprodutibilidade
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Carregar os dados processados
file_path = os.path.join(current_directory, 'banco_de_dados', 'absorcao_resultados.csv')
df = pd.read_csv(file_path)

# Remoção de outliers pelo método IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df = df[~((df < limite_inferior) | (df > limite_superior)).any(axis=1)]

# Separar entradas (X) e saídas (y)
X = df[['Tx1', 'Tx2', 'Tx3', 'w']].copy()
y = df[['Absorption_1', 'Absorption_2', 'Absorption_3', 'Freq_1', 'Freq_2', 'Freq_3']]

# Engenharia de Features (mantida como no original)
X['diff_1'] = X['Tx1'] - X['Tx2']
X['diff_2'] = X['Tx2'] - X['Tx3']
X['diff_3'] = X['Tx3'] - X['w']
X['ratio_1'] = X['Tx1'] / X['Tx2']
X['ratio_2'] = X['Tx2'] / X['Tx3']
X['ratio_3'] = X['Tx3'] / X['w']
X['Tx1_sq'] = X['Tx1'] ** 2
X['Tx2_sq'] = X['Tx2'] ** 2
X['Tx3_sq'] = X['Tx3'] ** 2
X['w_sq'] = X['w'] ** 2
X['interaction_1'] = X['Tx1'] * X['Tx2']
X['interaction_2'] = X['Tx2'] * X['Tx3']
X['interaction_3'] = X['Tx3'] * X['w']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- ETAPA 3: OTIMIZAÇÃO E AVALIAÇÃO DOS MODELOS COM VALIDAÇÃO CRUZADA ---

# Objeto de validação cruzada a ser usado por todos os modelos para consistência
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=seed)

# Dicionário para armazenar os resultados
results = {}

# --- Modelo 1: XGBoost ---
print("--- Otimizando XGBoost ---")
param_grid_xgb = {
    'max_depth': [5, 10, 15],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1]
}
grid_search_xgb = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=seed),
    param_grid=param_grid_xgb,
    cv=cv_strategy,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)
grid_search_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_scaled)
results['XGBoost'] = {
    'Melhor R² (CV)': grid_search_xgb.best_score_,
    'R² no Teste': r2_score(y_test, y_pred_xgb),
    'Melhores Parâmetros': grid_search_xgb.best_params_
}

# --- Modelo 2: Kernel PCA + Random Forest ---
print("\n--- Otimizando Kernel PCA + Random Forest ---")
pipeline_rf = Pipeline([
    ('kpca', KernelPCA(random_state=seed)),
    ('rf', RandomForestRegressor(random_state=seed))
])
param_grid_rf = {
    'kpca__n_components': [4, 6, 8],
    'kpca__kernel': ['rbf', 'poly', 'sigmoid'],
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 15]
}
grid_search_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid_rf,
    cv=cv_strategy,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)
grid_search_rf.fit(X_train_scaled, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
results['Kernel PCA + RF'] = {
    'Melhor R² (CV)': grid_search_rf.best_score_,
    'R² no Teste': r2_score(y_test, y_pred_rf),
    'Melhores Parâmetros': grid_search_rf.best_params_
}

# --- Modelo 3: Rede Neural (MLP) ---
print("\n--- Otimizando Rede Neural (MLP) ---")
# Função para criar o modelo Keras (necessário para o wrapper)
def create_mlp_model(neuron_count=128, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Dense(neuron_count, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neuron_count // 2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(y_train.shape[1]) # Camada de saída com o número de targets
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Parâmetros para passar ao método .fit()
fit_params = {
    'callbacks': [EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=0)]
}

# Espaço de busca para a rede neural
param_grid_mlp = {
    'model__neuron_count': [64, 128],
    'model__dropout_rate': [0.2, 0.3],
    'model__learning_rate': [0.001, 0.01],
    'batch_size': [16, 32],
    'epochs': [200]
}

# Criar o modelo KerasRegressor
mlp_model = KerasRegressor(model=create_mlp_model, verbose=0)

grid_search_mlp = GridSearchCV(
    estimator=mlp_model,
    param_grid=param_grid_mlp,
    cv=cv_strategy,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)
grid_search_mlp.fit(X_train_scaled, y_train, **fit_params)
best_mlp = grid_search_mlp.best_estimator_
y_pred_mlp = best_mlp.predict(X_test_scaled)
results['Rede Neural (MLP)'] = {
    'Melhor R² (CV)': grid_search_mlp.best_score_,
    'R² no Teste': r2_score(y_test, y_pred_mlp),
    'Melhores Parâmetros': grid_search_mlp.best_params_
}


# --- ETAPA 4: COMPARAÇÃO FINAL DOS RESULTADOS ---
print("\n\n--- Tabela Comparativa de Resultados ---\n")

# Criar um DataFrame com os resultados para fácil visualização
results_df = pd.DataFrame(results).T # .T transpõe o DataFrame
results_df = results_df[['R² no Teste', 'Melhor R² (CV)', 'Melhores Parâmetros']] # Reordenar colunas

# Imprimir DataFrame formatado
print(results_df)

# Encontrar o melhor modelo com base no R² do conjunto de teste
best_model_name = results_df['R² no Teste'].idxmax()
print(f"\n🏆 O melhor modelo com base no R² do conjunto de teste é: {best_model_name}")

import joblib

# 1. Salvar o modelo XGBoost treinado
best_xgb.save_model("melhor_modelo_xgboost.json")

# 2. Salvar o normalizador (scaler) que foi treinado com os dados de treino
joblib.dump(scaler, 'scaler.gz')

print("Modelo e normalizador foram salvos com sucesso!")