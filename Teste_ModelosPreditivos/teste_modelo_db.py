# --- ETAPA 1: CARREGAMENTO (sem conversão para absorção) ---
import pandas as pd
import numpy as np
import os

# Pega o diretório do seu script atual
current_directory = os.path.dirname(__file__)

# Constrói o caminho para o arquivo de entrada
input_file_path = os.path.join(current_directory, 'banco_de_dados', 'database_reduced.csv')

# Carregar seus dados originais
data = pd.read_csv(input_file_path)

print("Processo concluído e arquivo 'database_reduced.csv' carregado com sucesso!")

# --- ETAPA 2: PREPARAÇÃO PARA MODELAGEM ---

# Bibliotecas
import random
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
from scikeras.wrappers import KerasRegressor

# Semente
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Trabalhar direto com o dataframe original
df = data.copy()

# Remoção de outliers pelo método IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df = df[~((df < limite_inferior) | (df > limite_superior)).any(axis=1)]

# Separar entradas (X) e saídas (y) - usando S11 e Freq como targets
X = df[['Tx1', 'Tx2', 'Tx3', 'w']].copy()
y = df[['S11_1', 'S11_2', 'S11_3', 'Freq_1', 'Freq_2', 'Freq_3']]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- ETAPA 3: OTIMIZAÇÃO E AVALIAÇÃO DOS MODELOS COM VALIDAÇÃO CRUZADA ---
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=seed)
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
def create_mlp_model(neuron_count=128, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Dense(neuron_count, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neuron_count // 2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(y_train.shape[1])
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

fit_params = {
    'callbacks': [EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=0)]
}

param_grid_mlp = {
    'model__neuron_count': [64, 128],
    'model__dropout_rate': [0.2, 0.3],
    'model__learning_rate': [0.001, 0.01],
    'batch_size': [16, 32],
    'epochs': [200]
}

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

results_df = pd.DataFrame(results).T
results_df = results_df[['R² no Teste', 'Melhor R² (CV)', 'Melhores Parâmetros']]
print(results_df)

best_model_name = results_df['R² no Teste'].idxmax()
print(f"\n🏆 O melhor modelo com base no R² do conjunto de teste é: {best_model_name}")
