import joblib
from xgboost import XGBRegressor

# Supondo que 'best_xgb' e 'scaler' são as variáveis do seu script

# 1. Salvar o modelo XGBoost treinado
best_xgb.save_model("melhor_modelo_xgboost.json")

# 2. Salvar o normalizador (scaler) que foi treinado com os dados de treino
joblib.dump(scaler, 'scaler.gz')

print("Modelo e normalizador foram salvos com sucesso!")