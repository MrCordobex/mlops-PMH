import requests
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Cargar el dataset de Test que guardamos previamente
print("Cargando dataset de test (10%)...")
X_test = pd.read_csv("X_test_final.csv")
y_test = pd.read_csv("y_test_final.csv")

# 2. Preparar el JSON en formato Pandas (split)
payload = {
    "dataframe_split": X_test.to_dict(orient="split")
}

# 3. Hacer la petición POST a la API
print("Enviando datos a la API de MLflow (http://localhost:1234/invocations)...")
try:
    response = requests.post(
        url="http://localhost:1234/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    response.raise_for_status() 
    
    # 4. Leer predicciones
    preds = response.json()["predictions"]
    
    # 5. Calcular métricas finales
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n" + "="*40)
    print("MÉTRICAS FINALES SOBRE DATASET DE TEST (10%)")
    print("="*40)
    print(f"Total de pacientes evaluados: {len(y_test)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*40)

except Exception as e:
    print(f"Error al conectar con la API: {e}")