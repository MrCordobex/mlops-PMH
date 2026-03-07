import requests
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Cargar datos para coger unos pacientes de prueba
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preparar el JSON con 5 pacientes de prueba respetando el formato DataFrame
payload = {
    "dataframe_split": X_test.iloc[:5].to_dict(orient="split")
}

# 3. Hacer la petición POST a la API local
try:
    response = requests.post(
        url="http://localhost:1234/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    response.raise_for_status() # Lanza error si la API no responde bien
    
    # 4. Leer predicciones
    preds = response.json()["predictions"]
    
    print("=== PREDICCIONES DE LA API VS VALORES REALES ===")
    for idx, (pred, real) in enumerate(zip(preds, y_test.iloc[:5])):
        print(f"Paciente {idx + 1} -> Predicción: {pred} | Real: {real}")

except Exception as e:
    print(f"Error al conectar con la API: {e}")