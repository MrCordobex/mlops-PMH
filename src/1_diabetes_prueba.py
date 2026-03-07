import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configurar MLflow con SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Diabetes_Prueba")

# 2. Cargar el dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1) # Asumiendo que la columna objetivo se llama 'Outcome'
y = df["Outcome"]

# Para esta prueba rápida, hacemos un split simple (luego haremos el 70/20/10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_name = "Diabetes_LR_Prueba"

# 3. Entrenar y registrar en MLflow
with mlflow.start_run(run_name="Prueba_Inicial") as run:
    # Entrenar modelo
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Registrar métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Registrar el modelo con un ejemplo de entrada
    input_example = X_test.iloc[:2]
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
        registered_model_name=model_name,
    )

print(f"Entrenamiento finalizado.")
print(f"Métricas -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print(f"Modelo registrado como '{model_name}'")