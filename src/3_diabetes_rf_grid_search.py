import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Configuración MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Diabetes_Experimentos_Completos")

# 2. Cargar Dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# --- SPLIT 70% / 20% / 10% ---
# Guardamos el 10% de Test en la "caja fuerte"
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Del 90% restante, sacamos el 20% para Validación y el 70% para Train
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.20/0.90), random_state=42)

print(f"Tamaños de particiones -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Guardamos el Test intacto para el entregable final de la API
X_test.to_csv("X_test_final.csv", index=False)
y_test.to_csv("y_test_final.csv", index=False)

# 3. Grid Search de Hiperparámetros para Random Forest
n_estimators_list = [50, 100, 200]
max_depth_list = [None, 5, 10]

best_f1 = 0
best_run_id = None

print("\n=== Iniciando Entrenamiento de Random Forest y Registro en MLflow ===")

for n in n_estimators_list:
    for d in max_depth_list:
        # Cada combinación es un "run" diferente
        run_name = f"RF_n-{n}_depth-{d if d is not None else 'None'}"
        with mlflow.start_run(run_name=run_name) as run:
            
            # Entrenamos con TRAIN
            model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
            model.fit(X_train, y_train)

            # Predecimos con VALIDACIÓN
            y_pred_val = model.predict(X_val)

            # Calculamos métricas
            acc = accuracy_score(y_val, y_pred_val)
            prec = precision_score(y_val, y_pred_val)
            rec = recall_score(y_val, y_pred_val)
            f1 = f1_score(y_val, y_pred_val)

            # Registramos parámetros y métricas
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Registramos el modelo
            input_example = X_val.iloc[:2]
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=input_example
            )

            print(f"Run {run.info.run_id[:8]}... | n_estimators: {n}, max_depth: {d} -> F1: {f1:.4f} | Recall: {rec:.4f}")

            # Lógica para guardar el mejor modelo basado en F1
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run.info.run_id

print(f"\n¡Grid Search finalizado!")
print(f"El mejor modelo tuvo un F1 de {best_f1:.4f} (Run ID: {best_run_id})")

# 4. Registrar oficialmente el mejor modelo usando el Run ID
if best_run_id:
    model_name = "Diabetes_Best_RF_Model"
    best_model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=best_model_uri, name=model_name)
    print(f"\nModelo ganador registrado con éxito como: '{result.name}', versión: {result.version}")
else:
    print("\nNo se pudo encontrar un modelo válido para registrar.")