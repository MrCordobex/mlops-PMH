import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. Buscar el experimento
experiment = client.get_experiment_by_name("Diabetes_Experimentos_Completos")

# 2. Buscar todos los runs de ese experimento
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# 3. Encontrar el mejor F1 y el mejor Recall
best_f1_run = max(runs, key=lambda r: r.data.metrics.get("f1_score", 0))
best_recall_run = max(runs, key=lambda r: r.data.metrics.get("recall", 0))

print("=== SELECCIÓN DE LOS MEJORES MODELOS ===")
print(f"Mejor F1-Score: {best_f1_run.data.metrics['f1_score']:.4f} (Run ID: {best_f1_run.info.run_id})")
print(f"Mejor Recall:   {best_recall_run.data.metrics['recall']:.4f} (Run ID: {best_recall_run.info.run_id})")

# 4. Registrar ambos en MLflow
model_name = "Diabetes_Model_Seleccion"

print("\n=== REGISTRANDO MODELOS ===")
# Registrar el de mejor F1
res_f1 = mlflow.register_model(model_uri=f"runs:/{best_f1_run.info.run_id}/model", name=model_name)
print(f"Registrado Mejor F1 -> Versión: {res_f1.version}")

# Registrar el de mejor Recall (si es un run distinto, se registrará como versión 2)
if best_f1_run.info.run_id != best_recall_run.info.run_id:
    res_rec = mlflow.register_model(model_uri=f"runs:/{best_recall_run.info.run_id}/model", name=model_name)
    print(f"Registrado Mejor Recall -> Versión: {res_rec.version}")
else:
    print("El modelo con mejor F1 también tiene el mejor Recall. Solo se registra una vez.")