import os
import sys
import traceback
import platform
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

# Silenciar advertencias de convergencia (para que el pipeline se vea limpio)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Definir rutas base ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")

# Detectar sistema operativo y crear URI válida para MLflow
if platform.system() == "Windows":
    tracking_uri = f"file:///{mlruns_dir.replace(os.sep, '/')}"
    artifact_location = f"file:///{mlruns_dir.replace(os.sep, '/')}"
else:
    tracking_uri = f"file://{mlruns_dir}"
    artifact_location = f"file://{mlruns_dir}"

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# Asegurar que exista el directorio de MLruns
os.makedirs(mlruns_dir, exist_ok=True)

# Configurar MLflow
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o recuperar experimento ---
# Usa un nombre diferente para evitar conflicto entre Windows y Linux
experiment_name = "CI-CD-Lab2-Actions"
experiment_id = None

try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    error_msg = str(e)
    if "RESOURCE_ALREADY_EXISTS" in error_msg or "already exists" in error_msg:
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---")
        else:
            print(f"--- ERROR: No se pudo obtener el experimento existente '{experiment_name}' ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e

if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}' ---")
    sys.exit(1)

# ===================================================================================
# === CARGA Y ENTRENAMIENTO DEL MODELO CON DATASET EXTERNO ===
# ===================================================================================
print("\n--- Cargando dataset externo: drug.csv ---")

data_path = os.path.join(workspace_dir, "drug.csv")
if not os.path.exists(data_path):
    print(f"❌ ERROR: No se encontró el archivo {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)
print(f"✅ Dataset cargado correctamente. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# Codificar variables categóricas
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

# Separar variables
X = df_encoded.drop(columns=["Drug"])
y = df_encoded["Drug"]

# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Modelo entrenado. Accuracy: {accuracy:.4f}")

# ===================================================================================
# === REGISTRO EN MLflow ===
# ===================================================================================
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
run = None

try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        # Registrar parámetros y métricas
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("dataset", "drug.csv")
        mlflow.log_metric("accuracy", float(accuracy))

        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Guardar modelo localmente
        joblib.dump(model, "model.pkl")
        print("💾 Modelo guardado localmente como model.pkl")

        print(f"✅ Registro completado en MLflow | Accuracy={accuracy:.4f}")

except Exception as e:
    print("\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- CWD actual en el error: {os.getcwd()} ---")
    print(f"--- Tracking URI usada: {mlflow.get_tracking_uri()} ---")
    print(f"--- Experiment ID: {experiment_id} ---")
    if run:
        print(f"URI del Artefacto del Run: {run.info.artifact_uri}")
    sys.exit(1)

