# src/validate.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys
import os

# Parámetro de umbral mínimo aceptable
THRESHOLD = 0.80  # Precisión mínima esperada

print("🧪 Validando modelo registrado...")

# --- Cargar el dataset externo (mismo usado en train.py) ---
print("--- Debug: Cargando dataset externo: drug.csv ---")
data_path = os.path.join(os.getcwd(), "drug.csv")

if not os.path.exists(data_path):
    print(f"❌ ERROR: No se encontró el archivo '{data_path}'.")
    sys.exit(1)

try:
    df = pd.read_csv(data_path)
    print(f"✅ Dataset cargado correctamente. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
except Exception as e:
    print(f"❌ ERROR al cargar el dataset: {e}")
    sys.exit(1)

# --- Preprocesamiento: codificar variables categóricas ---
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

# Variables predictoras y objetivo
X = df_encoded.drop(columns=["Drug"])
y = df_encoded["Drug"]

# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

# --- Cargar el modelo entrenado ---
model_filename = "model.pkl"
model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))
print(f"--- Debug: Intentando cargar modelo desde: {model_path} ---")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo '{model_path}'. Ejecuta primero 'make train'.")
    print(f"--- Debug: Archivos disponibles en {os.getcwd()}: ---")
    try:
        print(os.listdir(os.getcwd()))
    except Exception as list_err:
        print(f"(No se pudo listar el directorio: {list_err})")
    print("---")
    sys.exit(1)

# --- Realizar predicciones ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred = model.predict(X_test)
except ValueError as pred_err:
    print(f"❌ ERROR durante la predicción: {pred_err}")
    print(f"Modelo esperaba {model.n_features_in_} features, X_test tiene {X_test.shape[1]}.")
    sys.exit(1)

accuracy = accuracy_score(y_test, y_pred)
print(f"🔍 Precisión del modelo: {accuracy:.4f} (umbral: {THRESHOLD})")

# --- Validar desempeño ---
if accuracy >= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad (Accuracy OK).")
else:
    print("❌ El modelo no cumple el umbral de precisión. Deteniendo pipeline.")
    sys.exit(1)

# ===========================================================
# ✅ Exportar accuracy real para GitHub Actions
# ===========================================================
try:
    # Guardar dentro del proyecto (solución para Windows local)
    accuracy_path = os.path.join(os.getcwd(), "accuracy.txt")

    print(f"💾 Guardando accuracy real en: {accuracy_path}")
    with open(accuracy_path, "w") as f:
        f.write(f"{accuracy:.4f}\n")

    print(f"🏁 Accuracy final del modelo: {accuracy:.4f}")
except Exception as e:
    print(f"⚠️ Error al guardar accuracy.txt: {e}")
