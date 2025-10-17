.PHONY: setup train validate mlflow-ui clean

# --- Instala dependencias ---
setup:
	@echo "📦 Instalando dependencias..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# --- Entrenamiento y registro del modelo ---
train:
	@echo "🚀 Entrenando modelo y registrando con MLflow..."
	python train.py

# --- Validación del modelo entrenado ---
validate:
	@echo "🧪 Validando modelo registrado..."
	python validate-1.py

# --- Interfaz local de MLflow (opcional) ---
mlflow-ui:
	@echo "🌐 Abriendo interfaz de MLflow en http://localhost:5000 ..."
	mlflow ui --backend-store-uri file://$$(pwd)/mlruns --port 5000

# --- Limpieza de archivos temporales ---
clean:
	@echo "🧹 Limpiando archivos temporales..."
	rm -rf __pycache__ .pytest_cache .mypy_cache
