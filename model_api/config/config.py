import numpy as np
import os

# --- Definición de Rutas ---

# Define el directorio base del proyecto (la carpeta 'model_api')
# Sube dos niveles desde este archivo (config -> model_api)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta al modelo ONNX exportado
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "onnx_model", "swin3d_t.onnx")
# Ruta para guardar los clips de video de eventos detectados
SAVE_CLIP_PATH = os.path.join(BASE_DIR, "data", "clips_guardados")
# Ruta para guardar los logs JSON de eventos detectados
SAVE_LOG_PATH = os.path.join(BASE_DIR, "data", "logs_eventos")


# --- Parámetros del Modelo ---

# Clases de violencia que el modelo puede detectar 
CLASSES = ["Forcejeo", "Patada", "Golpe"]

# Longitud del clip (en frames) que el modelo espera recibir
CLIP_LEN = 32
# FPS a los que el modelo fue entrenado (todos los videos se normalizarán a esto)
TARGET_FPS = 30


# --- Parámetros de Preprocesamiento ---

# El tamaño al que se redimensiona el lado más corto del frame
INPUT_RESIZE = 256
# El tamaño del recorte central que se pasa al modelo
INPUT_CROP_SIZE = 224

# Media y Desviación Estándar de ImageNet (escaladas a [0, 1])
# Usadas para normalizar los frames antes de la inferencia
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --- Parámetros del Pipeline ---

# Cada cuántos frames se ejecutará una predicción (ventana deslizante)
STRIDE = 16

# Tiempo (en segundos) de video que se guarda ANTES de que se detecte un evento
PRE_ROLL_SECONDS = 5
# Umbral de probabilidad (ej. 0.7 = 70%) para disparar una alerta/grabación
ALERT_THRESHOLD = 0.7


# --- Parámetros del Servicio de Inferencia ---

# (Actualmente no se usa 'batching' porque el modelo no lo soporta)
# (Se mantiene para futura optimización del modelo)
MAX_BATCH_SIZE = 16
BATCH_TIMEOUT_SECONDS = 0.1  # (100 ms)

# Lista de proveedores de ONNX Runtime, en orden de prioridad
INFERENCE_PROVIDERS = [
    'CUDAExecutionProvider',    # Para GPUs NVIDIA
    'DmlExecutionProvider',     # Para GPUs AMD/Intel (Windows)
    'CPUExecutionProvider'      # Respaldo para CPU
]