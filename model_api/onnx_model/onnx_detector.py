import onnxruntime
import numpy as np
import threading
import sys
import os

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 2 niveles: .../onnx_model -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    # Importamos el módulo (archivo) config.py
    from config import config
except ImportError as e:
    print(f"Error fatal en 'detector.py': No se pudo importar 'config'. {e}")
    sys.exit(1)


class ViolenceDetector:
    # Clase contenedora para el modelo de inferencia ONNX
    # Implementa "Lazy Loading" para ser segura con multiprocessing

    def __init__(self):
        # Constructor (Lazy Loading). No carga el modelo, solo prepara la config.
        self.session: onnxruntime.InferenceSession | None = None
        self.lock = threading.Lock() # Asegura que el modelo se cargue solo una vez
  
        # Carga la configuración desde el archivo config.py
        self.model_path = config.ONNX_MODEL_PATH
        self.providers = config.INFERENCE_PROVIDERS
        
        # Prepara las opciones de la sesión
        self.options = onnxruntime.SessionOptions()
        self.options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Función helper para aplicar sigmoid (el modelo devuelve logits)
        return 1 / (1 + np.exp(-x))

    def _load_model(self):
        # Método privado para cargar el modelo. Se llama solo una vez.
        # Esto se ejecuta DENTRO del proceso 'inference_service'.
        print(f"[Detector] Cargando modelo ONNX desde: {self.model_path}...")
        self.session = onnxruntime.InferenceSession(
            self.model_path,
            self.options,
            self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Imprime el proveedor que realmente se está usando (ej. CUDAExecutionProvider)
        print(f"[Detector] Modelo cargado y listo en: {self.session.get_providers()[0]}")
        print(f"[Detector] Nombre de Input: {self.input_name} | Nombre de Output: {self.output_name}")

    def predict_batch(self, preprocessed_batch: np.ndarray) -> np.ndarray:
        # Ejecuta la inferencia en un LOTE de clips preprocesados.
        # Args:
        #     preprocessed_batch (np.array): Lote de clips (N, 3, 32, 224, 224).
        # Returns:
        #     np.array: Lote de probabilidades (N, 3).
        
        # --- Carga Perezosa (Lazy Loading) ---
        # Revisa (de forma segura) si el modelo ya está cargado.
        with self.lock:
            if self.session is None:
                # Si no está cargado, cárgalo ahora.
                self._load_model()
        # --- Fin de Carga Perezosa ---

        # 1. Preparar el diccionario de entrada
        inputs = {self.input_name: preprocessed_batch}

        # 2. Ejecutar la inferencia
        logits_batch = self.session.run([self.output_name], inputs)[0]

        # 3. Aplicar sigmoid a todo el lote de logits
        probabilities_batch = self._sigmoid(logits_batch)

        # 4. Devolver el array 2D completo de probabilidades (N, 3)
        return probabilities_batch