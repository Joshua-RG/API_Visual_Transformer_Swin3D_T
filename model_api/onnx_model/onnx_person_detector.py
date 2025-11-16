import onnxruntime
import numpy as np
import threading
import sys
import os
import cv2  

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 2 niveles: .../onnx_model -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    # Importamos el módulo (archivo) config.py
    from config import config
except ImportError as e:
    print(f"Error fatal en 'onnx_person_detector.py': No se pudo importar 'config'. {e}")
    sys.exit(1)


class PersonDetector:
    """
    Clase contenedora para el modelo de detección de personas (YOLOv8n).
    Implementa "Lazy Loading" para coherencia de estilo y carga el modelo
    forzosamente en CPU para no competir con el 'inference_service' de la GPU.
    """

    def __init__(self):
        # Constructor (Lazy Loading). No carga el modelo, solo prepara la config.
        self.session: onnxruntime.InferenceSession | None = None
        self.lock = threading.Lock() # Asegura que el modelo se cargue solo una vez
  
        # Construimos la ruta al modelo YOLOv8n
        self.model_path = os.path.join(
            config.BASE_DIR, 
            "onnx_model", 
            "person_detector", 
            "yolov8n.onnx"
        )
        
        # --- ¡CRÍTICO! ---
        # Forzamos el uso de CPUExecutionProvider.
        # Esto evita conflictos con el 'inference_service' (GPU).
        self.providers = ['CPUExecutionProvider']
        
        # Prepara las opciones de la sesión
        self.options = onnxruntime.SessionOptions()
        self.options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Parámetros fijos para el modelo YOLOv8n (imgsz=320)
        self.input_height = 320
        self.input_width = 320
        self.person_class_id = 0  # 'person' es la clase 0 en el dataset COCO
        self.confidence_threshold = 0.4 # Umbral para contar una persona

    def _load_model(self):
        # Método privado para cargar el modelo. Se llama solo una vez.
        # Esto se ejecutará DENTRO de cada proceso 'camera_worker'.
        print(f"[PersonDetector] Cargando modelo ONNX desde: {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            print(f"[PersonDetector] FATAL: No se encontró el modelo en {self.model_path}")
            raise FileNotFoundError(f"Modelo YOLO no encontrado: {self.model_path}")

        self.session = onnxruntime.InferenceSession(
            self.model_path,
            self.options,
            self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Imprime el proveedor que realmente se está usando (debe ser CPUExecutionProvider)
        print(f"[PersonDetector] Modelo cargado y listo en: {self.session.get_providers()[0]}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Preprocesa un frame de OpenCV (H, W, C) para YOLOv8 (1, 3, 320, 320)
        
        # 1. Redimensionar manteniendo el aspect ratio (con letterboxing)
        img_h, img_w, _ = frame.shape
        scale = min(self.input_width / img_w, self.input_height / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        resized_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Crear un canvas y pegar la imagen (letterbox)
        # (El valor 114 es un gris estándar usado por YOLO para el padding)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        top_pad = (self.input_height - new_h) // 2
        left_pad = (self.input_width - new_w) // 2
        canvas[top_pad:top_pad + new_h, left_pad:left_pad + new_w] = resized_img
        
        # 3. Convertir BGR (OpenCV) a RGB
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # 4. Normalizar (0-255 -> 0.0-1.0) y transponer (HWC -> CHW)
        input_tensor = canvas_rgb.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        
        # 5. Añadir dimensión de batch (1, C, H, W)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor

    def _postprocess(self, output: np.ndarray) -> int:
        # Procesa la salida de YOLO (1, 84, 2100) y devuelve el conteo de personas.
        # 84 = 4 (bbox) + 80 (clases)
        # 2100 = propuestas de detección para 320x320
        
        # Transponer la salida a (1, 2100, 84) para iterar fácilmente
        output = output.transpose(0, 2, 1)
        
        person_count = 0
        
        # Iterar sobre las 2100 propuestas de detección
        for det in output[0]:
            # det[0:4] son BBox (cx, cy, w, h)
            # det[4:] son los 80 scores de clase
            
            class_scores = det[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Comprobar si es una persona Y si supera nuestro umbral
            if (confidence > self.confidence_threshold and 
                class_id == self.person_class_id):
                
                person_count += 1
        
        # Nota: Esto es un conteo simple sin NMS (Non-Max Suppression)
        # Para un pre-filtro de ">= 2" es mucho más rápido y suficiente.
        return person_count

    def count_persons(self, frame: np.ndarray) -> int:
        """
        Función principal. Recibe un frame de OpenCV (BGR, HWC) y 
        devuelve el número de personas detectadas.
        """
        
        # --- Carga Perezosa (Lazy Loading) ---
        # (Sigue el estilo de onnx_detector.py)
        with self.lock:
            if self.session is None:
                try:
                    self._load_model()
                except Exception as e:
                    print(f"[PersonDetector] CRÍTICO: Fallo al cargar el modelo: {e}")
                    return -1 # Devolvemos -1 para indicar un error
        
        if self.session is None:
            print("[PersonDetector] ERROR: Sesión no cargada. Omitiendo conteo.")
            return -1 # El modelo no se pudo cargar

        try:
            # 1. Preprocesar frame
            input_tensor = self._preprocess(frame)
            
            # 2. Preparar inputs
            inputs = {self.input_name: input_tensor}
            
            # 3. Ejecutar inferencia en CPU
            # La salida es una lista, tomamos el primer (y único) elemento [0]
            output_data = self.session.run([self.output_name], inputs)[0]
            
            # 4. Post-procesar y contar
            count = self._postprocess(output_data)
            
            return count
        
        except Exception as e:
            print(f"[PersonDetector] ERROR durante la inferencia: {e}")
            return -1 # Devolver -1 para indicar un error en la inferencia