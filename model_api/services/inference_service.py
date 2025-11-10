import numpy as np
import time
from multiprocessing import Queue
import sys
import os

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 2 niveles: .../services -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

def run_inference_service(inference_queue: Queue, results_queue: Queue):
    # Esta función se ejecuta en un proceso de GPU dedicado.
    # PROCESA CLIPS DE UNO EN UNO (Batch Size = 1)
    # Esta es la corrección para el error [ONNXRuntimeError... Reshape node]
    # que indica que el modelo exportado no soporta 'batching' dinámico.
    
    # Importaciones movidas DENTRO de la función
    # Esto previene 'deadlocks' de CUDA al iniciar el proceso en Windows
    try:
        from onnx_model.onnx_detector import ViolenceDetector
        from config import config
    except ImportError as e:
        print(f"[InferenceService] Error de importación: {e}")
        return

    print("[InferenceService] Proceso iniciado.")
    try:
        # 1. Crear la instancia del detector
        # (El modelo real se cargará en la primera predicción - Lazy Loading)
        detector = ViolenceDetector()
    except Exception as e:
        print(f"[InferenceService] CRÍTICO: No se pudo instanciar ViolenceDetector: {e}")
        return  

    print(f"[InferenceService] Esperando el primer clip para cargar el modelo...")
    
    # 2. Bucle infinito para procesar clips (uno por uno)
    while True:
        try:
            # 1. Obtener UN clip (bloqueante)
            # item = (camera_id, tensor_data)
            # tensor_data tiene forma (3, 32, 224, 224)
            camera_id, clip_tensor = inference_queue.get()

            # 2. Crear un Lote de 1
            # Añadimos la dimensión 'N' (N=1)
            batch_tensor = np.expand_dims(clip_tensor, axis=0) # Forma -> (1, 3, 32, 224, 224)

            # 3. Predecir (Lote de 1)
            # La primera vez que se llame, cargará el modelo.
            # batch_probs tendrá forma (1, 3)
            batch_probs = detector.predict_batch(batch_tensor)
            
            # 4. Poner en la Cola de Resultados
            # Extraemos la primera (y única) predicción del lote
            probabilities = batch_probs[0] # Forma -> (3,)
            results_queue.put((camera_id, probabilities))

        except (KeyboardInterrupt, SystemExit):
            print("[InferenceService] Deteniendo...")
            break
        except Exception as e:
            # Si un tensor corrupto (NaN) logra pasar, este 'try'
            # lo atrapará y solo fallará ese clip, no todo el servicio.
            print(f"[InferenceService] Error en el bucle principal: {e}")
            time.sleep(0.1) # Pausa breve para evitar inundar logs si hay un error