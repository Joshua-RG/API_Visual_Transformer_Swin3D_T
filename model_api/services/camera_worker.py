import time
import os
import sys
from multiprocessing import Queue
from queue import Empty
from collections import deque
import numpy as np
from typing import Union, List

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 2 niveles: .../services -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    from config import config
    from processing.video_processor import preprocess_clip
    from services.event_recorder import EventRecorder
    from services.stream_reader.file_reader import FileReader
    from services.stream_reader.base_reader import BaseReader
    
    # --- ¡NUEVO IMPORT! ---
    # Importamos el wrapper del detector de personas que creamos
    from onnx_model.onnx_person_detector import PersonDetector

except ImportError as e:
    print(f"Error fatal en 'camera_worker.py': No se pudo importar un módulo. {e}")
    sys.exit(1)

def run_camera_worker(
    camera_id: str,
    reader_type: str, 
    source_path: Union[str, List[str]], # Acepta un path o una lista de paths
    inference_queue: Queue,
    control_queue: Queue,
    
    # --- ¡NUEVO PARÁMETRO! ---
    # Necesitamos la 'results_queue' para enviar los resultados "neutrales"
    # (0,0,0) cuando no hay personas, sin pasar por la GPU.
    results_queue: Queue 
):
    # Esta función se ejecuta en un proceso de CPU dedicado por cada cámara.
    print(f"[Worker-{camera_id}] Proceso iniciado.")

    stream_reader: Union[BaseReader, None] = None 
    current_recorder: Union[EventRecorder, None] = None
    person_detector: Union[PersonDetector, None] = None
    
    try:
        # --- 1. Inicialización ---
        
        # 1a. Cargar el detector de personas (se cargará en CPU)
        try:
            person_detector = PersonDetector()
            print(f"[Worker-{camera_id}] Detector de personas (YOLOv8n) inicializado.")
        except Exception as e:
            print(f"[Worker-{camera_id}] CRÍTICO: No se pudo cargar PersonDetector: {e}")
            return # Salir del worker si el filtro de conteo falla

        print(f"[Worker-{camera_id}] Iniciando lector tipo '{reader_type}'")
        
        # 1b. Fábrica (factory) para construir el lector de video adecuado
        if reader_type == "file":
            stream_reader = FileReader(source_path)
        # elif reader_type == "rtsp":
        #     stream_reader = RtspReader(source_path) # Para producción
        else:
            raise ValueError(f"Tipo de lector no válido: {reader_type}")

        # 1c. Obtener FPS y calcular tamaños de búfer
        source_fps = stream_reader.get_fps()
        if source_fps == 0 or source_fps > 1000: # Fallback para FPS inválidos
            print(f"[Worker-{camera_id}] FPS de fuente no válido ({source_fps}), usando {config.TARGET_FPS}.")
            source_fps = config.TARGET_FPS

        CLIP_DURATION_SEC = config.CLIP_LEN / config.TARGET_FPS
        INFERENCE_BUFFER_SIZE = int(CLIP_DURATION_SEC * source_fps)
        PRE_ROLL_BUFFER_SIZE = int(config.PRE_ROLL_SECONDS * source_fps)

        inference_buffer = deque(maxlen=INFERENCE_BUFFER_SIZE)
        pre_roll_buffer = deque(maxlen=PRE_ROLL_BUFFER_SIZE)
        
        print(f"[Worker-{camera_id}] Búfer de Inferencia: {INFERENCE_BUFFER_SIZE} frames.")
        print(f"[Worker-{camera_id}] Búfer de Pre-Rollo: {PRE_ROLL_BUFFER_SIZE} frames.")

        frame_counter = 0
        delay_por_frame = 1.0 / source_fps # "Freno" para simular FPS reales
        last_known_probs = np.array([0.0] * len(config.CLASSES))

        # --- 2. Bucle Principal del Worker ---
        while True:
            loop_start_time = time.time()
            
            # 2a. Leer Frame
            ret, frame = stream_reader.read()
            if not ret:
                print(f"[Worker-{camera_id}] El stream de video ha terminado.")
                break
            
            frame_counter += 1
            
            # 2b. Almacenar en Búferes
            inference_buffer.append(frame)
            pre_roll_buffer.append(frame)

            # 2c. Lógica de Grabación (Revisar comandos de la API)
            # (Esta lógica permanece 100% idéntica a tu código original)
            while not control_queue.empty():
                try:
                    command = control_queue.get_nowait()
                    
                    if isinstance(command, np.ndarray):
                        last_known_probs = command
                    
                    elif command == "START_RECORDING" and current_recorder is None:
                        print(f"[Worker-{camera_id}] Recibida orden: START_RECORDING")
                        current_recorder = EventRecorder(
                            camera_id=camera_id,
                            pre_roll_frames=list(pre_roll_buffer),
                            source_fps=source_fps
                        )
                        current_recorder.start()
                    
                    elif command == "STOP_RECORDING" and current_recorder is not None:
                        print(f"[Worker-{camera_id}] Recibida orden: STOP_RECORDING")
                        current_recorder.close()
                        current_recorder = None
                
                except Empty:
                    break 
            
            if current_recorder is not None:
                current_recorder.add_frame(frame, last_known_probs) 

            # --- 2d. LÓGICA DE INFERENCIA Y FILTRADO (¡MODIFICADA!) ---
            if (len(inference_buffer) == INFERENCE_BUFFER_SIZE and 
                frame_counter % config.STRIDE == 0):
                
                person_count = -1 # Valor de error por defecto
                try:
                    # 1. Ejecutar el pre-filtro de conteo de personas (en CPU)
                    #    Usamos el 'frame' más reciente.
                    if person_detector:
                        person_count = person_detector.count_persons(frame)
                    else:
                        print(f"[Worker-{camera_id}] ERROR: person_detector no está inicializado.")
                
                except Exception as e:
                    print(f"[Worker-{camera_id}] ADVERTENCIA: Fallo en PersonDetector: {e}")

                # 2. Decidir el camino de inferencia
                if person_count >= 2:
                    # 2a. SÍ HAY PERSONAS -> Enviar a la GPU para análisis Swin3D
                    try:
                        tensor = preprocess_clip(list(inference_buffer))
                        
                        if not np.isfinite(tensor).all():
                            print(f"[Worker-{camera_id}] ADVERTENCIA: Tensor corrupto (NaN/Inf). Omitiendo clip.")
                        else:
                            # Enviar a la cola de la GPU (inference_service)
                            inference_queue.put((camera_id, tensor))
                    
                    except Exception as e:
                        print(f"[Worker-{camera_id}] Error al pre-procesar clip: {e}")
                
                elif person_count < 0:
                    # 2b. HUBO UN ERROR EN YOLO -> No hacer nada (solo log)
                    print(f"[Worker-{camera_id}] Error en el detector de personas. Omitiendo inferencia este ciclo.")
                
                else:
                    # 2c. NO HAY PERSONAS (< 2) -> Omitir la GPU
                    # Enviar un resultado neutral (0,0,0) directamente al EventManager
                    # para mantener la cámara "viva" en el frontend.
                    neutral_probs = np.array([0.0] * len(config.CLASSES))
                    results_queue.put((camera_id, neutral_probs))
                    
                    # También actualizamos last_known_probs por si estamos grabando
                    last_known_probs = neutral_probs
            
            # --- FIN DE LA MODIFICACIÓN ---

            # 2e. Controlar los FPS
            time_elapsed = time.time() - loop_start_time
            sleep_time = delay_por_frame - time_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit):
        print(f"[Worker-{camera_id}] Deteniendo...")
    except Exception as e:
        print(f"[Worker-{camera_id}] CRÍTICO: Error inesperado: {e}")
    finally:
        # --- 3. Limpieza ---
        print(f"[Worker-{camera_id}] Liberando recursos...")
        if current_recorder is not None:
            current_recorder.close()
        if stream_reader is not None:
            stream_reader.release()
        print(f"[Worker-{camera_id}] Proceso terminado.")