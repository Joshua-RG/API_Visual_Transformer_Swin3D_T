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
    # (Aquí se importará el RtspReader cuando se implemente)
except ImportError as e:
    print(f"Error fatal en 'camera_worker.py': No se pudo importar un módulo. {e}")
    sys.exit(1)

def run_camera_worker(
    camera_id: str,
    reader_type: str, 
    source_path: Union[str, List[str]], # Acepta un path o una lista de paths
    inference_queue: Queue,
    control_queue: Queue
):
    # Esta función se ejecuta en un proceso de CPU dedicado por cada cámara.
    print(f"[Worker-{camera_id}] Proceso iniciado.")

    stream_reader: Union[BaseReader, None] = None 
    current_recorder: Union[EventRecorder, None] = None
    
    try:
        # --- 1. Inicialización ---
        print(f"[Worker-{camera_id}] Iniciando lector tipo '{reader_type}'")
        
        # Fábrica (factory) para construir el lector de video adecuado
        if reader_type == "file":
            stream_reader = FileReader(source_path)
        # elif reader_type == "rtsp":
        #     stream_reader = RtspReader(source_path) # Para producción
        else:
            raise ValueError(f"Tipo de lector no válido: {reader_type}")

        # Obtener FPS y calcular tamaños de búfer
        source_fps = stream_reader.get_fps()
        if source_fps == 0 or source_fps > 1000: # Fallback para FPS inválidos
            print(f"[Worker-{camera_id}] FPS de fuente no válido ({source_fps}), usando {config.TARGET_FPS}.")
            source_fps = config.TARGET_FPS

        # Duración (seg) que el modelo espera ver (ej. 32 frames / 30 FPS = 1.06s)
        CLIP_DURATION_SEC = config.CLIP_LEN / config.TARGET_FPS
        # Tamaño del búfer de inferencia (en frames de la fuente)
        INFERENCE_BUFFER_SIZE = int(CLIP_DURATION_SEC * source_fps)
        # Tamaño del búfer de pre-grabación (en frames de la fuente)
        PRE_ROLL_BUFFER_SIZE = int(config.PRE_ROLL_SECONDS * source_fps)

        # Búfer para la IA (ej. ~32 frames si la fuente es 30 FPS)
        inference_buffer = deque(maxlen=INFERENCE_BUFFER_SIZE)
        # Búfer de memoria para grabación (ej. 150 frames si la fuente es 30 FPS)
        pre_roll_buffer = deque(maxlen=PRE_ROLL_BUFFER_SIZE)
        
        print(f"[Worker-{camera_id}] Búfer de Inferencia: {INFERENCE_BUFFER_SIZE} frames.")
        print(f"[Worker-{camera_id}] Búfer de Pre-Rollo: {PRE_ROLL_BUFFER_SIZE} frames.")

        frame_counter = 0
        delay_por_frame = 1.0 / source_fps # "Freno" para simular FPS reales
        last_known_probs = np.array([0.0] * len(config.CLASSES))

        # --- 2. Bucle Principal del Worker ---
        while True:
            # Guardar tiempo de inicio para el control de FPS
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
            # Procesamos todos los mensajes en la cola de control
            while not control_queue.empty():
                try:
                    command = control_queue.get_nowait()
                    
                    # 1. Comprobar si es un array de probabilidades
                    if isinstance(command, np.ndarray):
                        last_known_probs = command
                    
                    # 2. Comprobar si es un comando de INICIO
                    elif command == "START_RECORDING" and current_recorder is None:
                        print(f"[Worker-{camera_id}] Recibida orden: START_RECORDING")
                        current_recorder = EventRecorder(
                            camera_id=camera_id,
                            pre_roll_frames=list(pre_roll_buffer),
                            source_fps=source_fps
                        )
                        # Iniciar el hilo de grabación en segundo plano
                        current_recorder.start()
                    
                    # 3. Comprobar si es un comando de PARADA
                    elif command == "STOP_RECORDING" and current_recorder is not None:
                        print(f"[Worker-{camera_id}] Recibida orden: STOP_RECORDING")
                        current_recorder.close()
                        current_recorder = None
                
                except Empty:
                    break # La cola está vacía
            
            # Si el grabador está activo, pasarle el frame y las últimas probs
            if current_recorder is not None:
                # Esta llamada es (casi) instantánea (solo un 'queue.put()')
                current_recorder.add_frame(frame, last_known_probs) 

            # 2d. Lógica de Inferencia (Ventana deslizante)
            if (len(inference_buffer) == INFERENCE_BUFFER_SIZE and 
                frame_counter % config.STRIDE == 0):
                
                try:
                    # Pre-procesar (Normalizar FPS, Resize, Crop, etc.)
                    tensor = preprocess_clip(list(inference_buffer))
                    
                    # Validar que el tensor no esté corrupto (NaN o Inf)
                    if not np.isfinite(tensor).all():
                        print(f"[Worker-{camera_id}] ADVERTENCIA: Tensor corrupto (NaN/Inf) detectado. Omitiendo este clip.")
                    else:
                        # Enviar a la cola de la GPU (SOLO SI ES VÁLIDO)
                        inference_queue.put((camera_id, tensor))
                
                except Exception as e:
                    # Atrapa errores de 'preprocess_clip' (ej. videos corruptos)
                    print(f"[Worker-{camera_id}] Error al pre-procesar o validar clip: {e}")

            # 2e. Controlar los FPS
            # Esperamos el tiempo restante para mantener los FPS de la fuente
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
            current_recorder.close() # 'close()' ahora detiene y 'join' el hilo
        if stream_reader is not None:
            stream_reader.release()
        print(f"[Worker-{camera_id}] Proceso terminado.")