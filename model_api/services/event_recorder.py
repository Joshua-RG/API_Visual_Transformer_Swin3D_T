import cv2
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
import threading
import queue

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 2 niveles: .../services -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    from config import config
except ImportError as e:
    print(f"Error fatal en 'event_recorder.py': No se pudo importar 'config'. {e}")
    sys.exit(1)


class EventRecorder(threading.Thread):
    # Esta clase se ejecuta en un HILO (thread) separado para no bloquear al 'camera_worker'.
    # Implementa su propio control de FPS (sleep) para no saturar cv2.VideoWriter.
    
    def __init__(self, camera_id: str, pre_roll_frames: list, source_fps: float):
        # Inicializa el hilo grabador
        super().__init__(daemon=True)
        
        self.camera_id = camera_id
        self.is_open = False
        self.frame_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Lógica de control de FPS para este hilo
        self.source_fps = source_fps
        self.delay_por_frame = 1.0 / self.source_fps

        # Asegurarse de que los directorios de guardado existan
        os.makedirs(config.SAVE_CLIP_PATH, exist_ok=True)
        os.makedirs(config.SAVE_LOG_PATH, exist_ok=True)

        # Generar nombres de archivo únicos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_basename = f"{camera_id}_{timestamp}"
        
        self.video_path = os.path.join(config.SAVE_CLIP_PATH, f"{file_basename}.mp4")
        self.log_path = os.path.join(config.SAVE_LOG_PATH, f"{file_basename}.json")

        try:
            if not pre_roll_frames:
                print(f"[Recorder] ERROR: No se puede iniciar el grabador sin frames de pre-rollo.")
                return
            
            # Obtener dimensiones del primer frame
            h, w, _ = pre_roll_frames[0].shape
            
            # Definir el codec (mp4v para .mp4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.source_fps, (w, h))
            
            if not self.video_writer.isOpened():
                raise IOError(f"No se pudo abrir VideoWriter en: {self.video_path}")

            self.is_open = True
            self.start_time = time.time()
            self.logs = []
            
            # Escribir el búfer de pre-rollo inmediatamente
            print(f"[Recorder] Grabación iniciada: {file_basename}.mp4")
            for frame in pre_roll_frames:
                self.video_writer.write(frame)
            
            print(f"[Recorder] {len(pre_roll_frames)} frames de pre-rollo guardados.")

        except Exception as e:
            print(f"[Recorder] CRÍTICO: Error al inicializar: {e}")
            self.is_open = False

    def add_frame(self, frame: np.ndarray, probabilities: np.ndarray):
        # Añade un frame y sus probabilidades a la cola de grabación.
        # Esta operación es (casi) instantánea y no bloquea al 'camera_worker'.
        if self.is_open:
            self.frame_queue.put((frame, probabilities))

    def run(self):
        # Este es el bucle que se ejecuta en el hilo de fondo.
        # Saca frames de la cola y los escribe en el disco, respetando los FPS.
        print(f"[Recorder Thread-{self.camera_id}] Hilo de grabación iniciado (sincronizado a {self.source_fps:.2f} FPS).")
        
        while not self.stop_event.is_set():
            # Iniciamos el cronómetro del bucle del hilo
            loop_start_time = time.time()
            
            try:
                # Espera por un nuevo frame (con timeout de 0.1s)
                # El timeout es para que el bucle 'while' pueda comprobar el 'stop_event'
                frame, probabilities = self.frame_queue.get(timeout=0.1)
                
                # Escribir el frame de video (la operación lenta)
                self.video_writer.write(frame)
                
                # Guardar el log de predicción
                log_entry = {
                    "timestamp_ms": int((time.time() - self.start_time) * 1000),
                    "probabilities": {
                        config.CLASSES[0]: float(probabilities[0]),
                        config.CLASSES[1]: float(probabilities[1]),
                        config.CLASSES[2]: float(probabilities[2]),
                    }
                }
                self.logs.append(log_entry)
                
            except queue.Empty:
                # No llegaron frames en 0.1s. No hacemos 'sleep' y
                # volvemos a comprobar la cola y el 'stop_event'.
                continue
            except Exception as e:
                # Captura un error de escritura (ej. disco lleno) sin matar el hilo
                print(f"[Recorder Thread-{self.camera_id}] Error al escribir frame: {e}")
            
            # Aplicar el "freno" (sleep)
            # Esperamos el resto del tiempo asignado (ej. 33.3ms)
            time_elapsed = time.time() - loop_start_time
            sleep_time = self.delay_por_frame - time_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print(f"[Recorder Thread-{self.camera_id}] Hilo de grabación detenido.")

    def close(self):
        # Avisa al hilo 'run' que debe parar y espera a que termine.
        if not self.is_open:
            return
            
        print(f"[Recorder] Recibida orden de cierre para: {os.path.basename(self.video_path)}...")
        
        try:
            # 1. Avisar al hilo 'run' que debe parar
            self.is_open = False
            self.stop_event.set()
            
            # 2. Esperar a que el hilo 'run' termine (procese lo que queda en la cola)
            self.join() 
            
            # 3. Ahora que el hilo está muerto, cerrar el archivo de video
            self.video_writer.release()
            
            # 4. Crear y guardar el archivo .json
            summary = {
                "camera_id": self.camera_id,
                "event_start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "event_end_time": datetime.now().isoformat(),
                "video_file": os.path.basename(self.video_path),
                "log_file": os.path.basename(self.log_path),
                "total_logs": len(self.logs),
                "logs": self.logs
            }
            
            with open(self.log_path, 'w') as f:
                json.dump(summary, f, indent=4)
                
            print(f"[Recorder] Grabación finalizada. Video guardado en: {self.video_path}")
            print(f"[Recorder] Log de evento guardado en: {self.log_path}")

        except Exception as e:
            print(f"[Recorder] Error al cerrar: {e}")