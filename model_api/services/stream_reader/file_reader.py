import cv2
import os
import sys
import numpy as np
from typing import List, Union

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 3 niveles: .../stream_reader -> .../services -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(model_api_root)

try:
    # Usamos importación relativa (el punto) para 'base_reader'
    from .base_reader import BaseReader
except ImportError:
    # Fallback si la importación relativa falla (ej. al ejecutar como script)
    from services.stream_reader.base_reader import BaseReader

class FileReader(BaseReader):
    # Implementación de BaseReader para leer desde una LISTA de archivos de video.
    # Reproduce los videos en secuencia y vuelve al inicio de la lista (looping).
    
    def __init__(self, source: Union[str, List[str]]):
        # Constructor que acepta una sola ruta (str) o una lista de rutas (List[str])
        
        if isinstance(source, str):
            # Retrocompatibilidad: si es un solo string, lo convierte en lista
            self.video_paths = [source]
        elif isinstance(source, list):
            self.video_paths = source
        else:
            raise TypeError(f"FileReader 'source' debe ser str o List[str], no {type(source)}")

        if not self.video_paths:
            raise ValueError("FileReader 'source' no puede ser una lista vacía.")

        self.current_video_index = 0
        self.cap: Union[cv2.VideoCapture, None] = None # Se inicializará con _open_video
        self.source_fps: float = 30.0 # Valor por defecto
        
        # Abrir el primer video de la lista
        self._open_video(self.video_paths[self.current_video_index])

    def _open_video(self, file_path: str) -> bool:
        # Método helper para abrir un nuevo archivo de video
        self.current_file_path = file_path
        self.cap = cv2.VideoCapture(self.current_file_path)
        
        if not self.cap.isOpened():
            print(f"[FileReader] ADVERTENCIA: No se pudo abrir el video: {file_path}. Omitiendo.")
            return False
            
        # Obtenemos los FPS solo del primer video
        # Asumimos que todos los videos de la lista tienen los mismos FPS
        if self.current_video_index == 0:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.source_fps = fps
        
        print(f"[FileReader] Video '{os.path.basename(file_path)}' abierto. (Fuente FPS: {self.source_fps:.2f})")
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        # Lee el siguiente frame. Si el video actual termina,
        # carga el siguiente video en la lista.
        
        if not self.cap or not self.cap.isOpened():
            # Si el video anterior falló al abrir, intenta cargar el siguiente
            return self._get_next_video()

        ret, frame = self.cap.read()
        
        if not ret:
            # El video actual terminó, pasamos al siguiente
            print(f"[FileReader] Video '{os.path.basename(self.current_file_path)}' terminado.")
            return self._get_next_video()
            
        return ret, frame

    def _get_next_video(self) -> tuple[bool, np.ndarray | None]:
        # Libera el video actual, avanza al siguiente y lee el primer frame
        
        # Liberar el video anterior
        if self.cap:
            self.cap.release()
            
        # Avanzar al siguiente video
        self.current_video_index += 1
        
        # Si llegamos al final de la lista, volver al inicio (looping)
        if self.current_video_index >= len(self.video_paths):
            print("[FileReader] Lista de videos completada. Reiniciando (Looping)...")
            self.current_video_index = 0
        
        # Abrir el nuevo video
        new_path = self.video_paths[self.current_video_index]
        if not self._open_video(new_path):
            # Si este video falla al abrir, intentar recursivamente con el siguiente
            return self._get_next_video()
        
        # Leer el primer frame del nuevo video
        return self.cap.read()

    def get_fps(self) -> float:
        # Retorna los FPS del primer video de la lista.
        # El pipeline asume que todos los videos tienen los mismos FPS.
        return self.source_fps

    def release(self):
        # Cierra el archivo de video actual
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print(f"[FileReader] Lector liberado.")