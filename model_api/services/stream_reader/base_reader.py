from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseReader(ABC):
    # Define la interfaz abstracta (el "contrato") para todos los lectores de video.
    # Cualquier clase que herede de BaseReader debe implementar estos métodos.
    # Esto permite que el 'camera_worker' los use de forma intercambiable.
    
    @abstractmethod
    def __init__(self, source: Any):
        # Constructor que inicializa la fuente de video (ruta de archivo, lista o URL)
        pass

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        # Lee el siguiente frame del video.
        # Retorna (True, frame) si la lectura fue exitosa.
        # Retorna (False, None) si el video terminó o hubo un error.
        pass

    @abstractmethod
    def get_fps(self) -> float:
        # Retorna los FPS (frames por segundo) de la fuente de video.
        pass

    @abstractmethod
    def release(self):
        # Libera los recursos (cierra el archivo o la conexión de red).
        pass