import cv2
import numpy as np
import sys
import os

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 2 niveles: .../processing -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    # Importamos el módulo (archivo) config.py
    from config import config
except ImportError as e:
    print(f"Error fatal en 'video_processor.py': No se pudo importar 'config'. {e}")
    sys.exit(1)


def _resize_maintaining_aspect_ratio(frame: np.ndarray, target_size: int) -> np.ndarray:
    # Replica T.Resize(size=256). Escala el lado más corto a 'target_size'
    h, w = frame.shape[:2]
    
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
        
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _center_crop(frame: np.ndarray, crop_size: int) -> np.ndarray:
    # Replica T.CenterCrop(size=224). Extrae un recorte central.
    h, w = frame.shape[:2]
    
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    
    return frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

def preprocess_clip(frames: list) -> np.ndarray:
    # Preprocesa una lista de N frames (del búfer) para que coincida 
    # con la 'val_transform' del notebook de entrenamiento.
    # Aplica sub-muestreo para normalizar a TARGET_FPS (30).
    #
    # Args:
    #     frames (list): Lista de frames de video (de OpenCV).
    # Returns:
    #     np.ndarray: Un tensor con forma (3, 32, 224, 224), listo para la GPU.
    
    num_frames_in = len(frames)
    
    # 1. Calcular los índices de los 32 frames que queremos (Normalización de FPS)
    indices = np.linspace(
        0,                 
        num_frames_in - 1, 
        num=config.CLIP_LEN
    ).astype(int) 
    
    # 2. Seleccionar solo esos frames
    sampled_frames = [frames[i] for i in indices]
    
    processed_frames = []
    
    for frame in sampled_frames:
        # 3. Convertir de BGR (OpenCV) a RGB 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 4. Replicar T.Resize(size=256)
        frame_resized = _resize_maintaining_aspect_ratio(
            frame_rgb, 
            config.INPUT_RESIZE
        )
        
        # 5. Replicar T.CenterCrop(size=224)
        frame_cropped = _center_crop(
            frame_resized, 
            config.INPUT_CROP_SIZE
        )
        
        processed_frames.append(frame_cropped)
    
    # 6. Apilar todos los frames en un solo array (T, H, W, C)
    clip_array = np.stack(processed_frames, axis=0)

    # 7. Escalar píxeles a [0, 1]
    clip_array = clip_array.astype(np.float32) / 255.0
    
    # 8. Normalizar con la media y std de ImageNet
    clip_array = (clip_array - config.NORM_MEAN) / config.NORM_STD
    
    # 9. Permutar dimensiones a (C, T, H, W) como espera el modelo
    clip_array = np.transpose(clip_array, (3, 0, 1, 2))
    
    # 10. Devolver el tensor SIN la dimensión de lote (Batch)
    return clip_array.astype(np.float32)