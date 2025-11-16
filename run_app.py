import multiprocessing
import uvicorn
import os
import sys
import time
import glob   
import random 
import numpy as np
from typing import List, Dict, Any

# --- 1. Importar los Componentes del Backend ---
# (Importamos los módulos y funciones que este orquestador necesita iniciar)
try:
    from model_api.services.inference_service import run_inference_service
    from model_api.services.camera_worker import run_camera_worker
    from model_api.api import main as api_main  
    from model_api.config import config        
except ImportError as e:
    print(f"Error fatal: No se pudo importar un módulo desde 'model_api'. {e}")
    print("Asegúrate de que 'run_app.py' esté en la raíz del proyecto (junto a 'model_api').")
    sys.exit(1)

def get_video_files(video_dir: str, num_cameras: int = 4) -> List[List[str]]:
    # Escanea el directorio de videos, los baraja y los divide en N listas.
    print(f"Escaneando videos de prueba en: {video_dir}...")
    
    # Buscar todos los archivos .avi y .mp4
    video_paths = glob.glob(os.path.join(video_dir, "*.avi"))
    video_paths.extend(glob.glob(os.path.join(video_dir, "*.mp4")))
    
    if not video_paths:
        print(f"ADVERTENCIA: No se encontraron videos en {video_dir}.")
        print("Asegúrate de que tus videos de prueba estén en 'model_api/data/videos_prueba/'.")
        return []

    print(f"Se encontraron {len(video_paths)} videos de prueba.")
    
    # Barajarlos para que la distribución sea aleatoria
    random.shuffle(video_paths)
    
    # Dividir la lista de videos en N fragmentos (uno por cámara)
    video_chunks = np.array_split(video_paths, num_cameras)
    
    # Convertir los fragmentos de numpy a listas de Python
    video_lists = [list(chunk) for chunk in video_chunks]
    
    print(f"Videos divididos en {len(video_lists)} listas (cámaras).")
    return video_lists

def main(
    cameras_to_run: List[Dict[str, Any]], 
    inference_queue: multiprocessing.Queue, 
    results_queue: multiprocessing.Queue, 
    control_queues: Dict[str, multiprocessing.Queue]
):
    # Función principal para orquestar todos los servicios.
    # Recibe la configuración y las colas desde el bloque __main__.
    
    print("--- Iniciando UrbanSentinel Backend ---")
    worker_processes = []

    try:
        # --- 1. Inyectar las Colas en el Módulo de la API ---
        # Le damos al módulo 'api_main' acceso a las colas
        # ANTES de que uvicorn lo inicie.
        api_main.inference_queue = inference_queue
        api_main.results_queue = results_queue
        api_main.control_queues = control_queues
        print("Colas inyectadas en el módulo API.")

        # --- 2. Iniciar el Servicio de Inferencia (GPU) ---
        print("Iniciando servicio de inferencia (Proceso GPU)...")
        inference_process = multiprocessing.Process(
            target=run_inference_service,
            args=(inference_queue, results_queue),
            daemon=True # El proceso morirá si el script principal muere
        )
        inference_process.start()

        # --- 3. Iniciar los Workers de Cámara (CPU) ---
        for cam in cameras_to_run:
            print(f"Iniciando worker para cámara: {cam['id']}...")
            worker = multiprocessing.Process(
                target=run_camera_worker,
                
                # --- INICIO DE LA MODIFICACIÓN ---
                # Le pasamos la 'results_queue' al worker para que pueda
                # hacer el "bypass" y enviar resultados [0,0,0]
                # directamente al EventManager cuando no haya personas.
                args=(
                    cam["id"],
                    cam["type"],
                    cam["path"], # Le pasamos la LISTA de videos
                    inference_queue,
                    control_queues[cam["id"]],
                    results_queue  # <-- ¡AQUÍ ESTÁ EL AÑADIDO!
                ),
                # --- FIN DE LA MODIFICACIÓN ---
                
                daemon=True
            )
            worker.start()
            worker_processes.append(worker)

        print(f"{len(worker_processes)} workers de cámara iniciados.")
        
        # --- 4. Iniciar la API (Proceso Principal) ---
        # Uvicorn se ejecuta en el hilo principal y bloquea el script aquí.
        print("\n--- Iniciando API (FastAPI) en http://127.0.0.1:8000 ---")
        print("Puedes iniciar 'test_websocket.py' en otra terminal para ver los resultados.")
        
        uvicorn.run(
            "model_api.api.main:app",
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False # 'reload=True' no funciona bien con multiprocessing
        )

    except KeyboardInterrupt:
        print("\nDeteniendo servicios...")
    finally:
        # --- 5. Limpieza ---
        print("Enviando señal de terminación a los procesos...")
        if 'inference_process' in locals() and inference_process.is_alive():
            inference_process.terminate()
        for worker in worker_processes:
            if worker.is_alive():
                worker.terminate()
        print("Servicios detenidos. Saliendo.")


#if __name__ == "__main__":
#    
#    # 1. Establecer el método de 'spawn' PRIMERO.
#    # Esto es crucial para CUDA y Windows para evitar 'deadlocks'.
#    multiprocessing.set_start_method("spawn")
#
#    # 2. Configurar la prueba
#    # Esta lógica solo se ejecuta 1 vez en el proceso principal.
#    print("--- Configurando la prueba de 4 cámaras ---")
#    
#    CAMERA_IDS = ["cam_01", "cam_02", "cam_03", "cam_04"]
#    VIDEO_DIR_PATH = os.path.join(config.BASE_DIR, "data", "videos_prueba")
#
#    # Escanear y dividir los 554 videos de prueba
#    video_lists_for_cameras = get_video_files(VIDEO_DIR_PATH, num_cameras=len(CAMERA_IDS))
#
#    # Crear la configuración de las cámaras
#    CAMERAS_TO_RUN = []
#    for i, camera_id in enumerate(CAMERA_IDS):
#        if i < len(video_lists_for_cameras) and len(video_lists_for_cameras[i]) > 0:
#            cam_config = {
#                "id": camera_id,
#                "type": "file", # Usamos el FileReader
#                "path": video_lists_for_cameras[i]
#            }
#            CAMERAS_TO_RUN.append(cam_config)
#            print(f"Configurada '{camera_id}' con {len(cam_config['path'])} videos únicos.")
#        else:
#            print(f"ADVERTENCIA: No se asignaron videos a '{camera_id}'.")
#
#    # 3. Crear las Colas de Comunicación
#    inference_queue = multiprocessing.Queue()
#    results_queue = multiprocessing.Queue()
#    control_queues = {cam["id"]: multiprocessing.Queue() for cam in CAMERAS_TO_RUN}
#    print("Colas de comunicación creadas.")
#    
#    # 4. Iniciar la función 'main' con la configuración lista
#    main(CAMERAS_TO_RUN, inference_queue, results_queue, control_queues)

# ... (todo tu código anterior: imports, get_video_files, main) ...

if __name__ == "__main__":
    
    # 1. Establecer el método de 'spawn' PRIMERO.
    multiprocessing.set_start_method("spawn")

    # --- 2. Configuración de la Prueba (MODIFICADO) ---
    print("--- Configurando la prueba de 1 cámara (video sin violencia) ---")
    
    # --- ¡CAMBIA ESTA LÍNEA! ---
    # (Asegúrate de que este sea el nombre exacto de tu video de prueba)
    VIDEO_DE_PRUEBA_PATH = os.path.join(
        config.BASE_DIR, "data", "videos_prueba", "vid_107.avi"
    )

    if not os.path.exists(VIDEO_DE_PRUEBA_PATH):
        print(f"ERROR: No se encuentra el video de prueba en: {VIDEO_DE_PRUEBA_PATH}")
        print("Por favor, edita la variable 'VIDEO_DE_PRUEBA_PATH' en run_app.py")
        sys.exit(1)

    # Desactivamos la lógica de 4 cámaras
    # CAMERA_IDS = ["cam_01", "cam_02", "cam_03", "cam_04"]
    # VIDEO_DIR_PATH = os.path.join(config.BASE_DIR, "data", "videos_prueba")
    # video_lists_for_cameras = get_video_files(VIDEO_DIR_PATH, num_cameras=len(CAMERA_IDS))

    # Creamos la configuración de las cámaras manualmente
    CAMERAS_TO_RUN = [
        {
            "id": "cam_01",
            "type": "file", # Usamos el FileReader
            "path": [VIDEO_DE_PRUEBA_PATH] # Le pasamos una lista con un solo video
        }
    ]
    
    print(f"Configurada 'cam_01' con el video: {VIDEO_DE_PRUEBA_PATH}")

    # --- 3. Crear las Colas de Comunicación ---
    # (Esta lógica es idéntica, pero solo creará una control_queue)
    inference_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()
    control_queues = {cam["id"]: multiprocessing.Queue() for cam in CAMERAS_TO_RUN}
    print("Colas de comunicación creadas.")
    
    # 4. Iniciar la función 'main' con la configuración lista
    main(CAMERAS_TO_RUN, inference_queue, results_queue, control_queues)