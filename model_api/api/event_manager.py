import asyncio
import json
import sys
import os
import numpy as np
from multiprocessing import Queue
from typing import Dict, Union

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 1 nivel: .../api -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    from config import config
    from api.connection_manager import ConnectionManager
except ImportError as e:
    print(f"Error fatal en 'event_manager.py': No se pudo importar un módulo. {e}")
    sys.exit(1)

# Diccionario global para mantener el estado de cada cámara (ej. "IDLE", "RECORDING")
camera_states: Dict[str, str] = {}

async def event_manager_task(
    manager: ConnectionManager,
    results_queue: Queue,
    control_queues: Dict[str, Queue]
):
    # Esta es la tarea de fondo ("cerebro lógico") de la API.
    # Se ejecuta en un bucle infinito dentro del proceso de la API.
    
    print("[EventManager] Tarea de fondo iniciada. Esperando resultados de la GPU...")
    
    while True:
        try:
            # --- 1. Leer Resultados de la GPU ---
            
            # Usamos 'asyncio.to_thread' para ejecutar el .get() bloqueante
            # en un hilo separado, sin congelar el bucle de eventos de la API.
            camera_id, probabilities = await asyncio.to_thread(results_queue.get)

            # --- 2. Alerta WebSocket (al Frontend) ---
            
            # Formatear el mensaje JSON para el frontend (React)
            probs_dict = {
                config.CLASSES[i]: float(probabilities[i]) 
                for i in range(len(config.CLASSES))
            }
            message = json.dumps({
                "camera_id": camera_id, 
                "probabilities": probs_dict
            })
            
            # Enviar a todos los clientes suscritos a este WebSocket
            await manager.broadcast(camera_id, message)

            # --- 3. Lógica de Grabación (al Camera Worker) ---
            
            # Comprobar si alguna probabilidad supera el umbral de alerta
            is_violence_detected = any(p > config.ALERT_THRESHOLD for p in probabilities)
            
            # Obtener el estado actual de la cámara (default: "IDLE")
            current_state = camera_states.get(camera_id, "IDLE")
            
            control_queue = control_queues.get(camera_id)
            if not control_queue:
                # Si 'run_app.py' no registró una cola para esta cámara, no podemos controlarla.
                print(f"[EventManager] ERROR: No se encontró 'control_queue' para {camera_id}.")
                continue

            # --- Máquina de Estados de Grabación ---
            
            if is_violence_detected:
                if current_state == "IDLE":
                    # --- INICIAR GRABACIÓN ---
                    print(f"[EventManager] ¡Evento detectado en {camera_id}! Enviando orden START_RECORDING.")
                    control_queue.put("START_RECORDING")
                    camera_states[camera_id] = "RECORDING" # Actualizar estado
                
                # Enviar las probabilidades al worker para que las guarde en el JSON
                control_queue.put(probabilities)

            elif not is_violence_detected and current_state == "RECORDING":
                # --- DETENER GRABACIÓN ---
                print(f"[EventManager] Evento terminado en {camera_id}. Enviando orden STOP_RECORDING.")
                control_queue.put("STOP_RECORDING")
                camera_states[camera_id] = "IDLE" # Actualizar estado

        except (KeyboardInterrupt, SystemExit):
            print("[EventManager] Deteniendo tarea de fondo...")
            break
        except Exception as e:
            print(f"[EventManager] ERROR en el bucle: {e}")
            # Pausa breve para no inundar los logs si hay un error persistente
            await asyncio.sleep(1)