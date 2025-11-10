import asyncio
import sys
import os
import multiprocessing as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Union
from contextlib import asynccontextmanager

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 1 nivel: .../api -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    from api.event_manager import event_manager_task
    from api.connection_manager import ConnectionManager
except ImportError as e:
    print(f"Error fatal en 'main.py': No se pudo importar 'event_manager' o 'connection_manager'. {e}")
    sys.exit(1)


# --- Definición de las Colas Globales ---
# Estas variables son 'Platzholders' (marcadores de posición).
# El script 'run_app.py' las llenará ("inyectará") antes de iniciar el servidor.
inference_queue: Union[mp.Queue, None] = None
results_queue: Union[mp.Queue, None] = None
control_queues: Dict[str, mp.Queue] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Gestiona los eventos de arranque y apagado de la aplicación (reemplaza @app.on_event)
    
    print("[API] Servidor FastAPI iniciando...")
    
    # Comprobación de seguridad
    if results_queue is None:
        print("[API] CRÍTICO: Las colas no fueron inyectadas por run_app.py. Saliendo.")
        sys.exit(1)
        
    print("[API] Iniciando tarea de fondo 'event_manager'...")
    # Iniciar el 'cerebro' y pasarle acceso al gestor y las colas inyectadas
    asyncio.create_task(event_manager_task(
        manager=manager,
        results_queue=results_queue,
        control_queues=control_queues
    ))
    
    # Esto es lo que se ejecuta mientras la app está viva
    yield
    
    # Código de apagado
    print("[API] Servidor FastAPI apagándose.")


# --- Creación de la App FastAPI ---
app = FastAPI(
    title="UrbanSentinel API",
    description="API para la detección de violencia en tiempo real.",
    lifespan=lifespan  
)

# Instancia única del gestor de conexiones
manager = ConnectionManager()

# --- Endpoints ---

@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    # Mantiene una conexión persistente con un cliente frontend
    
    await manager.connect(websocket, camera_id)
    try:
        while True:
            # Espera a que el cliente envíe un mensaje (ej. 'ping')
            # Si el cliente se desconecta, esto lanzará WebSocketDisconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_id)

@app.get("/")
def read_root():
    # Endpoint simple para verificar que la API está viva (Health Check)
    return {"message": "UrbanSentinel API en funcionamiento."}