import asyncio
import websockets
import json
import sys
from typing import List

# Definir las 4 cámaras que queremos escuchar
CAMERAS_TO_TEST = ["cam_01", "cam_02", "cam_03", "cam_04"]

async def listen_to_websocket(uri: str, camera_id: str):
    # Se conecta a un endpoint WebSocket y muestra los mensajes que recibe.
    
    print(f"--- [Cliente para {camera_id}] Intentando conectar a: {uri} ---")
    
    try:
        # Conectarse al servidor
        async with websockets.connect(uri) as websocket:
            print(f"--- [Cliente para {camera_id}] ¡Conexión exitosa! Esperando predicciones... ---")
            
            # Bucle infinito para escuchar mensajes
            while True:
                try:
                    # Espera a recibir un mensaje del servidor
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Imprimir el resultado de forma bonita
                    print("\n--- ¡Predicción Recibida! ---")
                    print(f"  Cámara: {data.get('camera_id')}")
                    
                    probs = data.get('probabilities', {})
                    for class_name, prob in probs.items():
                        # Imprimir solo si la prob es alta para no saturar
                        if prob > 0.5:
                            print(f"  *** {class_name}: {prob:.1%} ***")
                        else:
                            print(f"  {class_name}: {prob:.1%}")
                
                except websockets.exceptions.ConnectionClosed:
                    print(f"--- [Cliente para {camera_id}] Conexión cerrada por el servidor. ---")
                    break
                except Exception as e:
                    print(f"--- [Cliente para {camera_id}] Error al procesar mensaje: {e} ---")

    except Exception as e:
        print(f"--- [Cliente para {camera_id}] No se pudo conectar al servidor: {e} ---")
        print("Asegúrate de que 'run_app.py' se esté ejecutando.")

async def main(camera_ids: List[str]):
    # Lanza tareas de escucha para todas las cámaras en paralelo.
    
    tasks = []
    for cam_id in camera_ids:
        uri = f"ws://127.0.0.1:8000/ws/{cam_id}"
        # Creamos una tarea para cada 'listen_to_websocket'
        tasks.append(listen_to_websocket(uri, cam_id))
    
    # 'asyncio.gather' ejecuta todas las tareas en paralelo
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    
    cameras = CAMERAS_TO_TEST
    
    # Permite anular las cámaras desde la línea de comandos
    # ej: python test_websocket.py cam_01 cam_03
    if len(sys.argv) > 1:
        cameras = sys.argv[1:]
    
    try:
        asyncio.run(main(cameras))
    except KeyboardInterrupt:
        print("\nCerrando clientes.")