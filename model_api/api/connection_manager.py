import asyncio
from fastapi import WebSocket
from typing import Dict, List

class ConnectionManager:
    # Esta clase gestiona todas las conexiones WebSocket activas (frontends).
    # Mapea un camera_id a una lista de WebSockets suscritos.
    
    def __init__(self):
        # El diccionario de conexiones activas
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        # Acepta y registra una nueva conexión de un cliente
        await websocket.accept()
        
        # Si es la primera conexión para esta cámara, crea la lista
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
            
        self.active_connections[camera_id].append(websocket)
        print(f"[API] Frontend conectado a WebSocket para: {camera_id}")

    def disconnect(self, websocket: WebSocket, camera_id: str):
        # Elimina una conexión de un cliente que se ha desconectado
        if camera_id in self.active_connections:
            self.active_connections[camera_id].remove(websocket)
        print(f"[API] Frontend desconectado de: {camera_id}")

    async def broadcast(self, camera_id: str, message: str):
        # Envía un mensaje (JSON) a todos los clientes que están viendo esa cámara
        if camera_id in self.active_connections:
            for connection in self.active_connections[camera_id]:
                await connection.send_text(message)