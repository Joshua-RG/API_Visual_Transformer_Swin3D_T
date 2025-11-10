# 🚀 Documentación del Pipeline de Detección de Violencia

Este documento detalla la arquitectura y el flujo de datos del *pipeline* de detección de violencia. Sirve como una guía técnica para entender el proyecto y como un manual para la implementación en producción.

## 1. 🏛️ Estructura de Carpetas

El proyecto está dividido en dos partes principales: un paquete de Python (`model_api`) que contiene todo el código fuente, y un conjunto de scripts en la raíz (`run_app.py`, `test_websocket.py`) que ejecutan el *pipeline*.

```
PROYECTO_URBANSENTINEL/
├── .gitignore
├── README.md
├── run_app.py
├── test_websocket.py
├── venv_api/
└── model_api/
    ├── __pycache__/
    ├── api/
    │   ├── __pycache__/
    │   ├── connection_manager.py
    │   ├── event_manager.py
    │   └── main.py
    ├── config/
    │   ├── __pycache__/
    │   └── config.py
    ├── data/
    │   ├── clips_guardados/
    │   ├── logs_eventos/
    │   └── videos_prueba/
    ├── onnx_model/
    │   ├── __pycache__/
    │   ├── onnx_detector.py
    │   ├── swin3d_t.onnx
    │   └── swin3d_t.onnx.data
    ├── processing/
    │   ├── __pycache__/
    │   └── video_processor.py
    └── services/
        ├── __pycache__/
        ├── stream_reader/
        │   ├── __pycache__/
        │   ├── base_reader.py
        │   └── file_reader.py
        ├── camera_worker.py
        ├── event_recorder.py
        └── inference_service.py

```

## 2. 🌊 Flujo de Datos del Pipeline (Paso a Paso)

El *pipeline* está diseñado en 3 procesos principales para máxima eficiencia:
1.  **Ingesta (CPU):** Un proceso `camera_worker.py` por cada cámara.
2.  **Inferencia (GPU):** Un único proceso `inference_service.py` que sirve a todas las cámaras.
3.  **Lógica (API):** Un proceso `api/main.py` que maneja las decisiones y la comunicación.

**El flujo de una predicción es el siguiente:**

1.  **`run_app.py`** (El Orquestador) se ejecuta. Inicia los procesos de GPU y API. Luego, lee su lista de cámaras (4 en nuestra prueba) e inicia 4 procesos `camera_worker.py`.
2.  **`camera_worker.py`** (Proceso CPU)
    * Inicia un `file_reader.py` que comienza a leer una lista de videos de prueba.
    * Mantiene dos búferes de frames: uno para inferencia (`inference_buffer`) y uno para pre-grabación (`pre_roll_buffer`).
    * Gracias a un `time.sleep()`, el *worker* se frena a sí mismo para simular 30 FPS.
    * Cada 16 frames (`STRIDE`), el *worker* llama a `preprocess_clip()`.
3.  **`video_processor.py`** (Lógica de CPU)
    * Recibe los frames (ej. 32 frames de un video de 30 FPS).
    * Realiza el preprocesamiento: `resize`, `crop`, y `normalize`.
    * Devuelve un tensor *único* con forma `(3, 32, 224, 224)`.
4.  **`camera_worker.py`** (de vuelta)
    * Recibe el tensor y lo valida con `np.isfinite()` para asegurarse de que no esté corrupto (evitando crasheos de GPU).
    * Si es válido, pone el tensor y su ID (ej. `("cam_01", tensor)`) en la `inference_queue`.
5.  **`inference_service.py`** (Proceso GPU)
    * Está escuchando la `inference_queue`.
    * Toma el clip `("cam_01", tensor)`. **(Procesa 1 a la vez)**.
    * Lo expande a un lote de 1: `(1, 3, 32, 224, 224)`.
    * Ejecuta `detector.predict_batch()`.
    * (En la primera llamada) `onnx_detector.py` carga el modelo en la GPU ("Lazy Loading").
    * Pone el resultado (ej. `("cam_01", [0.9, 0.1, 0.1])`) en la `results_queue`.
6.  **`event_manager.py`** (Proceso API)
    * Está escuchando la `results_queue`.
    * Recibe `("cam_01", [0.9, 0.1, 0.1])`.
    * **Transmite (Broadcast)** la predicción por WebSocket a todos los clientes que escuchen a `cam_01`.
    * **Toma una Decisión:** Compara `0.9` con el `ALERT_THRESHOLD` (0.7).
    * Como es `> 0.7` y el estado de la cámara era `IDLE`, pone dos mensajes en la `control_queue` de `cam_01`:
        1.  El *string* `"START_RECORDING"`.
        2.  El *array* `[0.9, 0.1, 0.1]`.
7.  **`camera_worker.py`** (de vuelta)
    * Detecta los mensajes en su `control_queue`.
    * Al recibir `"START_RECORDING"`, crea una instancia de `EventRecorder` (un hilo) y le pasa el búfer de pre-grabación (los 5 segundos *antes* de la detección).
    * Llama a `recorder.start()`, iniciando el hilo de grabación.
    * Al recibir `[0.9, 0.1, 0.1]`, actualiza su variable `last_known_probs`.
8.  **`event_recorder.py`** (Hilo de Grabación)
    * El bucle principal del *worker* (que sigue a 30 FPS) ahora solo pone el frame y las `last_known_probs` en la `frame_queue` del grabador (esto es instantáneo).
    * El hilo de `EventRecorder` saca el frame de su cola, lo escribe en el disco (`cv2.VideoWriter`) y aplica su propio `time.sleep()` para mantenerse a 30 FPS, evitando así los errores de FFmpeg.

---

## 3. 📚 Guía de Archivos y Lógica

Aquí se detalla qué hace cada archivo en el proyecto.

### Grupo 1: Configuración (`/model_api/config/`)

* **`config.py`**
    * **Qué hace:** Es el "panel de control" de todo el proyecto. Contiene todas las constantes y variables mágicas en un solo lugar.
    * **Lógica Clave:** Define `CLASSES` (las etiquetas), `CLIP_LEN` (32 frames), `TARGET_FPS` (30 FPS), `STRIDE` (16 frames), `PRE_ROLL_SECONDS` (5 segundos), `ALERT_THRESHOLD` (0.7 o 70%) y la lista de `INFERENCE_PROVIDERS` (CUDA, Dml, CPU).

### Grupo 2: Módulos de IA y Procesamiento (`/model_api/onnx_model/` y `/model_api/processing/`)

* **`onnx_detector.py`**
    * **Qué hace:** Una clase "envoltorio" (wrapper) que maneja el modelo ONNX.
    * **Lógica Clave:** Usa **Lazy Loading**: no carga el modelo en `__init__`. El modelo solo se carga en la GPU (`_load_model()`) la primera vez que se llama a `predict_batch()`. Esto es crucial para evitar *deadlocks* de CUDA con `multiprocessing`. Lee `config.INFERENCE_PROVIDERS` para decidir si usar NVIDIA (CUDA), AMD (DML) o CPU.
* **`video_processor.py`**
    * **Qué hace:** Una librería de funciones puras. Su única función, `preprocess_clip()`, convierte una lista de frames de video en un tensor listo para la IA.
    * **Lógica Clave:** La lógica de normalización de FPS está aquí (`np.linspace`). Toma una lista de frames (ej. 64 frames de un video de 60 FPS) y la "muestrea" a 32 frames (`config.CLIP_LEN`), replicando la forma en que el modelo fue entrenado. Devuelve un tensor de forma `(3, 32, 224, 224)`.

### Grupo 3: Módulos de I/O (`/model_api/services/stream_reader/` y `event_recorder.py`)

* **`base_reader.py`**
    * **Qué hace:** Define la "interfaz" o "contrato" que todos los lectores de video deben seguir. Fuerza a que todos tengan los métodos `read()`, `get_fps()` y `release()`.
* **`file_reader.py`**
    * **Qué hace:** Es el lector de video que usamos para **pruebas locales**.
    * **Lógica Clave:** Acepta una **lista** de rutas de video. Reproduce el video 1, luego el video 2, etc. Cuando termina la lista, vuelve al video 1 y repite (looping), simulando un *stream* de cámara infinito.
* **`event_recorder.py`**
    * **Qué hace:** Es el grabador de video. Está diseñado para ejecutarse como un **hilo** (`threading.Thread`) separado.
    * **Lógica Clave:** Al crearse, escribe el búfer de pre-rollo. Luego, su hilo `run()` se queda en un bucle sacando frames de una `queue.Queue` y escribiéndolos en el disco. Implementa su propio `time.sleep()` para sincronizarse a 30 FPS y evitar las advertencias de FFmpeg. `close()` detiene el hilo de forma segura y guarda el `.json` final.

### Grupo 4: Los Servicios (Workers) (`/model_api/services/`)

* **`camera_worker.py`**
    * **Qué hace:** Es el "Ingestor de CPU" y el *proceso* más complejo. Se ejecuta uno por cada cámara.
    * **Lógica Clave:**
        1.  **Ingesta:** Usa un `stream_reader` (como `FileReader`) para leer frames.
        2.  **Control de FPS:** Usa `time.sleep(delay_por_frame)` para frenarse a sí mismo a los 30 FPS de la fuente.
        3.  **Procesamiento:** Mantiene el `inference_buffer` y llama a `preprocess_clip()` cada 16 frames (`STRIDE`).
        4.  **Validación:** Comprueba el tensor resultante con `np.isfinite()` para proteger a la GPU de datos corruptos.
        5.  **Control de Grabación:** Escucha la `control_queue`. Inicia/Detiene el hilo `EventRecorder` y reenvía los *arrays* de probabilidades a la cola del grabador para que se guarden en el `.json`.
* **`inference_service.py`**
    * **Qué hace:** Es el "Corazón de la GPU". Solo se ejecuta **un** proceso de este tipo en todo el sistema.
    * **Lógica Clave:** **Procesa clips de uno en uno (Batch Size = 1)**. Esta fue la corrección clave para evitar los errores de `Reshape node` del modelo ONNX. Su lógica es un bucle simple: `inference_queue.get()`, `np.expand_dims()` (para crear un lote de 1), `detector.predict_batch()`, y `results_queue.put()`.

### Grupo 5: La API (`/model_api/api/`)

* **`connection_manager.py`**
    * **Qué hace:** Una clase simple que gestiona los clientes de WebSocket. Mantiene un diccionario que mapea un `camera_id` a una lista de conexiones (navegadores) que están viendo esa cámara.
* **`event_manager.py`**
    * **Qué hace:** Es el "Cerebro Lógico" de la aplicación. Se ejecuta como una tarea de fondo (`async`) dentro de la API.
    * **Lógica Clave (Detección y Decisión):**
        1.  Aquí es donde **se detecta la violencia por primera vez** (`is_violence_detected = any(p > config.ALERT_THRESHOLD ...)`).
        2.  Transmite **todas** las predicciones (violentas o no) al *frontend* vía WebSocket.
        3.  Implementa la "máquina de estados" (`IDLE` <-> `RECORDING`).
        4.  Envía los comandos `"START_RECORDING"`, `"STOP_RECORDING"` y los *arrays* de probabilidades a la `control_queue` del *worker* correspondiente.
* **`main.py`**
    * **Qué hace:** Define la aplicación FastAPI (`app = FastAPI(...)`) y los *endpoints*.
    * **Lógica Clave:** Define el *endpoint* `/ws/{camera_id}` al que se conecta el *frontend* (React). Usa una función `lifespan` (que reemplaza al `@app.on_event("startup")` obsoleto) para iniciar la tarea de fondo `event_manager_task` cuando se enciende el servidor.

### Grupo 6: Los Lanzadores (`/`)

* **`run_app.py`**
    * **Qué hace:** Es el **único script que debes ejecutar** para iniciar todo el *backend*.
    * **Lógica Clave:**
        1.  Establece `multiprocessing.set_start_method("spawn")` (crítico para CUDA en Windows).
        2.  Crea las `multiprocessing.Queue` (colas de procesos).
        3.  Escanea los videos de prueba y los divide en 4 listas.
        4.  Inicia el `inference_service` (1 Proceso).
        5.  Inicia los 4 `camera_worker` (4 Procesos).
        6.  "Inyecta" las colas en las variables globales del módulo `api_main`.
        7.  Inicia el servidor `uvicorn` en el proceso principal, que a su vez carga `api/main.py`.
* **`test_websocket.py`**
    * **Qué hace:** Un script de prueba para simular ser el *frontend*.
    * **Lógica Clave:** Usa `asyncio.gather()` para conectarse a los 4 *endpoints* WebSocket (`cam_01` a `cam_04`) en paralelo y muestra todas las predicciones que recibe.

---

## 4. 🚀 Guía de Ejecución (Para tu Compañero con AMD)

Esta guía explica cómo ejecutar el *pipeline* de prueba actual en una PC con una **GPU AMD (RX 6600)** usando **Python 3.12.10**.

### 4.1. Configuración del Entorno

1.  **Instalar Python:** Asegúrate de tener **Python 3.12.10** instalado. (La versión 3.10 o 3.11 también funciona).
2.  **Instalar Drivers:** Asegúrate de tener los últimos *drivers* **AMD Adrenalin** para la RX 6600.
3.  **Clonar el Proyecto:** `git clone ...`
4.  **Crear Entorno Virtual:**
    ```bash
    # (Asegúrate de que 'python' apunte a tu instalación de 3.12)
    python -m venv venv_api
    ```
5.  **Activar Entorno:**
    * En Windows: `.\venv_api\Scripts\Activate.ps1`
6.  **Instalar Dependencias:** (Esta es la parte más importante)

    ```bash
    # Instalar la versión de ONNX Runtime para AMD (DirectML)
    pip install onnxruntime-directml
    
    # Instalar el resto de dependencias
    pip install numpy opencv-python fastapi "uvicorn[standard]" websockets
    ```
    **¡No instales `onnxruntime-gpu`!** Esa librería es solo para NVIDIA (CUDA).

### 4.2. Modificaciones de Código (¡No se necesita ninguna!)

**No necesitas modificar ningún archivo.**

El *pipeline* ya está configurado para manejar AMD. La magia está en estos dos archivos:

1.  **`model_api/config/config.py`:**
    * La variable `INFERENCE_PROVIDERS` ya incluye la opción de AMD:
    * `['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']`

2.  **`model_api/onnx_model/onnx_detector.py`:**
    * Este archivo lee esa lista.
    * En tu PC (NVIDIA), encontrará `CUDAExecutionProvider` y lo usará.
    * En la PC de tu compañero (AMD), fallará en encontrar CUDA, pero luego **encontrará `DmlExecutionProvider` (DirectML) y lo usará automáticamente.**

### 4.3. Ejecutar la Prueba de 4 Cámaras

1.  **Abrir Terminal 1 (Backend):**
    * Activa el entorno (`.\venv_api\Scripts\Activate.ps1`).
    * Ejecuta el lanzador:
        ```bash
        python run_app.py
        ```
    * Espera a que todos los procesos se inicien. Verás los logs de los 4 *workers* y el `InferenceService`. Espera hasta que veas el log final:
        > `INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`

2.  **Abrir Terminal 2 (Cliente):**
    * Activa el entorno (`.\venv_api\Scripts\Activate.ps1`).
    * Ejecuta el script de prueba de WebSocket:
        ```bash
        python test_websocket.py
        ```

3.  **Verificar Resultados:**
    * La **Terminal 1** (`run_app.py`) debería mostrar el log `[Detector] Modelo cargado y listo en: DmlExecutionProvider` (confirmando que está usando la GPU de AMD).
    * La **Terminal 2** (`test_websocket.py`) debería empezar a mostrar el *stream* de predicciones en JSON de las 4 cámaras.

---

## 5. 🛠️ Guía de Implementación en Producción

Para mover este *pipeline* de la "prueba con archivos" a la "aplicación real", tu compañero del *backend* debe modificar 3 áreas principales:

### 1. Reemplazar el Lector de Video (Ingesta)

* [cite_start]**Paso 8 del Pipeline (`pipeline.md` [cite: 664-670]):** La tarea más importante es implementar la decodificación por hardware (NVDEC para NVIDIA o VAAPI/DXVA2 para AMD).
* **Acción:**
    1.  Crear el archivo `model_api/services/stream_reader/rtsp_reader.py`.
    2.  Esta clase debe usar `GStreamer` o `FFmpeg` con Python para conectarse a una URL `rtsp://` y decodificar el video usando la GPU, no la CPU.
    3.  Actualizar el `camera_worker.py` para que reconozca `reader_type="rtsp"`:
        ```python
        # (En camera_worker.py)
        if reader_type == "file":
            stream_reader = FileReader(source_path)
        elif reader_type == "rtsp":
            stream_reader = RtspReader(source_path) # <-- AÑADIR ESTO
        ```

### 2. Conectar la Base de Datos (Orquestación)

* **Problema:** `run_app.py` actualmente "hardcodea" la lista de cámaras leyendo videos locales.
* **Acción:**
    1.  Modificar el bloque `if __name__ == "__main__":` en `run_app.py`.
    2.  **Eliminar** la función `get_video_files()` y la lógica `CAMERAS_TO_RUN`.
    3.  [cite_start]En su lugar, añadir la lógica para **consultar la base de datos PostgreSQL** (definida en tu OB2).
    4.  El *script* debe hacer algo como: `SELECT id_conexion, url_rtsp FROM conexiones WHERE estado='ACTIVA'`.
    5.  Luego, construir la lista `CAMERAS_TO_RUN` dinámicamente a partir de esa consulta:
        ```python
        # (Ejemplo en run_app.py)
        # db_cameras = ... (código para consultar la BD)
        CAMERAS_TO_RUN = []
        for cam in db_cameras:
            CAMERAS_TO_RUN.append({
                "id": cam.id_conexion,
                "type": "rtsp",  # <-- Usar el nuevo lector
                "path": cam.url_rtsp # <-- Usar la URL de la BD
            })
        
        # El resto del script (crear colas, iniciar procesos) sigue igual.
        ```

### 3. Implementar los Endpoints REST (API)

* **Problema:** `api/main.py` actualmente solo tiene el *endpoint* de WebSocket (`/ws/{camera_id}`) y un *endpoint* raíz (`/`).
* **Acción:**
    1.  Tu compañero debe añadir aquí todos los *endpoints* **REST API** que la aplicación web necesita, como se define en tu arquitectura (`OB2- Diseño del proyecto.pdf`)[cite: 292, 342, 348, 399].
    2.  Ejemplos:
        * `@app.post("/login")` (Gestión de usuarios) [cite: 293, 306, 381]
        * `@app.get("/cameras")` (Gestión de cámaras) [cite: 298, 308, 384]
        * `@app.get("/reports")` (Gestión de reportes) [cite: 302, 798]
    3.  Estos *endpoints* contendrán la lógica de negocio para leer y escribir en la base de datos PostgreSQL.