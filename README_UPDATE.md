# Actualización del Backend: Integración de Filtro de Personas (YOLOv8n)

Hola,

He implementado el pre-filtro de conteo de personas en el *pipeline* del backend.

El **objetivo** es optimizar el `inference_service` (el proceso de la GPU) para que nuestro modelo principal (Swin3D-T) solo analice clips cuando haya **2 o más personas** en la escena. Esto evita que la GPU procese clips de escenas vacías o con una sola persona, reduciendo drásticamente la carga de trabajo y los falsos positivos.

Para implementar esto, tuve que crear 2 archivos nuevos y modificar 2 existentes. Aquí está el detalle completo para que puedas actualizar la versión del backend:

---

## 1. 📁 Nuevos Archivos Añadidos

### 1.1. Nuevo Modelo de Detección (YOLO)

* **Archivo:** `model_api/onnx_model/person_detector/yolov8n.onnx`
* **Descripción:** Es el modelo **YOLOv8n (nano)** pre-entrenado, exportado a formato ONNX (opset 12). Este es el modelo ligero que usaremos para el conteo rápido de personas.

### 1.2. Nuevo "Wrapper" del Detector de Personas

* **Archivo:** `model_api/onnx_model/onnx_person_detector.py`
* **Descripción:** Creé esta nueva clase `PersonDetector` (siguiendo el mismo estilo de nuestro `onnx_detector.py`) que encapsula toda la lógica de YOLO:
    * **Carga (Lazy Loading):** Carga el modelo `yolov8n.onnx`.
    * **Proveedor Forzado:** Está configurada para usar **`'CPUExecutionProvider'`**. Esto es crucial para que el filtro corra en la CPU del *worker* y no compita con el `inference_service` de la GPU.
    * **Funciones:** Contiene `_preprocess()` (para redimensionar el frame a 320x320), `_postprocess()` (para contar las detecciones de la clase 0 - "persona"), y la función principal `count_persons(frame)`.

---

## 2. 📝 Archivos Modificados

### 2.1. `run_app.py` (El Orquestador)

* **Objetivo del Cambio:** Darle al `camera_worker` acceso (permiso) al `results_queue`.
* **Función Modificada:** `main()`.
* **Detalle del Cambio:**
    * Dentro del bucle `for cam in cameras_to_run:`, modifiqué los `args` (argumentos) que se pasan al proceso `run_camera_worker`.
    * **Parámetro Añadido:** Ahora le pasamos la `results_queue` al final de la tupla de `args`, después del `control_queues[cam["id"]]`.
* **Por qué:** El `camera_worker` ahora necesita "saltarse" la GPU y enviar predicciones `[0.0, 0.0, 0.0]` directamente al `event_manager`. Como el `event_manager` ya escucha el `results_queue`, el *worker* necesitaba acceso a esa cola para implementar el *bypass*.

### 2.2. `model_api/services/camera_worker.py` (El Worker de CPU)

Este es el cambio más importante. Aquí es donde se implementa toda la nueva lógica de filtrado.

* **Objetivo del Cambio:** Integrar el `PersonDetector` para filtrar los clips antes de enviarlos a la GPU.
* **Nuevos Imports:**
    * `from onnx_model.onnx_person_detector import PersonDetector`.
* **Función Modificada:** `run_camera_worker(...)`.
* **Nuevos Parámetros de Función:**
    * La firma de la función ahora acepta `results_queue: Queue` al final. (Esto coincide con el cambio en `run_app.py`).
* **Variables Nuevas (dentro de la función):**
    * `person_detector: Union[PersonDetector, None] = None`.
    * Al inicio de la función (Sección `1. Inicialización`), ahora instanciamos `person_detector = PersonDetector()`. Esto carga el modelo YOLO en la memoria de la CPU **una sola vez** por *worker*, lo cual es muy eficiente.
* **Lógica Modificada (La parte más importante):**
    * La lógica de inferencia en la **Sección `2d. Lógica de Inferencia`**, que comenzaba con `if (len(inference_buffer) == INFERENCE_BUFFER_SIZE and ...)`, fue **completamente reescrita**.
    * **Paso 1 (Filtro):** Ahora, lo primero que hacemos dentro de ese `if` es llamar a `person_count = person_detector.count_persons(frame)`. Usamos el `frame` más reciente para la detección.
    * **Paso 2 (Decisión):** Se implementó una nueva lógica `if/elif/else`:
        * **`if person_count >= 2:` (Camino Caro):**
            * Si hay 2 o más personas, se ejecuta la lógica *anterior*.
            * Llama a `preprocess_clip(list(inference_buffer))`.
            * Pone el *tensor* de video resultante en la `inference_queue` (para la GPU).
        * **`elif person_count < 0:` (Manejo de Errores):**
            * Si YOLO falla (devuelve -1), solo se loguea el error y se omite el ciclo.
        * **`else: (person_count < 2)` (Camino Barato / Bypass):**
            * Si hay 0 o 1 persona, **NO** se llama a `preprocess_clip()` (¡Ahorro de CPU!).
            * **NO** se pone nada en la `inference_queue` (¡Ahorro de GPU!).
            * Se crea `neutral_probs = np.array([0.0] * len(config.CLASSES))`.
            * Se pone `neutral_probs` **directamente en la `results_queue`**. Esto es para que el `event_manager` reciba un `[0,0,0]` y el *frontend* sepa que la cámara sigue viva.
            * Se actualiza `last_known_probs = neutral_probs` para que el `EventRecorder` guarde los datos correctos si justo estaba grabando.

---

## 3. 🚀 Resumen de Tareas para Actualizar

Para actualizar el backend, necesitas hacer lo siguiente:

1.  **Añadir Nuevos Archivos:**
    * Asegúrate de tener el modelo en: `model_api/onnx_model/person_detector/yolov8n.onnx`.
    * Añade el nuevo archivo: `model_api/onnx_model/onnx_person_detector.py`.
2.  **Actualizar Archivos:**
    * Reemplaza el contenido de `run_app.py` con la nueva versión.
    * Reemplaza el contenido de `model_api/services/camera_worker.py` con la nueva versión.
3.  **Verificar Dependencias:**
    * El `PersonDetector` necesita `onnxruntime` (la versión de CPU, ¡no `onnxruntime-gpu`!). El script de exportación de YOLO (que corrí yo) ya debería haberlo instalado en el `venv_api`.