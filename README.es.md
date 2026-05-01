# EAGLE
<p align="center">
  EAGLE: <strong>E</strong>nd-to-end <strong>A</strong>utomatic <strong>G</strong>aze <strong>L</strong>ab<strong>E</strong>ling tool for interaction studies
</p>

<p align="center">
  <img src="assets/icon_name_trans.png" alt="EAGLE app icon" width="280" />
</p>

## Idiomas

- [English](README.md)
- [日本語](README.ja.md)
- [简体中文](README.zh.md)
- [Español](README.es.md)

## Descripción general
EAGLE es una herramienta de apoyo para anotación de mirada, basada en Streamlit, para análisis de imágenes y videos. Combina:

- Seguimiento de personas con YOLO pose
- Seguimiento de objetos no humanos con YOLO detection
- Detección de rostros con RetinaFace
- Estimación de mapas de calor de mirada con GAZELLE
- Estimación de dirección fuera de pantalla con MobileOne gaze
- Exportación CSV y exportación de imágenes/videos anotados

EAGLE es una herramienta de asistencia para anotación, no una fuente garantizada de verdad terreno. Debes revisar y validar todos los resultados antes de usarlos en investigación, análisis, informes o toma de decisiones.

## Ejemplos de salida
<p align="center">
  <img src="assets/book.gif" alt="EAGLE example output: book" width="31%" />
  <img src="assets/interview.gif" alt="EAGLE example output: interview" width="31%" />
  <img src="assets/three.gif" alt="EAGLE example output: three people" width="31%" />
</p>

## Ejemplo de GUI
<p align="center">
  <img src="assets/GUI.png" alt="EAGLE GUI example" width="90%" />
</p>

## Qué hace el sistema actual
Para cada imagen o video de entrada, el pipeline actual puede:

- Detectar y seguir personas a partir de detecciones de pose
- Mantener opcionalmente todas las clases COCO o solo clases seleccionadas
- Detectar un rostro por cada persona seguida
- Estimar mapas de calor y puntos de mirada
- Inferir etiquetas de dirección fuera de pantalla como `left`, `up right`, `down` o `front` (orientación centrada en la persona)
- Generar segmentos de mirada estilo ELAN en `annotation.csv`
- Reutilizar `objects.csv` y salidas de mirada en caché cuando los ajustes siguen coincidiendo
- Forzar la reutilización de cachés antiguos incluso cuando la app detecta diferencias de configuración

Para detecciones de personas, EAGLE también usa pose keypoints para asignar la mirada a partes del cuerpo como face, head, torso, arms o legs cuando es posible.

## Entradas compatibles
- Imágenes: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- Videos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.wmv`, `.webm`

## Notas actuales de ejecución
- En macOS, `Browse` abre un diálogo de archivo nativo mediante `osascript`.
- En Linux, incluido Docker, usa `Container File Browser` o introduce manualmente una ruta montada.
- El pipeline principal es código Python normal bajo [`eagle/`](eagle/).
- Se incluyen binarios de FFmpeg para macOS y Windows.
- Los archivos de modelos se guardan en caché bajo `~/.EAGLE/`.

## Licencia
Este proyecto está licenciado bajo `AGPL-3.0-or-later`. Consulta [LICENSE](LICENSE) para el texto de licencia del repositorio.

## Descarga
Los paquetes precompilados de la aplicación están disponibles en GitHub Releases:

- https://github.com/Taiga-Mori/EAGLE/releases

## Configuración
Crea un entorno virtual e instala las dependencias:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Versión recomendada de Python: `3.10`

## Iniciar la aplicación
Recomendado:

```bash
python app.py
```

Este lanzador inicia Streamlit por ti y abre la aplicación en un navegador. Si el puerto `8501` no está disponible, prueba automáticamente puertos cercanos.

También puedes ejecutar Streamlit directamente:

```bash
venv/bin/streamlit run app.py
```

## Crear una aplicación distribuible
Se incluyen archivos spec de PyInstaller para macOS y Windows.

macOS:

```bash
source venv/bin/activate
pip install pyinstaller
pyinstaller mac.spec
```

Windows:

```bash
venv\Scripts\activate
pip install pyinstaller
pyinstaller win.spec
```

Las salidas de compilación se crean bajo `dist/`. En la primera ejecución, la app empaquetada aún puede descargar pesos de modelos en `~/.EAGLE/`.

## Primera ejecución y descarga de modelos
En el primer uso, EAGLE puede descargar y guardar en caché:

- `yolo26x.pt`
- `yolo26x-pose.pt`
- Pesos preentrenados de RetinaFace
- Archivos torch.hub de GAZELLE
- `mobileone_s0.pt`

Si falla la carga inicial, comprueba:

- Que la máquina esté en línea
- Que GitHub / los endpoints de alojamiento de modelos sean accesibles
- Que el usuario actual pueda escribir en `~/.EAGLE/`

## Flujo de trabajo de la aplicación
1. Inicia la app con `python app.py`.
2. Configura `Input file`:
   - macOS: haz clic en `Browse`.
   - Linux/Docker: usa `Container File Browser` o escribe una ruta montada.
3. Selecciona una imagen o video de entrada.
4. Confirma el tipo de medio detectado.
5. Edita `Output folder name` si hace falta.
6. Abre `Detailed Settings` si quieres.
7. Haz clic en `Run Pipeline`.
8. Revisa las rutas de salida mostradas al final.

Comportamiento del directorio de salida:

- Si el directorio de entrada permite escritura, el padre de salida es el mismo que el padre del archivo de entrada.
- Si el directorio de entrada es de solo lectura, común con montajes Docker `:ro`, la salida usa `/app/output/<input_stem>`.
- El nombre predeterminado de la carpeta de salida es el stem de la entrada.
- Ejemplo:
  - Entrada: `/data/session01/test.mp4`
  - Padre con escritura: `/data/session01/test`
  - Fallback para padre de solo lectura: `/app/output/test`

## Ajustes principales

### Basic Settings
- `Input file`
  - macOS: campo de solo lectura rellenado por `Browse`.
  - Linux/Docker: campo editable, con entrada manual de ruta.
- `Output folder name`
  - Carpeta creada junto al archivo de entrada.
- `Device`
  - Usa `cuda:0`, `cuda:1`, ... cuando hay varias GPU NVIDIA disponibles; si no, `mps` o `cpu`.

### Inference
- `Detection threshold`
  - Umbral compartido para filtrar objetos, filtrar rostros e interpretar gaze in/out.
- `Visualization mode`
  - `both`, `point` o `heatmap`
- `Heatmap alpha`
  - Intensidad de superposición para salidas de heatmap.
- `Gaze target radius (px)`
  - `0` significa asignación solo por punto. Valores mayores usan un área circular alrededor del punto de mirada.
- `Person part distance scale`
  - Controla cuán lejos puede estar la mirada de los keypoints de una persona y seguir contando como parte del cuerpo.
- `Reuse existing objects.csv when available`
  - Reutiliza detecciones de objetos en caché cuando los metadatos siguen coincidiendo con la ejecución actual.
- `Reuse existing gaze.csv and heatmaps.npz when available`
  - Reutiliza salidas de mirada en caché cuando los metadatos siguen coincidiendo.
- `Track all object classes`
  - Al desactivarlo, puedes elegir explícitamente qué clases COCO conservar.

### Temporal Settings
- `Object smoothing window`
  - Ventana de suavizado para cajas de objetos seguidos.
- `Gaze smoothing window`
  - Ventana de suavizado para estimaciones de mirada y ángulos de dirección fuera de pantalla.
- `Object frame interval`
  - Solo videos. La detección/seguimiento de objetos se ejecuta cada N fotogramas.
- `Gaze frame interval`
  - Solo videos. La estimación de mirada se ejecuta cada N fotogramas y luego se interpola/suaviza.

Importante:

- `Gaze frame interval` puede ser menor que `Object frame interval`. Los object tracks se interpolan linealmente entre fotogramas de detección y los videos se renderizan al FPS de origen.
- Internamente, los ajustes de video se convierten a valores target FPS a partir del FPS de origen.

### BoT-SORT
La UI expone estos ajustes de tracker desde [`config/botsort.yaml`](config/botsort.yaml):

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- `Enable ReID`
- `proximity_thresh`
- `appearance_thresh`

## Comportamiento de caché
EAGLE escribe archivos de metadatos de caché y los revisa antes de reutilizar resultados:

- `objects.csv` con `.objects_meta.json`
- `gaze.csv` con `gaze_heatmaps.npz` y `.gaze_meta.json`

La reutilización de object cache depende de:

- Ruta del archivo de entrada
- Timestamp del archivo de entrada
- Detection threshold
- Object frame interval
- Object smoothing window
- Clases de objetos seleccionadas
- Ajustes de BoT-SORT
- Si la caché fue creada con el formato actual de seguimiento de personas basado en pose

La reutilización de gaze cache depende de:

- Ruta del archivo de entrada
- Timestamp del archivo de entrada
- Detection threshold
- Gaze frame interval
- Gaze smoothing window
- Timestamp de `objects.csv`

Si la app detecta una diferencia, muestra una advertencia y ofrece un checkbox `Force reuse`.

## Archivos de salida
Las salidas actuales pueden incluir:

- `objects.csv`
  - Resultados suavizados de seguimiento de objetos/personas
- `.objects_meta.json`
  - Metadatos usados para validar la reutilización de object cache
- `gaze.csv`
  - Detecciones de rostro, valores raw gaze, puntos de mirada procesados y campos de dirección fuera de pantalla
- `gaze_heatmaps.npz`
  - Mapas de calor dense de mirada en caché
- `.gaze_meta.json`
  - Metadatos usados para validar la reutilización de gaze cache
- `annotation.csv`
  - Segmentos de mirada estilo ELAN o etiquetas para una sola imagen
- `all_points.jpg`
  - Visualización de puntos para entrada de imagen
- `all_points.mp4`
  - Visualización de puntos para entrada de video
- `person_<track_id>_heatmap.jpg`
  - Heatmap por persona para entrada de imagen
- `person_<track_id>_heatmap.mp4`
  - Heatmap por persona para entrada de video

Las carpetas temporales como `temp/` y `heatmaps/` se eliminan después de exportar.

## Contenido de CSV
Columnas de `objects.csv`:

- `frame_idx`
- `cls`
- `track_id`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

Columnas de `gaze.csv`:

- `frame_idx`
- `track_id`
- `face_detected`
- `face_conf`
- `face_x1`, `face_y1`, `face_x2`, `face_y2`
- `raw_gaze_detected`
- `raw_inout`
- `raw_x_gaze`, `raw_y_gaze`
- `gaze_detected`
- `inout`
- `x_gaze`, `y_gaze`
- `offscreen_direction`
- `offscreen_yaw`
- `offscreen_pitch`

Columnas de `annotation.csv`:

- `tier`
- `start_time`
- `end_time`
- `gaze`

## Puntos de entrada para desarrollo
- [`app.py`](app.py)
  - UI de Streamlit y lanzador
- [`main.py`](main.py)
  - Entrada mínima para smoke test local
- [`eagle/pipeline.py`](eagle/pipeline.py)
  - Fachada principal de la API Python

Uso mínimo en código:

```python
from eagle import EAGLE

eagle = EAGLE()
eagle.preprocess(
    input_path="input.mp4",
    output_dir="output_dir",
    det_thresh=0.5,
    device="cpu",
    visualization_mode="both",
)
results = eagle.run_all()
```

## Validación
Validación básica de sintaxis:

```bash
python -m py_compile main.py app.py eagle/*.py
```

## Aviso legal
- Usa este software bajo tu propio riesgo.
- EAGLE no garantiza la corrección de detecciones, estimaciones de mirada, asignaciones de objetivo ni anotaciones exportadas.
- La responsabilidad final de revisión y corrección permanece con el usuario.

## Agradecimientos
- Ultralytics YOLO
  - https://docs.ultralytics.com/
- BoT-SORT
  - https://github.com/NirAharon/BoT-SORT
- RetinaFace
  - https://github.com/serengil/retinaface
- GAZELLE
  - https://github.com/fkryan/gazelle
- MobileOne gaze-estimation weights
  - https://github.com/yakhyo/gaze-estimation
