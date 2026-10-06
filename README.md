
# Dataset de Personas - Segmentación con SAM3 y YOLO26

## Descripción

Este proyecto corresponde a una actividad de Visión Artificial en la que se desarrolló un dataset personalizado de personas para posteriormente entrenar un modelo de segmentación.

El flujo completo utilizado fue:

**Recolección de imágenes → SAM3 → generación de segmentaciones → filtrado → dataset YOLO-seg → entrenamiento YOLO26n-seg → evaluación → predicción**

## Objetivo

Crear un dataset con al menos 100 imágenes relacionadas con la tarea del proyecto y utilizarlo para entrenar un modelo propio de identificación o segmentación.

En este caso se seleccionó la clase:

**person**

El objetivo final fue entrenar un modelo capaz de identificar y segmentar personas dentro de una imagen.

## Dataset

Se recopilaron:

- **100 imágenes originales**
- Formato principal: JPG
- Clase: `person`
- Imágenes procesadas: **100**
- Imágenes en las que SAM3 encontró al menos una persona: **89**
- Detecciones generadas inicialmente: **307**

Las imágenes originales se encuentran en:

```text
datos/raw/
```

Ejemplo:

```text
datos/raw/
├── persona_001.jpg
├── persona_002.jpg
├── ...
└── persona_100.jpg
```

## Generación automática de anotaciones

Para evitar realizar manualmente la segmentación de las 100 imágenes, se utilizó **SAM 3 (Segment Anything Model 3)** con el prompt:

```text
person
```

SAM3 procesó las imágenes y generó máscaras y detecciones de las personas encontradas.

La configuración utilizada para este equipo fue:

- GPU: NVIDIA GeForce GTX 1660 Ti con Max-Q Design
- CUDA: activado
- Resolución SAM3: 1008 × 1008
- Tipo de datos: Float32
- Clase detectada: `person`

La resolución de 1008 se mantuvo porque es la configuración compatible con el backbone utilizado en esta instalación de SAM3.

## Filtrado de detecciones

Después de generar las detecciones se aplicaron filtros para reducir detecciones pequeñas o de baja confianza.

Criterios utilizados:

```text
Score mínimo: 0.70
Área mínima: 0.005
```

Resultados:

- Detecciones iniciales: **307**
- Detecciones conservadas después del filtrado: **231**

Los resultados y metadatos originales de este procesamiento se encuentran localmente en:

```text
datos/salida/final/
```

El archivo principal de metadatos es:

```text
datos/salida/final/metadatos.csv
```

## Dataset para entrenamiento

Las máscaras generadas por SAM3 se utilizaron para construir un dataset compatible con **YOLO-seg**.

La estructura utilizada fue:

```text
datos/salida/yolo_personas/yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

La clase utilizada es:

```text
0: person
```

Después del filtrado y de la separación, el conjunto utilizado por YOLO quedó distribuido en:

- **66 imágenes para entrenamiento**
- **17 imágenes para validación**

## Entrenamiento

Se entrenó un modelo:

```text
YOLO26n-seg
```

utilizando Ultralytics.

Configuración principal:

```text
Modelo base: yolo26n-seg.pt
Épocas máximas: 100
Tamaño de imagen: 640 × 640
Batch: 2
GPU: NVIDIA GeForce GTX 1660 Ti
Workers: 0
Early Stopping: 20
AMP: desactivado
```

Aunque se establecieron 100 épocas máximas, el entrenamiento terminó anticipadamente mediante Early Stopping.

El mejor resultado se obtuvo en:

```text
Época 10
```

El entrenamiento finalizó después de:

```text
30 épocas
```

El modelo generado se guardó como:

```text
runs/segment/entrenamientos/personas_seg-3/weights/best.pt
```

## Resultados de validación

Resultados obtenidos realmente durante la validación:

### Detección de cajas

| Métrica | Resultado |
|---|---:|
| Precision | 0.772 |
| Recall | 0.507 |
| mAP50 | 0.594 |
| mAP50-95 | 0.346 |

### Segmentación de máscaras

| Métrica | Resultado |
|---|---:|
| Precision | 0.851 |
| Recall | 0.478 |
| mAP50 | 0.565 |
| mAP50-95 | 0.326 |

Para este proyecto, las métricas de máscaras son las más importantes porque el objetivo principal es la segmentación.

## Evidencia

La evidencia generada durante el proyecto incluye:

- Dataset original de 100 imágenes.
- Mosaico de detecciones generado por SAM3.
- Archivo `metadatos.csv`.
- Dataset YOLO-seg.
- Gráficas de entrenamiento.
- Matriz de confusión.
- Modelo entrenado `best.pt`.
- Imagen de predicción realizada con el modelo entrenado.

## Flujo del proyecto

```text
100 imágenes
      ↓
Dataset personalizado
      ↓
SAM 3
      ↓
Detección y segmentación automática
      ↓
307 detecciones
      ↓
Filtrado
      ↓
231 detecciones conservadas
      ↓
Dataset YOLO-seg
      ↓
66 imágenes train
17 imágenes val
      ↓
YOLO26n-seg
      ↓
Entrenamiento
      ↓
Modelo personalizado
      ↓
Predicción y segmentación de personas
```

## Scripts principales

### `scripts/descargar_personas.py`

Script utilizado para apoyar la recopilación de imágenes del dataset.

### `scripts/sam3_recortar_dataset.py`

Procesa las imágenes mediante SAM3, genera las detecciones y permite obtener resultados y anotaciones para el dataset.

### `scripts/entrenar_personas.py`

Contiene la configuración utilizada para entrenar el modelo de segmentación.

## Reproducibilidad

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Procesar las imágenes:

```bash
python scripts/sam3_recortar_dataset.py
```

Entrenar el modelo:

```bash
python scripts/entrenar_personas.py
```

El entrenamiento genera los resultados dentro de la carpeta:

```text
runs/
```

## Nota sobre archivos grandes

Los pesos de SAM3 y los resultados grandes del entrenamiento no se incluyen en el repositorio.

Las carpetas y archivos generados localmente que contienen resultados o pesos se mantienen fuera del control de versiones para evitar almacenar archivos innecesariamente grandes.

## Autor

Juan Diego Jiménez Vizcarra

Proyecto académico de Visión Artificial.