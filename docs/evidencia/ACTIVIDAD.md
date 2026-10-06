# Evidencia de la actividad: Dataset de personas

## Objetivo

Crear un dataset propio de imágenes de personas para posteriormente entrenar un modelo de segmentación.

## Dataset original

Se recolectaron 100 imágenes de personas y se almacenaron en:

datos/raw/

La clase utilizada fue:

person

Las 100 imágenes fueron procesadas mediante SAM3.

## Procesamiento con SAM3

Configuración utilizada:

- Modelo: SAM3.
- Dispositivo: NVIDIA GTX 1660 Ti Max-Q.
- GPU mediante CUDA.
- Resolución: 1008 × 1008.
- Precisión: Float32.
- Prompt: person.

Resultados:

- 100 imágenes procesadas.
- 89 imágenes con objetos detectados.
- 307 detecciones iniciales.
- 231 detecciones conservadas después del filtrado.

### Filtrado

Se conservaron las detecciones que cumplieron:

score >= 0.70
area_frac >= 0.005

## Dataset para YOLO

El dataset para entrenamiento quedó organizado como:

datos/salida/yolo_personas/yolo/

├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml

Imágenes utilizadas por YOLO:

- Entrenamiento: 66 imágenes.
- Validación: 17 imágenes.
- Total: 83 imágenes.

La separación se realizó por imagen original para evitar que imágenes derivadas de una misma imagen aparecieran tanto en entrenamiento como en validación.

## Entrenamiento

Modelo utilizado:

YOLO26n-seg

Configuración:

- Épocas máximas: 100.
- Tamaño de imagen: 640.
- Batch: 2.
- GPU: dispositivo 0.
- Workers: 0.
- Patience: 20.
- AMP: desactivado.

El entrenamiento terminó mediante Early Stopping en la época 30.

El mejor resultado se obtuvo en la época 10.

El mejor modelo generado fue:

runs/segment/entrenamientos/personas_seg-3/weights/best.pt

## Resultados

La validación se realizó con:

- 17 imágenes.
- 67 instancias.

### Detección de cajas

| Métrica | Resultado |
|---|---:|
| Precision | 0.772 |
| Recall | 0.507 |
| mAP50 | 0.594 |
| mAP50-95 | 0.346 |

### Segmentación

| Métrica | Resultado |
|---|---:|
| Precision | 0.851 |
| Recall | 0.478 |
| mAP50 | 0.565 |
| mAP50-95 | 0.326 |

## Evidencias generadas

Los resultados del entrenamiento incluyen:

- results.png
- results.csv
- confusion_matrix.png
- confusion_matrix_normalized.png
- BoxP_curve.png
- BoxR_curve.png
- BoxF1_curve.png
- BoxPR_curve.png
- MaskP_curve.png
- MaskR_curve.png
- MaskF1_curve.png
- MaskPR_curve.png

También se generaron imágenes de validación y una predicción de prueba.

## Scripts principales

- descargar_personas.py
- scripts/descargar_pesos_sam3.py
- scripts/sam3_recortar_dataset.py
- scripts/entrenar_personas.py

## Conclusión

Se completó el flujo de creación de un dataset propio de personas, incluyendo la recopilación de 100 imágenes, la generación de detecciones mediante SAM3, el filtrado de los resultados y el entrenamiento de un modelo YOLO26n-seg para segmentación.

El modelo obtuvo un mAP50 de 0.565 en segmentación durante la validación.
