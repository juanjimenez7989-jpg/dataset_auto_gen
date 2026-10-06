import json
import random
import requests
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

CANTIDAD = 100

BASE_DIR = Path(__file__).resolve().parent
SALIDA = BASE_DIR / "datos" / "raw"

SALIDA.mkdir(parents=True, exist_ok=True)

# Anotaciones de COCO 2017 Validation
URL_ANOTACIONES = (
    "https://images.cocodataset.org/annotations/"
    "annotations_trainval2017.zip"
)

URL_JSON = (
    "https://huggingface.co/datasets/merve/coco/"
    "resolve/main/annotations/instances_val2017.json"
)

# ============================================================
# COMPROBAR IMÁGENES EXISTENTES
# ============================================================

existentes = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    existentes.extend(SALIDA.glob(extension))

if len(existentes) >= CANTIDAD:
    print("=" * 60)
    print("YA EXISTEN LAS 100 IMÁGENES")
    print("=" * 60)
    print(f"Imágenes encontradas: {len(existentes)}")
    print(f"Carpeta: {SALIDA}")
    exit()

print("=" * 60)
print("DESCARGADOR DE PERSONAS - COCO 2017")
print("=" * 60)

print("\nDescargando información de COCO...")
print("Esto NO descarga las imágenes completas del dataset.")

# ============================================================
# DESCARGAR JSON DE ANOTACIONES
# ============================================================

json_temp = BASE_DIR / "instances_val2017.json"

if not json_temp.exists():

    respuesta = requests.get(URL_JSON, timeout=60)
    respuesta.raise_for_status()

    json_temp.write_bytes(respuesta.content)

    print("✓ Anotaciones descargadas.")

else:

    print("✓ Las anotaciones ya existen.")

# ============================================================
# LEER ANOTACIONES
# ============================================================

print("\nAnalizando imágenes que contienen personas...")

with open(json_temp, "r", encoding="utf-8") as archivo:
    datos = json.load(archivo)

# En COCO la categoría person tiene ID 1
imagenes_personas = set()

for anotacion in datos["annotations"]:

    if anotacion["category_id"] == 1:
        imagenes_personas.add(anotacion["image_id"])

print(f"Imágenes de COCO con personas: {len(imagenes_personas)}")

# ============================================================
# OBTENER INFORMACIÓN DE LAS IMÁGENES
# ============================================================

imagenes_dict = {
    imagen["id"]: imagen
    for imagen in datos["images"]
}

candidatas = [
    imagenes_dict[id_imagen]
    for id_imagen in imagenes_personas
    if id_imagen in imagenes_dict
]

random.seed(42)
random.shuffle(candidatas)

# ============================================================
# DESCARGAR 100
# ============================================================

faltantes = CANTIDAD - len(existentes)

print(f"\nNecesitamos descargar: {faltantes}")
print()

descargadas = len(existentes)

for imagen in candidatas:

    if descargadas >= CANTIDAD:
        break

    nombre = f"persona_{descargadas + 1:03d}.jpg"
    destino = SALIDA / nombre

    url = imagen["coco_url"]

    try:

        print(
            f"[{descargadas + 1:03d}/{CANTIDAD}] "
            f"Descargando {nombre}..."
        )

        respuesta = requests.get(
            url,
            timeout=60
        )

        respuesta.raise_for_status()

        # Comprobar que realmente sea una imagen
        if len(respuesta.content) < 1000:
            print("    ⚠ Archivo demasiado pequeño. Se omite.")
            continue

        destino.write_bytes(respuesta.content)

        descargadas += 1

    except Exception as error:

        print(f"    ⚠ Error: {error}")

# ============================================================
# LIMPIEZA
# ============================================================

try:
    json_temp.unlink()
except:
    pass

# ============================================================
# RESULTADO
# ============================================================

print()
print("=" * 60)
print("PROCESO TERMINADO")
print("=" * 60)

imagenes_finales = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    imagenes_finales.extend(SALIDA.glob(extension))

print(f"Imágenes encontradas: {len(imagenes_finales)}")
print(f"Ubicación: {SALIDA}")

if len(imagenes_finales) >= CANTIDAD:

    print("\n✓ ¡LISTO!")
    print("Tienes las 100 imágenes de personas.")

else:

    print(
        f"\n⚠ Solo se obtuvieron "
        f"{len(imagenes_finales)} imágenes."
    )