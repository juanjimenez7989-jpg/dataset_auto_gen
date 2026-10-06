#!/usr/bin/env python3
"""
sam3_recortar_dataset.py
========================

Recorre un directorio de imágenes, segmenta con SAM 3 los objetos descritos
por uno o varios prompts de texto y guarda cada objeto recortado como un
dataset nuevo, limpio y organizado por clase.

Opcionalmente filtra los recortes por parecido visual con imágenes de
referencia (DINOv2) y exporta etiquetas YOLO-seg de las imágenes completas
para subirlas a Roboflow o entrenar con Ultralytics.

Requisitos
----------
1. Repo oficial instalado:  https://github.com/facebookresearch/sam3
   (Python >= 3.12, PyTorch >= 2.7, GPU con CUDA recomendada)
2. Acceso aprobado al checkpoint en https://huggingface.co/facebook/sam3
   y sesión iniciada con:  hf auth login
3. pip install opencv-python tqdm

Ejemplos
--------
# Una clase (el prompt conviene escribirlo en inglés)
python sam3_recortar_dataset.py -i fotos/ -o dataset/ -p "casco=yellow hard hat"

# Varias clases, fondo transparente y recortes cuadrados de 224 px
python sam3_recortar_dataset.py -i fotos/ -o dataset/ \
    -p "casco=hard hat" -p "chaleco=safety vest" \
    --fondo transparente --tamano 224

# Calibrar umbrales con solo 20 imágenes antes de correr todo
python sam3_recortar_dataset.py -i fotos/ -o prueba/ -p "gato=cat" --limite 20

# Filtrar por parecido con fotos de referencia del objeto que quieres
python sam3_recortar_dataset.py -i fotos/ -o dataset/ -p "taza=mug" \
    --referencias refs/ --umbral-similitud 0.5

# Exportar también etiquetas YOLO-seg de las imágenes completas
python sam3_recortar_dataset.py -i fotos/ -o dataset/ -p "fruta=apple" --exportar-yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Utilidades generales
# --------------------------------------------------------------------------- #
def barra_progreso(iterable, total=None, desc=""):
    """Usa tqdm si está instalado; si no, imprime avance simple."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        def _gen():
            for i, x in enumerate(iterable, 1):
                if i == 1 or i % 10 == 0 or i == total:
                    print(f"{desc}: {i}/{total}", flush=True)
                yield x

        return _gen()


def listar_imagenes(directorio: Path, recursivo: bool) -> list[Path]:
    patron = "**/*" if recursivo else "*"
    return sorted(
        p for p in directorio.glob(patron)
        if p.is_file() and p.suffix.lower() in EXTENSIONES
    )


def abrir_imagen(ruta: Path) -> Image.Image:
    """Abre en RGB respetando la orientación EXIF (fotos de celular)."""
    img = Image.open(ruta)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def parsear_prompt(texto: str) -> tuple[str, str]:
    """'clase=frase' -> (clase, frase). Sin '=', la clase es la frase."""
    if "=" in texto:
        clase, frase = texto.split("=", 1)
    else:
        clase, frase = texto, texto
    clase = clase.strip().replace(" ", "_").replace("/", "_")
    return clase, frase.strip()


def iou_mascaras(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


# --------------------------------------------------------------------------- #
# Detección con SAM 3
# --------------------------------------------------------------------------- #
@dataclass
class Deteccion:
    clase: str
    score: float
    caja: tuple[int, int, int, int]  # x0, y0, x1, y1 en píxeles
    mascara: np.ndarray               # bool (H, W)
    area_frac: float = 0.0
    similitud: float | None = None
    extra: dict = field(default_factory=dict)


class SegmentadorSAM3:
    def __init__(self, checkpoint: str | None, dispositivo: str, umbral: float):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.dispositivo = dispositivo
        if dispositivo == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("Cargando SAM 3 (la primera vez descarga ~3 GB)...", flush=True)
        modelo = build_sam3_image_model(
            device=dispositivo,
            checkpoint_path=checkpoint,
            load_from_HF=checkpoint is None,
        )
        self.procesador = Sam3Processor(
            modelo, device=dispositivo, confidence_threshold=umbral
        )

    def detectar(self, imagen: Image.Image, prompts: list[tuple[str, str]]) -> list[Deteccion]:
        torch = self.torch
        usar_autocast = self.dispositivo == "cuda"
        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if usar_autocast
            else torch.autocast("cpu", enabled=False)
        )
        detecciones: list[Deteccion] = []
        with ctx:
            # El encoder de imagen corre una sola vez; los prompts reutilizan sus features
            estado = self.procesador.set_image(imagen)
            for clase, frase in prompts:
                self.procesador.reset_all_prompts(estado)
                estado = self.procesador.set_text_prompt(prompt=frase, state=estado)
                mascaras = estado["masks"].squeeze(1).cpu().numpy().astype(bool)
                cajas = estado["boxes"].float().cpu().numpy()
                scores = estado["scores"].float().cpu().numpy()
                for m, c, s in zip(mascaras, cajas, scores):
                    x0, y0, x1, y1 = (int(round(v)) for v in c)
                    detecciones.append(Deteccion(clase, float(s), (x0, y0, x1, y1), m))
        return detecciones


# --------------------------------------------------------------------------- #
# Filtro opcional por parecido visual (DINOv2)
# --------------------------------------------------------------------------- #
class FiltroReferencias:
    """
    SAM 3 acepta 'ejemplares visuales' solo como cajas dentro de la MISMA imagen.
    Para usar fotos de referencia externas, calculamos embeddings DINOv2 de las
    referencias y de cada recorte, y conservamos los recortes parecidos.

    Si el directorio de referencias tiene subcarpetas con el nombre de cada
    clase, se usan por clase; si no, todas las imágenes aplican a todas las clases.
    """

    def __init__(self, directorio: Path, dispositivo: str):
        import torch
        import torchvision.transforms as T

        self.torch = torch
        self.dispositivo = dispositivo
        print("Cargando DINOv2 para el filtro de referencias...", flush=True)
        self.modelo = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.modelo.eval().to(dispositivo)
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.por_clase: dict[str, "torch.Tensor"] = {}
        self.global_: "torch.Tensor | None" = None
        subdirs = [d for d in directorio.iterdir() if d.is_dir()]
        if subdirs:
            for d in subdirs:
                imgs = listar_imagenes(d, recursivo=True)
                if imgs:
                    self.por_clase[d.name] = self._embeber([abrir_imagen(p) for p in imgs])
        sueltas = listar_imagenes(directorio, recursivo=False)
        if sueltas:
            self.global_ = self._embeber([abrir_imagen(p) for p in sueltas])
        if not self.por_clase and self.global_ is None:
            raise SystemExit(f"No encontré imágenes de referencia en {directorio}")

    def _embeber(self, imagenes: list[Image.Image]):
        torch = self.torch
        lote = torch.stack([self.transform(im) for im in imagenes]).to(self.dispositivo)
        with torch.inference_mode():
            emb = self.modelo(lote)
        return torch.nn.functional.normalize(emb.float(), dim=-1)

    def similitud(self, recorte: Image.Image, clase: str) -> float:
        refs = self.por_clase.get(clase, self.global_)
        if refs is None:
            return 1.0  # sin referencias para esta clase: no se filtra
        emb = self._embeber([recorte])
        return float((emb @ refs.T).max().item())


# --------------------------------------------------------------------------- #
# Recorte y guardado
# --------------------------------------------------------------------------- #
def caja_con_padding(caja, padding: float, ancho: int, alto: int):
    x0, y0, x1, y1 = caja
    pw, ph = (x1 - x0) * padding, (y1 - y0) * padding
    return (
        max(0, int(x0 - pw)), max(0, int(y0 - ph)),
        min(ancho, int(x1 + pw)), min(alto, int(y1 + ph)),
    )


def construir_recorte(imagen: Image.Image, det: Deteccion, padding: float, fondo: str) -> Image.Image:
    ancho, alto = imagen.size
    x0, y0, x1, y1 = caja_con_padding(det.caja, padding, ancho, alto)
    region = imagen.crop((x0, y0, x1, y1))
    if fondo == "original":
        return region

    mascara = det.mascara[y0:y1, x0:x1]
    alfa = Image.fromarray((mascara * 255).astype(np.uint8), mode="L")
    if fondo == "transparente":
        rgba = region.convert("RGBA")
        rgba.putalpha(alfa)
        return rgba
    color = (0, 0, 0) if fondo == "negro" else (255, 255, 255)
    lienzo = Image.new("RGB", region.size, color)
    lienzo.paste(region, (0, 0), alfa)
    return lienzo


def cuadrar(recorte: Image.Image, tamano: int, fondo: str) -> Image.Image:
    """Letterbox: escala manteniendo proporción y rellena a un cuadrado."""
    if recorte.mode == "RGBA":
        lienzo = Image.new("RGBA", (tamano, tamano), (0, 0, 0, 0))
    else:
        color = (255, 255, 255) if fondo == "blanco" else (0, 0, 0)
        lienzo = Image.new("RGB", (tamano, tamano), color)
    r = recorte.copy()
    r.thumbnail((tamano, tamano), Image.Resampling.LANCZOS)
    lienzo.paste(r, ((tamano - r.width) // 2, (tamano - r.height) // 2))
    return lienzo


def mascara_a_poligonos_yolo(mascara: np.ndarray, ancho: int, alto: int, area_min_px: int = 20):
    """Contornos externos de la máscara en coordenadas normalizadas (YOLO-seg)."""
    import cv2

    contornos, _ = cv2.findContours(
        mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    poligonos = []
    for c in contornos:
        if cv2.contourArea(c) < area_min_px or len(c) < 3:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        pts = []
        for x, y in c:
            pts += [x / ancho, y / alto]
        poligonos.append(pts)
    return poligonos


def generar_mosaico(rutas: list[Path], destino: Path, lado: int = 128, columnas: int = 10, maximo: int = 100):
    rutas = rutas[:maximo]
    if not rutas:
        return
    filas = (len(rutas) + columnas - 1) // columnas
    hoja = Image.new("RGB", (columnas * lado, filas * lado), (40, 40, 40))
    for i, ruta in enumerate(rutas):
        im = Image.open(ruta)
        if im.mode == "RGBA":
            fondo = Image.new("RGB", im.size, (128, 128, 128))
            fondo.paste(im, mask=im.split()[3])
            im = fondo
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        x = (i % columnas) * lado + (lado - im.width) // 2
        y = (i // columnas) * lado + (lado - im.height) // 2
        hoja.paste(im, (x, y))
    hoja.save(destino, quality=90)


# --------------------------------------------------------------------------- #
# Filtros de calidad
# --------------------------------------------------------------------------- #
def filtrar_detecciones(dets: list[Deteccion], ancho: int, alto: int, args) -> list[Deteccion]:
    total_px = ancho * alto
    buenas = []
    for d in dets:
        x0, y0, x1, y1 = d.caja
        d.area_frac = float(d.mascara.sum()) / total_px
        if d.area_frac < args.area_min or d.area_frac > args.area_max:
            continue
        if (x1 - x0) < args.lado_min or (y1 - y0) < args.lado_min:
            continue
        if args.sin_bordes:
            m = 2
            if x0 <= m or y0 <= m or x1 >= ancho - m or y1 >= alto - m:
                continue  # objeto probablemente cortado por el borde de la foto
        buenas.append(d)

    # Quitar duplicados de la misma clase (máscaras casi iguales)
    buenas.sort(key=lambda d: d.score, reverse=True)
    unicas: list[Deteccion] = []
    for d in buenas:
        if all(
            u.clase != d.clase or iou_mascaras(u.mascara, d.mascara) < args.iou_duplicado
            for u in unicas
        ):
            unicas.append(d)

    if args.max_por_imagen:
        conteo: dict[str, int] = {}
        limitadas = []
        for d in unicas:
            conteo[d.clase] = conteo.get(d.clase, 0) + 1
            if conteo[d.clase] <= args.max_por_imagen:
                limitadas.append(d)
        unicas = limitadas
    return unicas


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera un dataset de recortes de objetos usando SAM 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--entrada", type=Path, required=True, help="Directorio con imágenes")
    p.add_argument("-o", "--salida", type=Path, required=True, help="Directorio del nuevo dataset")
    p.add_argument("-p", "--prompt", action="append", required=True,
                   help='Prompt de texto, opcionalmente "clase=frase". Repetible para varias clases')
    p.add_argument("--recursivo", action="store_true", help="Buscar imágenes en subcarpetas")
    p.add_argument("--limite", type=int, default=0, help="Procesar solo N imágenes (para calibrar)")

    g = p.add_argument_group("Filtros")
    g.add_argument("--umbral", type=float, default=0.5, help="Confianza mínima de SAM 3")
    g.add_argument("--area-min", type=float, default=0.001, help="Área mínima de la máscara (fracción de la imagen)")
    g.add_argument("--area-max", type=float, default=0.95, help="Área máxima de la máscara (fracción de la imagen)")
    g.add_argument("--lado-min", type=int, default=32, help="Ancho y alto mínimos de la caja en px")
    g.add_argument("--max-por-imagen", type=int, default=0, help="Máximo de objetos por clase por imagen (0 = sin límite)")
    g.add_argument("--iou-duplicado", type=float, default=0.85, help="IoU de máscara a partir del cual se considera duplicado")
    g.add_argument("--sin-bordes", action="store_true", help="Descartar objetos que tocan el borde de la imagen")
    g.add_argument("--referencias", type=Path, default=None, help="Imágenes de referencia del objeto (filtro DINOv2)")
    g.add_argument("--umbral-similitud", type=float, default=0.5, help="Similitud coseno mínima con las referencias")

    s = p.add_argument_group("Salida")
    s.add_argument("--fondo", choices=["original", "transparente", "negro", "blanco"], default="original",
                   help="Qué hacer con el fondo del recorte")
    s.add_argument("--padding", type=float, default=0.05, help="Margen extra alrededor de la caja (fracción)")
    s.add_argument("--tamano", type=int, default=0, help="Redimensionar a cuadrado de N px (0 = tamaño original)")
    s.add_argument("--exportar-yolo", action="store_true", help="Exportar imágenes completas con etiquetas YOLO-seg")
    s.add_argument("--guardar-rechazados", action="store_true",
                   help="Guardar también los recortes descartados por similitud (para revisar)")

    m = p.add_argument_group("Modelo")
    m.add_argument("--checkpoint", type=str, default=None, help="Ruta local a sam3.pt (si no, se descarga de HF)")
    m.add_argument("--dispositivo", choices=["cuda", "cpu"], default=None, help="Por defecto usa cuda si está disponible")
    return p


def main():
    args = construir_parser().parse_args()

    import torch

    dispositivo = args.dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo == "cpu":
        print("AVISO: corriendo en CPU. SAM 3 tiene ~848M parámetros; será lento.", flush=True)

    if not args.entrada.is_dir():
        sys.exit(f"No existe el directorio de entrada: {args.entrada}")
    imagenes = listar_imagenes(args.entrada, args.recursivo)
    if args.limite:
        imagenes = imagenes[: args.limite]
    if not imagenes:
        sys.exit("No encontré imágenes en el directorio de entrada.")

    prompts = [parsear_prompt(t) for t in args.prompt]
    clases = [c for c, _ in prompts]
    if len(set(clases)) != len(clases):
        sys.exit("Hay nombres de clase repetidos en los prompts.")

    ext = ".png" if args.fondo == "transparente" else ".jpg"
    dir_recortes = args.salida / "recortes"
    for c in clases:
        (dir_recortes / c).mkdir(parents=True, exist_ok=True)
    if args.guardar_rechazados:
        for c in clases:
            (args.salida / "rechazados" / c).mkdir(parents=True, exist_ok=True)
    if args.exportar_yolo:
        (args.salida / "yolo" / "images").mkdir(parents=True, exist_ok=True)
        (args.salida / "yolo" / "labels").mkdir(parents=True, exist_ok=True)

    segmentador = SegmentadorSAM3(args.checkpoint, dispositivo, args.umbral)
    filtro = FiltroReferencias(args.referencias, dispositivo) if args.referencias else None

    ruta_csv = args.salida / "metadatos.csv"
    campos = ["archivo_recorte", "clase", "imagen_origen", "score", "x0", "y0", "x1", "y1",
              "area_frac", "similitud", "aceptado"]
    guardados: dict[str, list[Path]] = {c: [] for c in clases}
    stats = {"imagenes": 0, "con_objetos": 0, "errores": 0, "rechazados_sim": 0}

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f_csv:
        escritor = csv.DictWriter(f_csv, fieldnames=campos)
        escritor.writeheader()

        for ruta in barra_progreso(imagenes, total=len(imagenes), desc="Segmentando"):
            stats["imagenes"] += 1
            try:
                imagen = abrir_imagen(ruta)
                ancho, alto = imagen.size
                dets = segmentador.detectar(imagen, prompts)
                dets = filtrar_detecciones(dets, ancho, alto, args)
            except Exception as e:  # una imagen corrupta no debe tumbar todo el proceso
                stats["errores"] += 1
                print(f"\n[ERROR] {ruta.name}: {e}", flush=True)
                continue

            nombre_base = ruta.relative_to(args.entrada).with_suffix("").as_posix().replace("/", "__")
            aceptadas = []
            contador: dict[str, int] = {}
            for d in dets:
                k = contador.get(d.clase, 0)
                contador[d.clase] = k + 1
                recorte = construir_recorte(imagen, d, args.padding, args.fondo)

                aceptado = True
                if filtro is not None:
                    recorte_ref = recorte if args.fondo == "original" else construir_recorte(imagen, d, args.padding, "original")
                    d.similitud = filtro.similitud(recorte_ref, d.clase)
                    aceptado = d.similitud >= args.umbral_similitud

                if args.tamano:
                    recorte = cuadrar(recorte, args.tamano, args.fondo)

                nombre = f"{nombre_base}_{d.clase}_{k:02d}{ext}"
                if aceptado:
                    destino = dir_recortes / d.clase / nombre
                    aceptadas.append(d)
                elif args.guardar_rechazados:
                    destino = args.salida / "rechazados" / d.clase / nombre
                else:
                    destino = None
                if not aceptado:
                    stats["rechazados_sim"] += 1

                if destino is not None:
                    if ext == ".jpg":
                        recorte.convert("RGB").save(destino, quality=95)
                    else:
                        recorte.save(destino)
                    if aceptado:
                        guardados[d.clase].append(destino)

                escritor.writerow({
                    "archivo_recorte": destino.relative_to(args.salida).as_posix() if destino else "",
                    "clase": d.clase,
                    "imagen_origen": ruta.relative_to(args.entrada).as_posix(),
                    "score": round(d.score, 4),
                    "x0": d.caja[0], "y0": d.caja[1], "x1": d.caja[2], "y1": d.caja[3],
                    "area_frac": round(d.area_frac, 5),
                    "similitud": "" if d.similitud is None else round(d.similitud, 4),
                    "aceptado": int(aceptado),
                })

            if aceptadas:
                stats["con_objetos"] += 1

            if args.exportar_yolo and aceptadas:
                lineas = []
                for d in aceptadas:
                    idx = clases.index(d.clase)
                    for poli in mascara_a_poligonos_yolo(d.mascara, ancho, alto):
                        lineas.append(f"{idx} " + " ".join(f"{v:.6f}" for v in poli))
                if lineas:
                    img_dest = args.salida / "yolo" / "images" / f"{nombre_base}.jpg"
                    imagen.save(img_dest, quality=95)
                    (args.salida / "yolo" / "labels" / f"{nombre_base}.txt").write_text(
                        "\n".join(lineas) + "\n", encoding="utf-8"
                    )

    # Archivos auxiliares
    for c, rutas in guardados.items():
        generar_mosaico(rutas, args.salida / f"mosaico_{c}.jpg")

    if args.exportar_yolo:
        yaml = ["path: .", "train: images", "val: images", "names:"]
        yaml += [f"  {i}: {c}" for i, c in enumerate(clases)]
        (args.salida / "yolo" / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    resumen = {
        "prompts": {c: f for c, f in prompts},
        "parametros": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "estadisticas": {**stats, "recortes_por_clase": {c: len(r) for c, r in guardados.items()}},
    }
    (args.salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n==== Resumen ====")
    print(f"Imágenes procesadas : {stats['imagenes']}  (errores: {stats['errores']})")
    print(f"Imágenes con objetos: {stats['con_objetos']}")
    for c, r in guardados.items():
        print(f"  {c:<20} {len(r)} recortes")
    if filtro is not None:
        print(f"Rechazados por similitud: {stats['rechazados_sim']}")
    print(f"Dataset en: {args.salida.resolve()}")
    print("Revisa los mosaicos y metadatos.csv antes de entrenar.")


if __name__ == "__main__":
    main()
