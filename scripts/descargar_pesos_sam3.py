#!/usr/bin/env python3
"""
descargar_pesos_sam3.py
=======================

Descarga los pesos de SAM 3 y deja la carpeta lista para usarse, incluyendo
una copia de la licencia (requisito si vas a repartir el archivo a terceros).

Dos modos:

1. Hugging Face (necesita cuenta, acceso aprobado y `hf auth login`):

       python descargar_pesos_sam3.py -d ./pesos
       python descargar_pesos_sam3.py -d ./pesos --version sam3.1

2. Espejo propio (para el laboratorio: un servidor HTTP, un NAS, una URL
   interna). No necesita token ni cuenta:

       python descargar_pesos_sam3.py -d ./pesos --url http://192.168.1.50:8000/sam3.pt

Verificación de integridad:

       # en tu máquina, después de bajarlo de HF
       python descargar_pesos_sam3.py -d ./pesos --solo-hash

       # en las máquinas del laboratorio
       python descargar_pesos_sam3.py -d ./pesos --url http://... --sha256 <hash>

Para servir el archivo en la red del aula:

       cd pesos && python -m http.server 8000
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

VERSIONES = {
    "sam3": {
        "repo": "facebook/sam3",
        "checkpoint": "sam3.pt",
        "extras": ["config.json"],
    },
    "sam3.1": {
        "repo": "facebook/sam3.1",
        "checkpoint": "sam3.1_multiplex.pt",
        "extras": ["config.json"],
    },
}

LICENCIA_URL = "https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE"
TAMANO_MINIMO = 500 * 1024 * 1024  # un .pt real pesa GB; menos que esto es un error disfrazado

LEEME = """Pesos de SAM 3 (Meta / Meta Superintelligence Labs)

Origen: {origen}
Archivo: {archivo}
SHA-256: {hash}

El uso de estos pesos está sujeto a la SAM License, incluida en el archivo
LICENSE de esta misma carpeta. Al usarlos aceptas esos términos.

Modelo: https://huggingface.co/facebook/sam3
Código: https://github.com/facebookresearch/sam3
"""


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def formato_bytes(n: float) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if n < 1024 or unidad == "GB":
            return f"{n:.1f} {unidad}"
        n /= 1024
    return f"{n:.1f} GB"


def sha256_de(ruta: Path, mostrar: bool = True) -> str:
    h = hashlib.sha256()
    total = ruta.stat().st_size
    leido = 0
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloque)
            leido += len(bloque)
            if mostrar and total:
                pct = 100 * leido / total
                print(f"\r  calculando sha256... {pct:5.1f}%", end="", flush=True)
    if mostrar:
        print("\r  calculando sha256... listo      ", flush=True)
    return h.hexdigest()


def descargar_url(url: str, destino: Path, timeout: int = 60) -> None:
    """Descarga con barra de progreso a un archivo temporal y luego renombra."""
    temporal = destino.with_suffix(destino.suffix + ".parcial")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as respuesta:
            total = int(respuesta.headers.get("Content-Length") or 0)
            tipo = (respuesta.headers.get("Content-Type") or "").lower()
            if "text/html" in tipo:
                raise SystemExit(
                    "El servidor devolvió una página HTML, no un archivo.\n"
                    "Pasa con links de Drive o Dropbox que piden confirmación: usa el\n"
                    "link de descarga directa o baja el archivo desde el navegador."
                )
            leido = 0
            with open(temporal, "wb") as f:
                while bloque := respuesta.read(1024 * 1024):
                    f.write(bloque)
                    leido += len(bloque)
                    if total:
                        pct = 100 * leido / total
                        print(f"\r  {formato_bytes(leido)} / {formato_bytes(total)}  ({pct:4.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r  {formato_bytes(leido)}", end="", flush=True)
            print(flush=True)
    except urllib.error.HTTPError as e:
        temporal.unlink(missing_ok=True)
        raise SystemExit(f"Error HTTP {e.code} al descargar {url}")
    except urllib.error.URLError as e:
        temporal.unlink(missing_ok=True)
        raise SystemExit(f"No pude conectar con {url}: {e.reason}")
    temporal.replace(destino)


def descargar_de_hf(version: str, nombre: str, token: str | None) -> Path:
    """Descarga un archivo del repo restringido y devuelve su ruta en la caché."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Falta huggingface_hub. Instálalo con:\n  pip install -U huggingface_hub")

    repo = VERSIONES[version]["repo"]
    try:
        return Path(hf_hub_download(repo_id=repo, filename=nombre, token=token))
    except Exception as e:
        texto = f"{type(e).__name__}: {e}"
        pistas = []
        if "401" in texto or "authenticated" in texto.lower():
            pistas.append("No estás autenticado. Corre:  hf auth login")
        if "403" in texto or "gated" in texto.lower():
            pistas.append("Tu token no tiene permiso para repos restringidos.\n"
                          "    Si es fine-grained, marca «Read access to contents of all\n"
                          "    public gated repos you can access», o usa un token de tipo Read.")
        if "authorized list" in texto.lower() or "restricted" in texto.lower():
            pistas.append(f"Todavía no te aprueban el acceso en https://huggingface.co/{repo}")
        mensaje = f"Falló la descarga de {nombre} desde {repo}.\n  {texto}"
        if pistas:
            mensaje += "\n\nPosible causa:\n  - " + "\n  - ".join(pistas)
        raise SystemExit(mensaje)


def conseguir_licencia(destino: Path, version: str, token: str | None) -> bool:
    """Deja una copia de la SAM License junto a los pesos. Intenta HF y luego GitHub."""
    salida = destino / "LICENSE"
    if salida.exists():
        return True
    try:
        ruta = descargar_de_hf(version, "LICENSE", token)
        shutil.copy2(ruta, salida)
        return True
    except SystemExit:
        pass
    try:
        descargar_url(LICENCIA_URL, salida)
        return True
    except SystemExit:
        print("AVISO: no pude bajar la licencia automáticamente.")
        print(f"       Descárgala a mano de {LICENCIA_URL} y ponla junto a los pesos.")
        return False


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Descarga los pesos de SAM 3 desde Hugging Face o desde un espejo propio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-d", "--destino", type=Path, default=Path("./pesos"),
                   help="Carpeta donde dejar los archivos")
    p.add_argument("--version", choices=list(VERSIONES), default="sam3",
                   help="Qué checkpoint bajar")
    p.add_argument("--url", type=str, default=None,
                   help="Bajar de esta URL en vez de Hugging Face (espejo local)")
    p.add_argument("--token", type=str, default=None,
                   help="Token de Hugging Face (si no, usa el de 'hf auth login' o HF_TOKEN)")
    p.add_argument("--sha256", type=str, default=None,
                   help="Hash esperado; aborta si no coincide")
    p.add_argument("--solo-hash", action="store_true",
                   help="No descarga nada: solo calcula el sha256 del archivo ya presente")
    p.add_argument("--sin-licencia", action="store_true",
                   help="No descargar la copia de la licencia (no recomendado si vas a repartirlo)")
    p.add_argument("--forzar", action="store_true", help="Volver a descargar aunque ya exista")
    args = p.parse_args()

    nombre = VERSIONES[args.version]["checkpoint"]
    args.destino.mkdir(parents=True, exist_ok=True)
    final = args.destino / nombre

    # Modo solo-hash
    if args.solo_hash:
        if not final.exists():
            sys.exit(f"No existe {final}")
        print(f"{nombre}")
        print(f"  tamaño : {formato_bytes(final.stat().st_size)}")
        print(f"  sha256 : {sha256_de(final)}")
        return

    # Descarga
    if final.exists() and not args.forzar:
        print(f"Ya existe {final} ({formato_bytes(final.stat().st_size)}). Usa --forzar para rebajarlo.")
        origen = "(archivo ya presente)"
    elif args.url:
        print(f"Descargando {nombre} desde {args.url}")
        descargar_url(args.url, final)
        origen = args.url
    else:
        print(f"Descargando {nombre} desde Hugging Face ({VERSIONES[args.version]['repo']})")
        print("La primera vez son varios GB; queda cacheado en ~/.cache/huggingface")
        ruta = descargar_de_hf(args.version, nombre, args.token)
        print("  copiando a la carpeta destino...", flush=True)
        shutil.copy2(ruta, final)
        for extra in VERSIONES[args.version]["extras"]:
            try:
                shutil.copy2(descargar_de_hf(args.version, extra, args.token), args.destino / extra)
            except SystemExit:
                print(f"AVISO: no pude bajar {extra} (normalmente no hace falta para inferencia).")
        origen = f"https://huggingface.co/{VERSIONES[args.version]['repo']}"

    # Validaciones
    tamano = final.stat().st_size
    if tamano < TAMANO_MINIMO:
        sys.exit(f"El archivo pesa solo {formato_bytes(tamano)}. Algo salió mal; bórralo y repite.")

    print(f"Archivo: {final}  ({formato_bytes(tamano)})")
    digest = sha256_de(final)
    print(f"sha256 : {digest}")

    if args.sha256:
        if digest.lower() != args.sha256.lower().strip():
            final.rename(final.with_suffix(final.suffix + ".corrupto"))
            sys.exit("El hash NO coincide. El archivo se renombró a .corrupto; bórralo y vuelve a bajarlo.")
        print("El hash coincide con el esperado.")

    if not args.sin_licencia:
        conseguir_licencia(args.destino, args.version, args.token)

    (args.destino / "LEEME.txt").write_text(
        LEEME.format(origen=origen, archivo=nombre, hash=digest), encoding="utf-8"
    )

    print("\nListo. Para usarlo:")
    print(f"  python sam3_recortar_dataset.py -i fotos/ -o dataset/ -p \"objeto=object\" \\")
    print(f"      --checkpoint {final}")
    print("\nSi vas a repartir este archivo, incluye también el LICENSE de esta carpeta.")


if __name__ == "__main__":
    main()
