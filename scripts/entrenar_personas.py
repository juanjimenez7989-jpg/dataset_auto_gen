from ultralytics import YOLO

DATASET = "datos/salida/yolo_personas/yolo/data.yaml"
MODELO_BASE = "yolo26n-seg.pt"


def main() -> None:
    model = YOLO(MODELO_BASE)

    model.train(
        data=DATASET,
        epochs=100,
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        project="entrenamientos",
        name="personas_seg",
        patience=20,
        cache=False,
        plots=True,
        amp=False,
    )


if __name__ == "__main__":
    main()
