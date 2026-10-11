from ultralytics import YOLO

model = YOLO("yolo/weights/best.pt")

model.export(
    format="onnx",
    opset=17,
    dynamic=False,
    simplify=True
)