import os
import torch
from torch import nn
from transformers import AutoModel

os.environ["TORCH_ONNX_USE_EXPERIMENTAL_EXPORTER"] = "0"

MODEL_ID = "google/siglip-base-patch16-384"
OUT_PATH = "models/siglip_image_encoder.onnx"

print("[INFO] Chargement du modèle HF…")
base_model = AutoModel.from_pretrained(MODEL_ID)
base_model.eval()

class SiglipImageEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, 384, 384] (normalisés comme dans le processor HF)
        Retourne un embedding [B, D] (pooler_output).
        """
        vision_outputs = self.m.vision_model(pixel_values=pixel_values)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            pooled = vision_outputs.pooler_output  # [B, D]
        else:
            pooled = vision_outputs.last_hidden_state[:, 0, :]  # [B, D]
        return pooled

print("[INFO] Création du wrapper SigLIP image encoder…")
model = SiglipImageEncoder(base_model)
model.eval()

dummy = torch.randn(1, 3, 384, 384, dtype=torch.float32)

print("[INFO] Export ONNX…")
torch.onnx.export(
    model,
    dummy,
    OUT_PATH,
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "image_embeds": {0: "batch"},
    },
    opset_version=17,
)

print(f"[INFO] Export terminé : {OUT_PATH}")