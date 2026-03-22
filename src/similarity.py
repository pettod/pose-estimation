import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoV2:
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        x = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**x)
        return F.normalize(out.last_hidden_state[:, 0], dim=-1)

    def embed_bgr(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Embed a BGR uint8 crop (OpenCV) without writing to disk."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            raise ValueError("crop too small for embedding")
        rgb = crop_bgr[:, :, ::-1]  # BGR -> RGB
        img = Image.fromarray(rgb)
        x = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**x)
        return F.normalize(out.last_hidden_state[:, 0], dim=-1)

    def similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        a = emb1.detach() if isinstance(emb1, torch.Tensor) else emb1
        b = emb2.detach() if isinstance(emb2, torch.Tensor) else emb2
        return float(torch.cosine_similarity(a, b).item())
