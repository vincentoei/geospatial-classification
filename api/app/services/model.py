"""Model inference service."""

import logging
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load project root .env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# Ensure we can import the src package
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from geospatial_classification.data.transforms import (  # isort:skip  # noqa: E402
    get_inference_transform,
)

from geospatial_classification.models import CNN_ViT_Hybrid  # isort:skip  # noqa: E402


class ModelService:
    """Singleton-like service that loads the CNN-ViT model on GPU."""

    _instance: "ModelService | None" = None

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.device = os.getenv("DEVICE", "cuda")
        self.model_path = os.getenv("MODEL_PATH", "checkpoints/hybrid_best.pth")
        self.model: CNN_ViT_Hybrid | None = None
        self.transform = get_inference_transform((64, 64))
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        logger.info("Loading model from %s onto %s", self.model_path, self.device)
        self.model = CNN_ViT_Hybrid(
            num_classes=2,
            embed_dim=768,
            depth=3,
            heads=6,
        ).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()
        logger.info(
            "Model loaded successfully (%s parameters)",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        # Warm-up inference
        dummy = torch.randn(1, 3, 64, 64, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        logger.info("GPU warm-up complete")

    def predict(self, image_tensor: torch.Tensor) -> dict:
        """Run inference on a single pre-processed image tensor.

        Args:
            image_tensor: (3, 64, 64) or (1, 3, 64, 64) tensor

        Returns:
            Dictionary with prediction, confidence, and probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)

        prob_values = probs.cpu().numpy()[0]
        pred_idx = int(prob_values.argmax())
        confidence = float(prob_values[pred_idx])

        labels = ["non-agri", "agri"]
        prediction = labels[pred_idx]

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "non-agri": float(prob_values[0]),
                "agri": float(prob_values[1]),
            },
        }

    def predict_batch(self, image_tensors: torch.Tensor) -> list[dict]:
        """Run inference on a batch of pre-processed image tensors.

        Args:
            image_tensors: (N, 3, 64, 64) tensor.

        Returns:
            List of dictionaries with prediction, confidence, and probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if image_tensors.dim() == 3:
            image_tensors = image_tensors.unsqueeze(0)

        image_tensors = image_tensors.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensors)
            probs = torch.softmax(logits, dim=1)

        prob_values = probs.cpu().numpy()
        results = []
        for i in range(len(prob_values)):
            pred_idx = int(prob_values[i].argmax())
            confidence = float(prob_values[i][pred_idx])
            labels = ["non-agri", "agri"]
            results.append(
                {
                    "prediction": labels[pred_idx],
                    "confidence": confidence,
                    "probabilities": {
                        "non-agri": float(prob_values[i][0]),
                        "agri": float(prob_values[i][1]),
                    },
                }
            )

        return results
