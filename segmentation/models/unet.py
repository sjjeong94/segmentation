import os
import gdown
import torch
import torchvision
import numpy as np
from typing import Dict, Any
from segmentation_models_pytorch import Unet
import torchvision.transforms.functional


class UnetEfficientNetB0(Unet):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__(encoder_name="efficientnet-b0", classes=num_classes)
        self.num_classes = num_classes
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = super().forward(x)
        return x

    @torch.inference_mode()
    def inference(self, image: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).get_device()
        device = torch.device("cpu") if device < 0 else device
        image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
        image = image.to(device)
        output = self.forward(image)
        output = output.cpu().numpy()
        torch.cuda.empty_cache()
        return output

    def get_model_config(self) -> Dict[str, Any]:
        return dict(
            encoder="efficientnet-b0",
            decoder="unet",
            num_classes=self.num_classes,
        )

    @classmethod
    def from_pretrained(cls) -> "UnetEfficientNetB0":
        model_dir = "pretrained"
        file_name = "UnetEfficientNetB0.pth"
        file_id = "1WNQvVrivv96LcbQARhiyeHfVS7EzU9hh"
        path = os.path.join(model_dir, file_name)
        if not os.path.exists(path):
            os.makedirs(model_dir, exist_ok=True)
            gdown.download(id=file_id, output=path, quiet=False)
        model = cls()
        pretrained = torch.load(path, map_location="cpu")
        model.load_state_dict(pretrained, strict=False)
        return model
