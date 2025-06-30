import os
import gdown
import torch
from typing import Dict, Any
from segmentation_models_pytorch import Unet


class UnetEfficientNetB0(Unet):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__(encoder_name="efficientnet-b0", classes=num_classes)
        self.num_classes = num_classes

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
        file_id = "1GOVxoZhdxg898G4y4JXA2oj-V_lg-BYe"
        path = os.path.join(model_dir, file_name)
        if not os.path.exists(path):
            os.makedirs(model_dir, exist_ok=True)
            gdown.download(id=file_id, output=path, quiet=False)
        model = cls()
        pretrained = torch.load(path, map_location="cpu")
        model.load_state_dict(pretrained, strict=False)
        return model