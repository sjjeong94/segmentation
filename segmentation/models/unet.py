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
