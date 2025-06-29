# Semantic Segmentation
This is a PyTorch implementation of semantic segmentation models on [Comma10k](https://github.com/commaai/comma10k) and [Cityscapes](https://www.cityscapes-dataset.com/)

## Comma10k
| Date   | miou (1280x960) | miou (640x480) | video                                |
| ------ | --------------- | -------------- | ------------------------------------ |
| 220622 | 89.38           | 89.05          | [Link](https://youtu.be/-xZ5Vsq1JDg) |
| 220902 | 88.39           | 89.79          | [Link](https://youtu.be/l2cX6F_69wI) |

## Cityscapes
| Date   | miou (2048x1024) | miou (1024x512) | miou (512x256) | video                                |
| ------ | ---------------- | --------------- | -------------- | ------------------------------------ |
| 220720 | 47.20            | 58.09           | 52.53          | [Link](https://youtu.be/iUEUOsw3ViQ) |
| 220721 | 58.44            | 60.01           | 50.34          | [Link](https://youtu.be/WyZvsIS7eq8) |
| 220723 | 60.79            | 62.19           | 52.60          | [Link](https://youtu.be/XUA3fDtz4IE) |

## How to use

### Train
```python
from segmentation.data import transforms
from segmentation.data.dataset import Comma10k
from segmentation.engine.trainer import Trainer
from segmentation.models.unet import UnetEfficientNetB0

T_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(640, 640), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
dataset = Comma10k("/path/to/dataset/", split="train", transform=T_train)
model = UnetEfficientNetB0(num_classes=5)
trainer = Trainer(model, dataset, batch_size=32)
trainer.run()
```


### License

MIT
