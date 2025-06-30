from segmentation.data import transforms
from segmentation.data.dataset import Comma10k
from segmentation.engine.trainer import Trainer
from segmentation.models.unet import UnetEfficientNetB0

T_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(416, 416), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
dataset = Comma10k("dataset/comma10k", split="train", transform=T_train)
model = UnetEfficientNetB0(num_classes=5)
trainer = Trainer(model, dataset, batch_size=32, max_iters=500000)
trainer.run()
