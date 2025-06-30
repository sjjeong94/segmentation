import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from segmentation.models.unet import UnetEfficientNetB0
from segmentation.data import dataset, transforms
from segmentation.utils.helpers import get_device


def get_transforms(size):
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(model_path, num_classes, device):
    model = UnetEfficientNetB0(num_classes=num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def get_dataloader(dataset_cls, data_root, split, transform, batch_size, num_workers):
    dataset = dataset_cls(data_root, split, transform)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def init_metrics(num_classes):
    return [{"mask": 0, "pred": 0, "inter": 0, "union": 0} for _ in range(num_classes)]


def compute_metrics(pred, mask, pack, num_classes):
    correct = pred == mask
    for c in range(num_classes):
        mask_c = mask == c
        pred_c = pred == c
        inter = torch.sum(correct[mask_c]).item()
        area_mask = mask_c.sum().item()
        area_pred = pred_c.sum().item()
        union = area_mask + area_pred - inter

        pack[c]["mask"] += area_mask
        pack[c]["pred"] += area_pred
        pack[c]["inter"] += inter
        pack[c]["union"] += union


def evaluate_model(net, dataloader, num_classes, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    metrics = init_metrics(num_classes)

    with torch.inference_mode():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            output = net(x)
            loss = criterion(output, y)
            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            compute_metrics(pred, y, metrics, num_classes)

    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics


def print_metrics(metrics):
    acc_list, iou_list, dice_list = [], [], []

    print(
        f"{'class':>8} {'mask':>12} {'pred':>12} {'inter':>12} {'union':>12} {'acc':>8} {'iou':>8} {'dice':>8}"
    )
    for c, m in enumerate(metrics):
        eps = 1e-9
        acc = m["inter"] / (m["mask"] + eps)
        iou = m["inter"] / (m["union"] + eps)
        dice = 2 * m["inter"] / (m["mask"] + m["pred"] + eps)

        acc_list.append(acc)
        iou_list.append(iou)
        dice_list.append(dice)

        print(
            f"{c:8d} {m['mask']:12d} {m['pred']:12d} {m['inter']:12d} {m['union']:12d} {acc:8.6f} {iou:8.6f} {dice:8.6f}"
        )

    print("acc  ->", np.mean(acc_list))
    print("miou ->", np.mean(iou_list))
    print("dice ->", np.mean(dice_list))


def evaluate(
    data_root,
    model_path,
    size=(640, 480),
    batch_size=8,
    num_workers=2,
    dataset_cls=dataset.Comma10k,
    num_classes=5,
):
    device = get_device()
    transform = get_transforms(size)
    dataloader = get_dataloader(
        dataset_cls, data_root, "val", transform, batch_size, num_workers
    )
    model = load_model(model_path, num_classes, device)

    avg_loss, metrics = evaluate_model(model, dataloader, num_classes, device)

    print(f"\nValidation Loss: {avg_loss:.4f}\n")
    print_metrics(metrics)
