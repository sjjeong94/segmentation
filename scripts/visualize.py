import cv2
import numpy as np
import videocv
import torch
import torchvision
import segmentation_models_pytorch as smp


from segmentation.data import dataset, transforms
from segmentation.utils import metrics


class Module:
    def __init__(self, model_path, num_classes=5):

        device = device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone = "efficientnet-b0"  # 'efficientnet-b0'
        net = smp.Unet(backbone, classes=num_classes)

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)
        net = net.to(device)
        net = net.eval()

        T_Compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.device = device
        self.net = net
        self.transform = T_Compose

    @torch.inference_mode()
    def __call__(self, image):
        x = self.transform(image).unsqueeze(0)
        x = x.to(self.device)
        out = self.net(x)

        y = torch.argmax(out, 1)
        y = y.cpu().numpy().squeeze()

        p = torch.softmax(out, dim=1)
        entropy = -torch.sum(p * torch.log(p), dim=1)
        entropy = entropy.cpu().numpy().squeeze()

        return y, entropy


def get_maximum_entropy(n):
    p = np.ones(n, dtype=float) / n
    maximum_entropy = -np.sum(p * np.log(p))
    return maximum_entropy


def visualze_video(
    model_path,
    num_classes=5,
    video_path="./videos/test.mp4",
    save_path="./videos/segmentation.mp4",
    size=(1024, 576),
):
    print(model_path)
    module = Module(model_path, num_classes)

    video = videocv.Video(video_path)
    writer = videocv.Writer(save_path, video.fps, (size[0] * 2, size[1] * 2))

    me = get_maximum_entropy(num_classes)

    idx = 0
    while video():
        idx += 1
        print("%d / %d" % (idx, video.frame_count))

        image = cv2.resize(video.frame, size, interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        y, entropy = module(x)

        mask = dataset.Comma10k.get_color(y)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        edge = cv2.Laplacian(y.astype(np.uint8), -1)
        edge = edge != 0

        overlap = image // 2 + mask // 2
        overlap[edge] = 255

        entropy = (np.clip(entropy / me, 0, 1) * 255).astype(np.uint8)
        entropy = cv2.applyColorMap(entropy, cv2.COLORMAP_INFERNO)

        boundary = np.zeros(image.shape, np.uint8)
        boundary[edge] = (0, 255, 255)

        view_top = np.concatenate([image, mask], axis=1)
        view_bot = np.concatenate([overlap, entropy], axis=1)
        view = np.concatenate([view_top, view_bot])
        writer(view)

        cv2.imshow("view", view)
        key = cv2.waitKey(1)
        if key == 27:
            break


def visualize_eval(
    data_root="./data/comma10k",
    model_path="./logs/comma10k/test2/models/model_050.pth",
    size=(640, 480),
):

    module = Module(model_path)

    dataset = dataset.Comma10k(data_root, "val", transforms.Resize(size))

    idx = 0
    while True:
        image, label = dataset[idx]

        y, _ = module(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(dataset.get_color(label), cv2.COLOR_RGB2BGR)
        pred = cv2.cvtColor(dataset.get_color(y), cv2.COLOR_RGB2BGR)

        results = metrics.calculate_results(label, y, 5)
        print("%8s %8s %8s %8s %8s" % ("class", "mask", "pred", "inter", "union"))
        for c, r in enumerate(results):
            print(
                "%8d %8d %8d %8d %8d"
                % (c, r["mask"], r["pred"], r["inter"], r["union"])
            )
        print()

        # error = np.zeros(image.shape, np.uint8)
        # error[y != label] = (0, 255, 255)

        error = mask.copy()
        error[y == label] = 0

        view_top = np.concatenate([image, error], axis=1)
        view_bot = np.concatenate([mask, pred], axis=1)
        view = np.concatenate([view_top, view_bot])

        cv2.imshow("view", view)

        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord("q"):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset) - 1
        elif idx >= len(dataset):
            idx = 0


if __name__ == "__main__":
    visualze_video(
        model_path="./logs/comma10k/models/220902.pt",
        video_path="D:/videos/test.mp4",
        save_path="D:/videos/segmentation.mp4",
    )
    # visualize_eval()
