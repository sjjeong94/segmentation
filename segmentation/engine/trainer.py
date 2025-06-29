import json
import os
import platform
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from segmentation.data.handler import DataHandler
from segmentation.utils.helpers import Logger, get_device


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 0.0002,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        max_iters: int = 500000,
        gamma: float = 0.5,
        milestones: List[int] = [250000, 400000, 450000, 475000],
        loss_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = nn.CrossEntropyLoss(),
        eval_interval: int = 1000,
        ckpt_path: str = "checkpoints",
        bfloat16: bool = True,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.dataset = train_dataset

        self.batch_size = batch_size
        self.num_workers = 0 if platform.system() == "Windows" else num_workers
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.ckpt_path = ckpt_path
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.learning_rate = learning_rate
        self.betas = (beta1, beta2)
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        self.device = get_device()
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported() and bfloat16
            else torch.float32
        )
        self.seed = seed

        self.optimizer = None
        self.scheduler = None
        self.criterion = loss_function

    def run(self) -> None:
        device, dtype = self.device, self.dtype
        print(f"device: {device}  dtype: {dtype}")
        ctx = torch.autocast(device_type=device, dtype=dtype)

        self.data_handler = DataHandler(self.dataset, self.batch_size, self.num_workers)
        self.data_handler.set_seed(self.seed)

        model = self.model.to(device)
        if self.load("latest"):
            print("-> The latest checkpoint was loaded.")

        if self.data_handler.ddp_enabled:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[device], output_device=device)

        if self.data_handler.is_main_process:
            logger = Logger(os.path.join(self.ckpt_path, "train.log"))

        model = model.train()
        while self.data_handler.iterations < self.max_iters:
            x, y = self.data_handler.get_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                out = model(x)
                loss = self.criterion(out, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            iterations = self.data_handler.iterations
            print(f" Iterations = {iterations:<8}", end="\r")
            if (
                iterations % self.eval_interval == 0
                and self.data_handler.is_main_process
            ):
                log = f" Iterations = {iterations:<8}  Loss: {loss:9.6f}"
                logger.info(log)
                self.save("latest")

        self.data_handler.close()

    def build_optimizer(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
        )
        return optimizer, scheduler

    def save(self, file_name: str) -> str:
        os.makedirs(self.ckpt_path, exist_ok=True)
        model_path = os.path.join(self.ckpt_path, file_name + ".model.pth")
        train_path = os.path.join(self.ckpt_path, file_name + ".train.pth")
        torch.save(self.model.state_dict(), model_path)
        train_dict = dict(
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            iteration=self.data_handler.iterations,
        )
        torch.save(train_dict, train_path)
        config = self.model.get_model_config()
        config_path = os.path.join(self.ckpt_path, "params.json")
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)
        return model_path, train_path

    def load(self, file_name: str) -> bool:
        model_path = os.path.join(self.ckpt_path, file_name + ".model.pth")
        train_path = os.path.join(self.ckpt_path, file_name + ".train.pth")
        if not (os.path.isfile(model_path) and os.path.isfile(train_path)):
            self.optimizer, self.scheduler = self.build_optimizer()
            return False
        model_dict = torch.load(model_path, map_location=self.device)
        train_dict = torch.load(train_path, map_location=self.device)
        self.model.load_state_dict(model_dict)
        self.optimizer, self.scheduler = self.build_optimizer()
        self.optimizer.load_state_dict(train_dict["optimizer"])
        self.scheduler.load_state_dict(train_dict["scheduler"])
        self.data_handler.set_iterations(train_dict["iteration"])
        return True
