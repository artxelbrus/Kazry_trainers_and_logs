import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np
import torchvision.transforms.functional as VF
from PIL import Image
import torch.optim as optim
import io
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
import triton
import triton.language as tl
import math
import loss_landscapes
import matplotlib.pyplot as plt
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
gen = torch.Generator()
gen.manual_seed(0)
@triton.jit
def kazry_forward(
    x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr, N):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr+offs, mask=mask).to(tl.float32)
    exp = tl.exp(x)
    scale = tl.where(x >= 0.0, 1.0, exp)
    x = x * scale
    tl.store(y_ptr+offs, x.to(tl.bfloat16), mask=mask)
@triton.jit
def kazry_backward(
    x_ptr, grad_ptr, grad_out_ptr, BLOCK_SIZE: tl.constexpr, N):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr+offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr+offs, mask=mask).to(tl.float32)
    grad = grad * tl.where(x >= 0.0, 1.0, tl.exp(x) * (1 + x))
    tl.store(grad_out_ptr+offs, grad.to(tl.bfloat16), mask=mask)
class KazryTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        BLOCK_SIZE=512
        N = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
        kazry_forward[grid](x_ptr=x, y_ptr=y, BLOCK_SIZE=BLOCK_SIZE, N=N)
        ctx.save_for_backward(x)
        return y
    @staticmethod
    def backward(ctx, grad):
        BLOCK_SIZE=1024
        x = ctx.saved_tensors[0]
        N = grad.numel()
        grad_out = torch.empty_like(grad)
        grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
        kazry_backward[grid](x_ptr=x, grad_ptr=grad, grad_out_ptr=grad_out, BLOCK_SIZE=BLOCK_SIZE, N=N)
        return grad_out
class Kazry(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return KazryTriton.apply(x)
class TransformerLayer(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.mlp_in = nn.Linear(dim, dim*3, bias=False)
        self.mlp_in_gate = nn.Sequential(
            nn.Linear(dim, dim*3, bias=False),
            activation()
        )
        self.mlp_out = nn.Linear(dim*3, dim, bias=False)
        self.pre_att_norm = nn.RMSNorm(dim)
        self.pre_mlp_norm = nn.RMSNorm(dim)
    def forward(self, x):
        res_x = x
        x = self.pre_att_norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.out_proj(x) + res_x
        res_x = x
        x = self.pre_mlp_norm(x)
        x = self.mlp_in(x) * self.mlp_in_gate(x)
        x = self.mlp_out(x)
        return x + res_x
class Model(nn.Module):
    def __init__(self, dim, activation, layers, num_classes):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(dim//4, dim),
            activation()
        )
        nn.init.normal_(self.emb[0].weight, mean=0, std=0.02, generator=gen)
        self.layers_list = nn.ModuleList()
        for i in range(layers):
            layer = TransformerLayer(dim, activation)
            for layer_name in layer.modules():
                if isinstance(layer_name, nn.Linear):
                    nn.init.normal_(layer_name.weight, mean=0, std=0.02, generator=gen)
            self.layers_list.append(layer)
        self.unemb = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.unemb.weight, mean=0, std=0.02, generator=gen)
        self.layers = layers
    def forward(self, x: torch.Tensor):
        x = self.emb(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).contiguous().transpose(1, 2)
        for i in range(self.layers):
            x = self.layers_list[i](x)
        x = x.mean(dim=1) 
        return self.unemb(x)
class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = pd.read_parquet(path)
    def __len__(self):
        return len(self.data["fine_label"])
    def __getitem__(self, index):
        with Image.open(io.BytesIO(self.data["img"].iloc[index]["bytes"])) as file:
            picture = VF.pil_to_tensor(file).to(torch.bfloat16)
        target = torch.tensor(self.data["fine_label"].iloc[index], dtype=torch.long)
        return target, picture
def train(
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-3,
    epochs: int = 100,
    shuffle: bool = True,
    num_workers: int = 10,
    train_dataset_path: str | None = None,
    val_dataset_path: str | None = None,
    checkpoint_dir: str | None = None,
    log_dir: str | None = None,
    activation = Kazry,
    dim: int = 64,
    layers: int = 10,
    num_classes: int = 100
):
    dataset_train = MyDataset(path=train_dataset_path)
    dataset_val = MyDataset(path=val_dataset_path)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True, generator=gen, persistent_workers=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True, generator=gen, persistent_workers=True)
    model = Model(dim=dim, activation=activation, layers=layers, num_classes=num_classes)
    model = model.to("cuda", torch.bfloat16)
    model = torch.compile(model=model, fullgraph=True, dynamic=False, mode="max-autotune")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(dataloader_train)*epochs, eta_min=learning_rate/10)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(epochs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        losses_train = []
        losses_val = []
        for batch_index, (target, picture) in enumerate(dataloader_train):
            target = target.to("cuda")
            picture = picture.to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                out = model(picture)
                loss = criterion(out, target)
                losses_train.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sch.step()
        model.eval()
        for batch_index, (target, picture) in enumerate(dataloader_val):
            target = target.to("cuda")
            picture = picture.to("cuda")
            with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
                out = model(picture)
                loss = criterion(out, target)
                losses_val.append(loss.item())
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        print(f"mean train loss: {sum(losses_train)/len(losses_train)}, mean val loss: {sum(losses_val)/len(losses_val)}, time for epoch {end_time-start_time}")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}.pth"))
        writer.add_scalar("losses/train loss", sum(losses_train)/len(losses_train), global_step=epoch)
        writer.add_scalar("losses/val loss", sum(losses_val)/len(losses_val), global_step=epoch)
        writer.add_scalar("time/time to epoch", end_time-start_time, global_step=epoch)
if __name__ == "__main__":
    seed_everything()
    train()
