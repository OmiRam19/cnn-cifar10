"""Pregunta 1: carga de CIFAR-10, visualizacion, transformaciones y particion estratificada."""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 69  # ultimos dos digitos del codigo de matricula 022100269G
CLASSES = ["avion", "automovil", "pajaro", "gato", "ciervo",
           "perro", "rana", "caballo", "barco", "camion"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def get_datasets(val_size=0.1, seed=SEED, data_dir=DATA_DIR):
    """Devuelve (train, val, test). Particion estratificada del train original."""
    set_seed(seed)
    train_tf, eval_tf = get_transforms()
    full_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    full_eval = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    idx = np.arange(len(full_aug))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=full_aug.targets,
        shuffle=True, random_state=seed)
    # entrenamiento con augmentation; validacion sin augmentation
    train = Subset(full_aug, train_idx)
    val = Subset(full_eval, val_idx)
    print(f"Train: {len(train)} | Validacion: {len(val)} | Test: {len(test)}")
    return train, val, test


def get_dataloaders(batch_size=128, val_size=0.1, seed=SEED, num_workers=0):
    train, val, test = get_datasets(val_size, seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size, shuffle=True, generator=g, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=num_workers)
    xb, yb = next(iter(train_loader))
    print(f"Batch de entrenamiento: X {tuple(xb.shape)} | y {tuple(yb.shape)}")
    return train_loader, val_loader, test_loader


def show_samples(n=16, seed=SEED, save_path=None, data_dir=DATA_DIR):
    """Cuadricula con n imagenes del dataset y su etiqueta real."""
    ds = datasets.CIFAR10(data_dir, train=True, download=True)
    rng = np.random.RandomState(seed)
    picks = rng.choice(len(ds), n, replace=False)
    cols = 8 if n >= 16 else 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    for ax, i in zip(np.array(axes).ravel(), picks):
        img, label = ds[i]
        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"CIFAR-10: muestras (semilla {seed})")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    get_dataloaders()
    show_samples(16, save_path=os.path.join(OUT_DIR, "samples.png"))
