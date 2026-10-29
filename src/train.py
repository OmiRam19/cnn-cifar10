"""Pregunta 4: entrenamiento (CrossEntropyLoss + Adam), metricas y artefactos graficos."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import CLASSES, OUT_DIR, SEED, get_dataloaders, set_seed
from model import CNN, validate_shapes

EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 128


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_curves(h, path):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, h["train_loss"], label="Train")
    ax[0].plot(ep, h["val_loss"], label="Validacion")
    ax[0].set(title="Loss", xlabel="Epoca", ylabel="CrossEntropy")
    ax[1].plot(ep, h["train_acc"], label="Train")
    ax[1].plot(ep, h["val_acc"], label="Validacion")
    ax[1].set(title="Accuracy", xlabel="Epoca", ylabel="Accuracy")
    for a in ax:
        a.legend()
        a.grid(alpha=.3)
    fig.suptitle(f"Curvas de aprendizaje (semilla {SEED})")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Prediccion", ylabel="Real", title="Matriz de confusion (test)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Semilla: {SEED}")
    validate_shapes()
    set_seed(SEED)  # la validacion de shapes consume RNG: se reinicia antes de crear el modelo

    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    h = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val, best_path = 0.0, os.path.join(OUT_DIR, "best_model.pt")
    for epoch in range(EPOCHS):
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va = run_epoch(model, val_loader, criterion, device)
        for k, v in zip(h, [tl, ta, vl, va]):
            h[k].append(v)
        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)
        print(f"Epoca {epoch+1:02d}/{EPOCHS} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f}", flush=True)

    plot_curves(h, os.path.join(OUT_DIR, "loss_accuracy.png"))

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    cm = confusion_matrix(labels, preds)
    plot_confusion(cm, os.path.join(OUT_DIR, "confusion_matrix.png"))
    print(f"\nTEST: loss {test_loss:.4f} | accuracy {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"seed": SEED, "epochs": EPOCHS, "test_loss": test_loss, "test_acc": test_acc,
                   "best_val_acc": best_val, "history": h}, f, indent=2)


if __name__ == "__main__":
    main()
