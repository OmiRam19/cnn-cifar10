"""Pregunta 3: arquitectura CNN en PyTorch y validacion de shapes del tensor 4D."""
import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    """(B,3,32,32) -> (B,32,16,16) -> (B,64,8,8) -> (B,128,4,4) -> (B,2048) -> (B,10)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, verbose=False):
        def log(name, t):
            if verbose:
                print(f"{name:<24}-> {tuple(t.shape)}")
        log("Entrada", x)
        x = self.block1(x); log("Bloque 1 (Conv+ReLU+MP)", x)
        x = self.block2(x); log("Bloque 2 (Conv+ReLU+MP)", x)
        x = self.block3(x); log("Bloque 3 (Conv+ReLU+MP)", x)
        x = self.flatten(x); log("Flatten", x)
        x = self.classifier(x); log("Linear -> logits", x)
        return x


def validate_shapes(batch_size=8):
    model = CNN()
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        out = model(x, verbose=True)
    assert x.ndim == 4, "La entrada debe ser 4D (B,C,H,W)"
    assert out.shape == (batch_size, 10), f"Shape de salida inesperado: {tuple(out.shape)}"
    print("OK: el modelo procesa tensores 4D sin Shape Mismatch.")
    print(f"Parametros entrenables: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    validate_shapes()
