"""
U-Net (CNN) treinada do zero.

Codificador com 4 reduções (MaxPool 2x2), gargalo e decodificador com conexões de salto (skip connections).
Blocos de duas convoluções 3x3 + BatchNorm + ReLU. Entrada (B, 10, H, W), saída: LOGITS (B, 1, H, W).
H e W devem ser múltiplos de 2**depth (128 é múltiplo de 16, não precisa de padding).
Com base=32 e depth=4 tem ~7,8 milhões de parâmetros.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=1, base=32, depth=4):
        super().__init__()
        self.depth = depth
        ch = [base * 2 ** i for i in range(depth + 1)]                       # ex.: 32, 64, 128, 256, 512
        self.enc = nn.ModuleList([DoubleConv(in_channels, ch[0])] +
                                 [DoubleConv(ch[i], ch[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        assert h % (2 ** self.depth) == 0 and w % (2 ** self.depth) == 0, \
            f"H e W devem ser múltiplos de {2 ** self.depth} (recebido {h}x{w})"
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()                                                        # gargalo
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([skips.pop(), x], dim=1))
        return self.head(x)
