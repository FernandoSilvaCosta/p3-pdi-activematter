"""
SegFormer (Transformer hierárquico) via Hugging Face `transformers`.

Variante MiT-B0 (a menor). Com `pretrained="nvidia/mit-b0"` o codificador parte dos pesos do ImageNet e o
decodificador (cabeça) é inicializado ao acaso; com `pretrained=None` tudo é inicializado ao acaso.
A primeira camada (patch embedding) é adaptada de 3 para 10 canais pela média dos pesos RGB.
Saída: LOGITS (B, 1, H, W): os logits nativos (1/4 da resolução) são ampliados por interpolação bilinear.
"""
import torch.nn as nn
import torch.nn.functional as F

from .common import replace_first_conv


class SegFormer(nn.Module):
    def __init__(self, in_channels=10, pretrained="nvidia/mit-b0"):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(pretrained, num_labels=1)
        else:
            model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=1))     # padrão = MiT-B0
        self.first_conv = replace_first_conv(model.segformer, in_channels)      # robusto à versão do transformers
        model.config.num_channels = in_channels
        self.model = model

    def forward(self, x):
        logits = self.model(pixel_values=x).logits                      # (B, 1, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
