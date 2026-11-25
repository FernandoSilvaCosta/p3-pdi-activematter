"""
Mask2Former (Transformer com previsão de máscaras) via Hugging Face `transformers`, backbone Swin-T.

- O backbone Swin-T pode partir dos pesos do ImageNet (`pretrained="microsoft/swin-tiny-patch4-window7-224"`);
  decodificador de pixels e decodificador Transformer são inicializados ao acaso. Com `pretrained=None`,
  tudo é aleatório. A primeira camada é adaptada de 3 para 10 canais (média dos pesos RGB).
- Tarefa binária: 1 classe ("alta concentração") + "sem objeto". Poucas consultas (num_queries=20).
- Perda: a própria do Mask2Former (casamento húngaro + entropia cruzada + BCE/Dice nas máscaras), com a
  máscara de primeiro plano de cada imagem como único segmento da classe 0 (vazio se não houver primeiro plano).
- Probabilidade de primeiro plano (inferência semântica padrão do Mask2Former):
      P(fg)(x,y) = soma_q  P(classe fg | consulta q) * sigmoide(máscara_q(x,y)),   limitada a [0, 1].
Contrato com o training.py: training_loss(x, y) e predict_proba(x).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import adapt_conv_in_weight, replace_first_conv


def _load_swin_imagenet(encoder, name, in_channels):
    from transformers import SwinModel
    src = SwinModel.from_pretrained(name).state_dict()
    key = "embeddings.patch_embeddings.projection.weight"
    src[key] = adapt_conv_in_weight(src[key], in_channels)
    replace_first_conv(encoder, in_channels)
    target = encoder.swin if hasattr(encoder, "swin") else encoder        # SwinBackbone guarda o SwinModel em .swin
    own = target.state_dict()
    matched = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    target.load_state_dict(matched, strict=False)
    frac = sum(v.numel() for v in matched.values()) / sum(v.numel() for v in own.values())
    if frac < 0.9:
        raise RuntimeError(f"Só {100 * frac:.0f}% dos pesos do backbone foram carregados de '{name}'. "
                           "Os nomes dos módulos devem ter mudado nesta versão do transformers.")
    return frac


class Mask2Former(nn.Module):
    def __init__(self, in_channels=10, num_queries=20, pretrained="microsoft/swin-tiny-patch4-window7-224",
                 drop_path_rate=0.1):
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig
        backbone = SwinConfig(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                              drop_path_rate=drop_path_rate, out_features=["stage1", "stage2", "stage3", "stage4"])
        cfg = Mask2FormerConfig(backbone_config=backbone, num_labels=1, num_queries=num_queries)
        self.model = Mask2FormerForUniversalSegmentation(cfg)
        encoder = self.model.model.pixel_level_module.encoder
        self.loaded_fraction = None
        if pretrained:
            self.loaded_fraction = _load_swin_imagenet(encoder, pretrained, in_channels)
        else:
            replace_first_conv(encoder, in_channels)
        self.model.config.backbone_config.num_channels = in_channels

    def _outputs_to_proba(self, out, size):
        cls = out.class_queries_logits.float().softmax(dim=-1)[..., :-1]              # (B, Q, 1)
        msk = F.interpolate(out.masks_queries_logits.float(), size=size, mode="bilinear",
                            align_corners=False).sigmoid()                           # (B, Q, H, W)
        return torch.einsum("bqc,bqhw->bchw", cls, msk).clamp(0.0, 1.0)              # (B, 1, H, W)

    def predict_proba(self, x):
        return self._outputs_to_proba(self.model(pixel_values=x), x.shape[-2:])

    def forward(self, x):
        return self.predict_proba(x)

    def training_loss(self, x, y):
        mask_labels, class_labels = [], []
        for yi in y:                                                                  # yi: (1, H, W)
            has_fg = bool(yi.sum() > 0)
            mask_labels.append(yi.float() if has_fg else yi[:0].float())
            class_labels.append(torch.zeros(1 if has_fg else 0, dtype=torch.long, device=y.device))
        return self.model(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels).loss
