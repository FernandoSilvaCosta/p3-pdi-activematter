"""
Teste de aceite dos modelos (U-Net, SegFormer, Mask2Former).

Para cada arquitetura verifica:
  1. a saída tem forma (B, 1, 128, 128) e as probabilidades ficam em [0, 1];
  2. a primeira camada recebe 10 canais;
  3. a perda é finita e o gradiente chega aos parâmetros (também para um quadro SEM primeiro plano);
  4. conta parâmetros e mede o tempo de uma passada.

Por padrão os transformers são criados SEM pesos pré-treinados (não precisa de internet).
Com --pretrained baixa os pesos do ImageNet (nvidia/mit-b0 e microsoft/swin-tiny-patch4-window7-224) e confere
que eles foram carregados: use esta opção no seu computador/Colab.

Uso:
    python test_models.py                 # os três, sem baixar pesos
    python test_models.py --pretrained    # com pesos pré-treinados
    python test_models.py --archs unet    # só uma arquitetura
"""
import argparse
import time

import torch
import torch.nn as nn

from models import ARCHS, build_model, load_arch_config
from models.common import count_parameters
from training import bce_dice_loss


def first_conv_in_channels(model):
    return next(m.in_channels for m in model.modules() if isinstance(m, nn.Conv2d))


def check_arch(arch, pretrained, device, cfg_root):
    cfg = load_arch_config(arch, cfg_root)
    kw = dict(cfg.get("model", {}))
    if not pretrained and "pretrained" in kw:
        kw["pretrained"] = None
    t0 = time.time()
    model = build_model(arch, in_channels=10, **kw).to(device)
    build_s = time.time() - t0
    has_loss = hasattr(model, "training_loss")
    proba = (lambda z: model.predict_proba(z)) if hasattr(model, "predict_proba") else (lambda z: torch.sigmoid(model(z)))
    loss_fn = (lambda a, b: model.training_loss(a, b)) if has_loss else (lambda a, b: bce_dice_loss(model(a), b))

    torch.manual_seed(0)
    x = torch.randn(2, 10, 128, 128, device=device)
    y = (torch.rand(2, 1, 128, 128, device=device) > 0.8).float()

    # 1) saída e probabilidades
    model.eval()
    with torch.no_grad():
        p = proba(x)
        t0 = time.time()
        for _ in range(3):
            proba(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_ms = (time.time() - t0) / 3 * 1000
    assert tuple(p.shape) == (2, 1, 128, 128), f"forma inesperada: {tuple(p.shape)}"
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0, "probabilidades fora de [0, 1]"

    # 2) entrada de 10 canais
    assert first_conv_in_channels(model) == 10, "a primeira camada não tem 10 canais"

    # 3) perda e gradiente (com primeiro plano e com um quadro sem primeiro plano)
    model.train()
    y_mixed = y.clone()
    y_mixed[1] = 0.0                                              # segundo quadro sem nenhum positivo
    for tag, target in (("com primeiro plano", y), ("com quadro vazio", y_mixed)):
        model.zero_grad(set_to_none=True)
        loss = loss_fn(x, target)
        assert torch.isfinite(loss), f"perda não finita ({tag})"
        loss.backward()
        grads = [p_.grad for p_ in model.parameters() if p_.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), f"gradiente inválido ({tag})"
        assert any(float(g.abs().sum()) > 0 for g in grads), f"gradiente nulo ({tag})"

    note = ""
    if getattr(model, "loaded_fraction", None) is not None:
        note = f" | pesos do backbone carregados: {100 * model.loaded_fraction:.0f}%"
    print(f"[ok] {arch:12s} | {count_parameters(model):>11,} parâmetros | passada (lote 2): {fwd_ms:7.1f} ms | "
          f"construção: {build_s:4.1f}s | perda de teste: {float(loss):.3f}{note}")
    return count_parameters(model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS))
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--configs", default="configs")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {dev} | pesos pré-treinados: {a.pretrained}")
    for arch in a.archs:
        check_arch(arch, a.pretrained, dev, a.configs)
    print("test_models.py: todos os modelos passaram")
