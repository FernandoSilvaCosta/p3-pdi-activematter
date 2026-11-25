"""Utilidades compartilhadas pelos modelos: adaptação da primeira camada para 10 canais."""
import torch
import torch.nn as nn


def adapt_conv_in_weight(w, in_channels):
    """
    (out, 3, k, k) -> (out, in_channels, k, k).
    Usa a média dos pesos RGB replicada em todos os canais e reescala por 3/in_channels, para manter
    a mesma escala de ativação da camada original.
    """
    m = w.mean(dim=1, keepdim=True)
    return m.repeat(1, in_channels, 1, 1) * (w.shape[1] / float(in_channels))


def _set_module(root, dotted, new):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def replace_first_conv(root, in_channels):
    """
    Troca a primeira Conv2d de entrada de 3 canais encontrada em `root` por uma de `in_channels`,
    inicializada pela média dos pesos originais. Retorna o nome da camada trocada.
    """
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            new = nn.Conv2d(in_channels, mod.out_channels, mod.kernel_size, mod.stride, mod.padding,
                            mod.dilation, mod.groups, bias=mod.bias is not None)
            with torch.no_grad():
                new.weight.copy_(adapt_conv_in_weight(mod.weight, in_channels))
                if mod.bias is not None:
                    new.bias.copy_(mod.bias)
            _set_module(root, name, new)
            return name
    raise RuntimeError("Nenhuma Conv2d com 3 canais de entrada encontrada para adaptar.")


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))
