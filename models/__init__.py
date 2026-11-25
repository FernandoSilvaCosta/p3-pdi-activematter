"""Fábrica de modelos e leitura das configurações por arquitetura (configs/<arq>.yaml)."""
from pathlib import Path

ARCHS = ("unet", "segformer", "mask2former")


def build_model(arch, in_channels=10, **kwargs):
   
    if arch == "unet":
        from .unet import UNet
        return UNet(in_channels=in_channels, **kwargs)
    if arch == "segformer":
        from .segformer import SegFormer
        return SegFormer(in_channels=in_channels, **kwargs)
    if arch == "mask2former":
        from .mask2former import Mask2Former
        return Mask2Former(in_channels=in_channels, **kwargs)
    raise ValueError(f"Arquitetura desconhecida: {arch}. Use uma de {ARCHS}")


def load_arch_config(arch, root="configs"):
    """Lê configs/<arq>.yaml -> {'model': {...}, 'train': {...}}."""
    import yaml
    with open(Path(root) / f"{arch}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
