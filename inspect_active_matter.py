"""
Etapa 1 - Inspeção do dataset active_matter (The Well).

Uso:
    python inspect_active_matter.py --root ./data/raw/active_matter

Antes, baixe os dados (exemplo, começando por um split pequeno):
    pip install the_well h5py matplotlib numpy
    the-well-download --base-path ./data/raw --dataset active_matter --split valid
(Confira a sintaxe atual na documentação: https://github.com/PolymathicAI/the_well)

O script NÃO assume nomes de campos: ele percorre o HDF5 e mostra o que existe.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def find_h5(root: Path):
    files = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo HDF5 encontrado em {root}")
    return files


def describe(path: Path):
    print(f"\n=== {path} ===")
    with h5py.File(path, "r") as f:
        print("Atributos globais:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [dataset] {name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print(f"  [grupo]   {name}")

        f.visititems(visit)


def collect_fields(f):
    """Retorna dict nome -> dataset para os campos t0/t1/t2."""
    out = {}
    for grp in ("t0_fields", "t1_fields", "t2_fields"):
        if grp in f:
            for name, ds in f[grp].items():
                out[f"{grp}/{name}"] = ds
    return out


def stats_and_plots(path: Path, out_dir: Path, traj=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        fields = collect_fields(f)
        print("\nCampos encontrados:", list(fields.keys()))

        # Estatísticas por campo (primeira trajetória, todos os instantes)
        for name, ds in fields.items():
            arr = ds[traj]  # (T, H, W, [comp...])
            print(f"\n{name}: shape(traj)={arr.shape}")
            print(f"  min={arr.min():.4g}  max={arr.max():.4g}  "
                  f"média={arr.mean():.4g}  desvio={arr.std():.4g}")
            print(f"  NaN: {np.isnan(arr).sum()}")

        # Concentração: histograma e percentis (base para escolher o limiar)
        conc_key = next((k for k in fields if "concentration" in k), None)
        if conc_key is not None:
            c = fields[conc_key][traj][..., 0] if fields[conc_key].ndim == 5 \
                else fields[conc_key][traj]
            vals = c.ravel()
            pcts = [50, 60, 70, 75, 80, 85, 90, 95]
            print("\nPercentis da concentração:")
            for p in pcts:
                print(f"  P{p}: {np.percentile(vals, p):.4g}")

            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=100)
            plt.xlabel("concentração")
            plt.ylabel("contagem")
            plt.title("Distribuição da concentração (trajetória %d)" % traj)
            plt.tight_layout()
            plt.savefig(out_dir / "hist_concentration.png", dpi=150)
            plt.close()

            # Alguns instantes da concentração
            T = c.shape[0]
            ts = [0, T // 4, T // 2, T - 1]
            fig, axs = plt.subplots(1, len(ts), figsize=(4 * len(ts), 4))
            for ax, t in zip(axs, ts):
                im = ax.imshow(c[t], cmap="viridis")
                ax.set_title(f"t = {t}")
                ax.axis("off")
            fig.colorbar(im, ax=axs, shrink=0.8)
            fig.savefig(out_dir / "concentration_frames.png", dpi=150)
            plt.close(fig)
        else:
            print("\nAviso: campo 'concentration' não encontrado; verifique os nomes acima.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("results/inspection"), type=Path)
    ap.add_argument("--traj", default=0, type=int)
    args = ap.parse_args()

    files = find_h5(args.root)
    print(f"{len(files)} arquivo(s) HDF5 encontrados.")
    describe(files[0])
    stats_and_plots(files[0], args.out, traj=args.traj)
    print(f"\nFiguras salvas em: {args.out}")
