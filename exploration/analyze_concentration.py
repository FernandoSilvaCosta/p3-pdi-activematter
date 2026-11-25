"""
Etapa 2a - Análise da concentração para escolher o critério de limiar do Ground Truth.

A concentração do active_matter é uma pequena flutuação em torno de 1.
Este script trabalha com o desvio d = c - 1 e compara critérios candidatos
em TODOS os arquivos HDF5 de um split (todas as combinações L, zeta, alpha).

Uso (exemplo com a sua estrutura de pastas):
    python analyze_concentration.py --root ./data/raw/datasets/active_matter/data/valid

Saídas em results/inspection/:
    concentration_summary.csv        resumo por arquivo e trajetória
    criteria_fractions.csv           fração de pixels positivos por critério
    std_vs_time.png                  desvio padrão da concentração ao longo do tempo
    criteria_boxplot.png             distribuição da fração positiva por critério
    mask_preview.png                 máscaras de cada critério em um quadro exemplo
"""
import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Critérios candidatos. Cada um recebe d = c - 1 (T, H, W) e devolve máscara booleana (T, H, W).
CRITERIA = {
    "d > 0 (acima da média)":      lambda d, s: d > 0,
    "d > 0.5*sigma_frame":         lambda d, s: d > 0.5 * s[:, None, None],
    "d > 1.0*sigma_frame":         lambda d, s: d > 1.0 * s[:, None, None],
    "d > 1e-4 (global)":           lambda d, s: d > 1e-4,
    "d > 5e-4 (global)":           lambda d, s: d > 5e-4,
    "d > 1e-3 (global)":           lambda d, s: d > 1e-3,
}


def list_files(root: Path):
    files = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"Nenhum HDF5 em {root}")
    return files


def main(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = list_files(root)
    print(f"{len(files)} arquivos encontrados em {root}")

    summary_rows = []
    frac = {k: [] for k in CRITERIA}      # fração positiva por quadro, todos os quadros
    std_curves = []                        # (alpha, zeta, std_t)
    preview = None

    for fp in files:
        with h5py.File(fp, "r") as f:
            c = f["t0_fields/concentration"][...]   # (N, T, H, W)
            alpha = float(f.attrs["alpha"])
            zeta = float(f.attrs["zeta"])
            L = float(f.attrs["L"])
        N, T = c.shape[:2]
        for n in range(N):
            d = c[n] - 1.0
            sigma_t = d.reshape(T, -1).std(axis=1)
            mean_t = d.reshape(T, -1).mean(axis=1)
            std_curves.append((alpha, zeta, sigma_t))
            summary_rows.append({
                "file": fp.name, "traj": n, "L": L, "zeta": zeta, "alpha": alpha,
                "mean_dev_max_abs": float(np.abs(mean_t).max()),
                "std_t0": float(sigma_t[0]), "std_t40": float(sigma_t[T // 2]),
                "std_tend": float(sigma_t[-1]), "std_mean": float(sigma_t.mean()),
                "min": float(d.min()), "max": float(d.max()),
            })
            for name, fn in CRITERIA.items():
                m = fn(d, sigma_t)
                frac[name].extend(m.reshape(T, -1).mean(axis=1).tolist())
            if preview is None:
                preview = (fp.name, d[T // 2], sigma_t[T // 2])

    # ---- tabelas
    with open(out / "concentration_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    print("\nDesvio padrão da concentração (std de c-1) entre trajetórias:")
    s_all = np.array([r["std_mean"] for r in summary_rows])
    print(f"  mínimo={s_all.min():.3e}  mediana={np.median(s_all):.3e}  máximo={s_all.max():.3e}")
    print(f"  razão máx/mín = {s_all.max() / s_all.min():.1f}x")
    print(f"  |média de c-1| máxima por quadro = {max(r['mean_dev_max_abs'] for r in summary_rows):.2e}")

    print("\nFração de pixels positivos por critério (todos os quadros):")
    print(f"{'critério':32s} {'média':>8s} {'p5':>8s} {'p95':>8s} {'%quadros<1%':>12s} {'%quadros>50%':>13s}")
    with open(out / "criteria_fractions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["criterio", "media", "p5", "p95", "pct_quadros_lt_1pct", "pct_quadros_gt_50pct"])
        for name, v in frac.items():
            v = np.array(v)
            row = [name, v.mean(), np.percentile(v, 5), np.percentile(v, 95),
                   100 * (v < 0.01).mean(), 100 * (v > 0.5).mean()]
            w.writerow(row)
            print(f"{name:32s} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.3f} {row[4]:12.1f} {row[5]:13.1f}")

    # ---- figura 1: desvio padrão ao longo do tempo
    alphas = np.array([a for a, _, _ in std_curves])
    norm = plt.Normalize(alphas.min(), alphas.max())
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a, z, s in std_curves:
        ax.plot(s, color=cmap(norm(a)), alpha=0.7, lw=1)
    ax.set_yscale("log"); ax.set_xlabel("instante t"); ax.set_ylabel("std de (c - 1)")
    ax.set_title("Amplitude das flutuações de concentração")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="alpha")
    fig.tight_layout(); fig.savefig(out / "std_vs_time.png", dpi=150); plt.close(fig)

    # ---- figura 2: boxplot das frações
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([np.array(v) for v in frac.values()], showfliers=False)
    ax.set_xticklabels(list(frac.keys()), rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("fração de pixels positivos por quadro")
    ax.set_title("Balanceamento de classes por critério")
    fig.tight_layout(); fig.savefig(out / "criteria_boxplot.png", dpi=150); plt.close(fig)

    # ---- figura 3: prévia das máscaras
    name, d, s = preview
    fig, axs = plt.subplots(1, len(CRITERIA) + 1, figsize=(3 * (len(CRITERIA) + 1), 3.4))
    axs[0].imshow(d, cmap="viridis"); axs[0].set_title("c - 1"); axs[0].axis("off")
    for ax, (cname, fn) in zip(axs[1:], CRITERIA.items()):
        m = fn(d[None], np.array([s]))[0]
        ax.imshow(m, cmap="gray"); ax.set_title(cname, fontsize=7); ax.axis("off")
    fig.suptitle(f"{name} (trajetória 0, t = 40)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "mask_preview.png", dpi=150); plt.close(fig)

    print(f"\nArquivos salvos em {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("results/inspection"), type=Path)
    a = ap.parse_args()
    main(a.root, a.out)
