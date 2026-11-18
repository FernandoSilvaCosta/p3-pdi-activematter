"""
Etapa 2b - Geração do Ground Truth do active_matter.

Critério (relativo por quadro):
    d(x,y,t) = c(x,y,t) - 1
    M(x,y,t) = 1  se  d(x,y,t) > k * sigma(t)      (sigma(t) = desvio padrão de d no quadro t)
    M(x,y,t) = 0  caso contrário

Quadros com sigma(t) < sigma_floor são marcados como inválidos (valid = False): neles a
flutuação de concentração está próxima da resolução numérica de float32 e a máscara
não tem significado físico. Esses quadros devem ser excluídos do treino e da avaliação.

A concentração é usada SOMENTE aqui. Ela não entra como canal de entrada das redes.

Uso:
    python ground_truth.py --root ./data/raw/datasets/active_matter/data/valid \
                           --out  ./data/ground_truth --k 1.0 --sigma-floor 1e-5

Saída: <out>/<split>/<mesmo nome do arquivo>.hdf5 com
    mask   (N, T, H, W) uint8
    sigma  (N, T)       float32
    valid  (N, T)       bool
e atributos k, sigma_floor, criterion, source_file.
Também grava <out>/<split>_gt_summary.csv e <out>/<split>_gt_validation.png.
"""
import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def make_mask(c, k, sigma_floor):
    """c: (N, T, H, W) -> mask (N,T,H,W) uint8, sigma (N,T), valid (N,T)."""
    d = c.astype(np.float64) - 1.0
    N, T = d.shape[:2]
    sigma = d.reshape(N, T, -1).std(axis=2)                    # (N, T)
    mask = (d > k * sigma[:, :, None, None]).astype(np.uint8)
    valid = sigma >= sigma_floor
    return mask, sigma.astype(np.float32), valid


def main(root, out, k, sigma_floor):
    files = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"Nenhum HDF5 em {root}")
    split = root.name
    out_dir = out / split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, examples = [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            c = f["t0_fields/concentration"][...]
            alpha, zeta, L = float(f.attrs["alpha"]), float(f.attrs["zeta"]), float(f.attrs["L"])
        mask, sigma, valid = make_mask(c, k, sigma_floor)
        N, T = mask.shape[:2]

        with h5py.File(out_dir / fp.name, "w") as g:
            g.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
            g.create_dataset("sigma", data=sigma)
            g.create_dataset("valid", data=valid)
            g.attrs.update(k=k, sigma_floor=sigma_floor, source_file=fp.name,
                           criterion="mask = (c - 1) > k * std_frame(c - 1)")

        for n in range(N):
            frac = mask[n].reshape(T, -1).mean(axis=1)
            v = valid[n]
            rows.append({
                "file": fp.name, "traj": n, "L": L, "zeta": zeta, "alpha": alpha,
                "frames_total": T, "frames_valid": int(v.sum()), "frames_excluded": int((~v).sum()),
                "first_invalid_t": int(np.argmax(~v)) if (~v).any() else -1,
                "pos_frac_mean_valid": float(frac[v].mean()) if v.any() else float("nan"),
                "pos_frac_min_valid": float(frac[v].min()) if v.any() else float("nan"),
                "pos_frac_max_valid": float(frac[v].max()) if v.any() else float("nan"),
                "sigma_mean": float(sigma[n].mean()),
            })
            examples.append((float(sigma[n].mean()), fp.name, n, c[n] - 1.0, mask[n], valid[n]))

    # ---- resumo
    csv_path = out / f"{split}_gt_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    tot = sum(r["frames_total"] for r in rows)
    exc = sum(r["frames_excluded"] for r in rows)
    pm = np.nanmean([r["pos_frac_mean_valid"] for r in rows])
    print(f"Split '{split}': {len(files)} arquivos, {len(rows)} trajetórias, {tot} quadros")
    print(f"Critério: d > {k} * sigma_frame ; sigma_floor = {sigma_floor:g}")
    print(f"Quadros excluídos (sigma < floor): {exc} ({100 * exc / tot:.1f}%)")
    n_traj_exc = sum(1 for r in rows if r["frames_excluded"] > 0)
    print(f"Trajetórias com algum quadro excluído: {n_traj_exc} de {len(rows)}")
    print(f"Fração positiva média (quadros válidos): {pm:.3f}")
    print(f"Resumo salvo em {csv_path}")

    # ---- figura de validação: trajetórias de menor, mediana e maior amplitude
    examples.sort(key=lambda e: e[0])
    picks = [examples[0], examples[len(examples) // 2], examples[-1]]
    times = [0, 20, 40, 80]
    fig, axs = plt.subplots(len(picks), len(times), figsize=(3.2 * len(times), 3.3 * len(picks)))
    for i, (sm, name, n, d, m, v) in enumerate(picks):
        for j, t in enumerate(times):
            ax = axs[i, j]
            ax.imshow(d[t], cmap="viridis"); ax.axis("off")
            if m[t].any() and not m[t].all():
                ax.contour(m[t], levels=[0.5], colors="red", linewidths=0.8)
            tag = "" if v[t] else "  [INVÁLIDO]"
            ax.set_title(f"{name[14:-5]}\ntraj {n}, t={t}, sigma={d[t].std():.1e}{tag}", fontsize=6)
    fig.tight_layout()
    fig_path = out / f"{split}_gt_validation.png"
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"Figura de validação: {fig_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="pasta do split (ex.: .../data/valid)")
    ap.add_argument("--out", default=Path("data/ground_truth"), type=Path)
    ap.add_argument("--k", default=1.0, type=float, help="múltiplo de sigma do quadro")
    ap.add_argument("--sigma-floor", default=1e-5, type=float)
    a = ap.parse_args()
    main(a.root, a.out, a.k, a.sigma_floor)
