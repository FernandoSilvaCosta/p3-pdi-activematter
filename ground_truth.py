"""
Etapa 5 - Ground Truth em 128 px (versão 2), a partir dos dados processados por prepare_dataset.py.

Critério (igual à versão 1, agora aplicado à concentração JÁ REDUZIDA para 128 x 128):
    d(x,y,t) = c(x,y,t) - 1
    M(x,y,t) = 1  se  d(x,y,t) > k * sigma(t)     (sigma(t) = desvio padrão de d no quadro t)
Quadros com sigma(t) < sigma_floor são marcados como inválidos (valid = False) e devem ser
excluídos do treino e da avaliação.

Verificação opcional (--compare-raw): para os splits cujos arquivos originais de 256 x 256 existem
localmente (normalmente o valid), gera a máscara em 256 px, reduz com nearest-neighbor e compara com a
máscara de 128 px (IoU, Dice, concordância de pixels, razão entre sigmas, flags de validade). Isso é a
validação quantitativa da escolha "gerar o GT depois da redução" em vez de reduzir a máscara.

Uso:
    python ground_truth.py --splits valid --compare-raw ./data/raw/datasets/active_matter/data
    python ground_truth.py --splits valid test train        (depois que test e train estiverem processados)

Saídas (em --out, padrão data/ground_truth):
    <split>/<mesmo nome>.hdf5    mask (N,Ts,h,w) uint8, sigma (N,Ts), valid (N,Ts), time_index (Ts,)
    <split>_gt_summary.csv       resumo por trajetória
    <split>_gt_validation.png    figura de validação
    <split>_gt_resolution_check.csv   (se --compare-raw) comparação com 256 px
    ground_truth_manifest.json   parâmetros e contagens (para congelar e comparar entre computadores)
ATENÇÃO: sobrescreve as máscaras da versão 1 (256 px) com o mesmo nome de arquivo.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

VERSION = "v2_128px"


def make_mask(c, k, sigma_floor):
    """c: (N, Ts, h, w) -> mask (N,Ts,h,w) uint8, sigma (N,Ts) float32, valid (N,Ts) bool."""
    d = c.astype(np.float64) - 1.0
    N, Ts = d.shape[:2]
    sigma = d.reshape(N, Ts, -1).std(axis=2)
    mask = (d > k * sigma[:, :, None, None]).astype(np.uint8)
    valid = sigma >= sigma_floor
    return mask, sigma.astype(np.float32), valid


def compare_with_raw(raw_path, t_idx, factor, mask128, sigma128, valid128, k, sigma_floor):
    """Compara a máscara de 128 px com a de 256 px reduzida por nearest-neighbor."""
    iou, dice, agree, ratio, mism, nfr = [], [], [], [], 0, 0
    with h5py.File(raw_path, "r") as f:
        ds = f["t0_fields/concentration"]
        for n in range(mask128.shape[0]):
            c = ds[n, t_idx.tolist()]                              # (Ts, 256, 256)
            d = c.astype(np.float64) - 1.0
            s = d.reshape(len(t_idx), -1).std(axis=1)
            m256 = d > k * s[:, None, None]
            nn = m256[:, ::factor, ::factor]                       # nearest-neighbor
            valid256 = s >= sigma_floor
            mism += int((valid256 != valid128[n]).sum())
            nfr += len(t_idx)
            for t in np.where(valid256 & valid128[n])[0]:
                g, p = mask128[n, t].astype(bool), nn[t]
                inter, union = (g & p).sum(), (g | p).sum()
                if union == 0:
                    continue
                iou.append(inter / union)
                dice.append(2 * inter / max(g.sum() + p.sum(), 1))
                agree.append((g == p).mean())
                ratio.append(sigma128[n, t] / s[t])
    return {k_: np.array(v) for k_, v in dict(iou=iou, dice=dice, agree=agree, ratio=ratio).items()}, mism, nfr


def validation_figure(split, picks, out_dir, proc_dir, gt_dir):
    fig, axs = plt.subplots(len(picks), 4, figsize=(12.8, 3.3 * len(picks)))
    for i, (name, n) in enumerate(picks):
        with h5py.File(proc_dir / split / name, "r") as p, h5py.File(gt_dir / split / name, "r") as g:
            Ts = p["time_index"].shape[0]
            pos = [0, Ts // 4, Ts // 2, Ts - 1]
            ti = p["time_index"][...]
            for j, t in enumerate(pos):
                d = p["concentration"][n, t].astype(np.float64) - 1.0
                m = g["mask"][n, t]
                v = bool(g["valid"][n, t])
                ax = axs[i, j]
                ax.imshow(d, cmap="viridis"); ax.axis("off")
                if m.any() and not m.all():
                    ax.contour(m, levels=[0.5], colors="red", linewidths=0.8)
                tag = "" if v else "  [INVÁLIDO]"
                ax.set_title(f"{name[14:-5]}\ntraj {n}, t={int(ti[t])}, sigma={d.std():.1e}{tag}", fontsize=6)
    fig.tight_layout()
    path = out_dir / f"{split}_gt_validation.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def main(args):
    out = args.out
    manifest = {"version": VERSION, "created": datetime.now().isoformat(timespec="seconds"),
                "criterion": "mask = (c - 1) > k * std_frame(c - 1), c = concentração reduzida para 128 px",
                "k": args.k, "sigma_floor": args.sigma_floor, "per_split": {}}

    for split in args.splits:
        pdir = args.processed / split
        files = sorted(pdir.glob("*.hdf5")) if pdir.exists() else []
        if not files:
            print(f"[{split}] nenhum arquivo processado em {pdir}, pulando")
            continue
        (out / split).mkdir(parents=True, exist_ok=True)
        rows, tot = [], dict(traj=0, frames=0, valid=0, pos_px=0)
        cmp_all = {"iou": [], "dice": [], "agree": [], "ratio": []}
        cmp_rows, mism_tot, nfr_tot = [], 0, 0
        do_cmp = args.compare_raw is not None and (args.compare_raw / split).exists()
        if args.compare_raw is not None and not do_cmp:
            print(f"[{split}] originais de 256 px não encontrados em {args.compare_raw / split}; sem comparação")

        for fp in files:
            with h5py.File(fp, "r") as p:
                c = p["concentration"][...]
                t_idx = p["time_index"][...]
                factor = int(p.attrs["factor"])
            mask, sigma, valid = make_mask(c, args.k, args.sigma_floor)
            N, Ts = mask.shape[:2]
            with h5py.File(out / split / fp.name, "w") as g:
                g.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
                g.create_dataset("sigma", data=sigma)
                g.create_dataset("valid", data=valid)
                g.create_dataset("time_index", data=t_idx)
                g.attrs.update(version=VERSION, k=args.k, sigma_floor=args.sigma_floor,
                               source_file=fp.name, criterion=manifest["criterion"])
            for n in range(N):
                frac = mask[n].reshape(Ts, -1).mean(axis=1)
                v = valid[n]
                rows.append({
                    "file": fp.name, "traj": n, "frames_total": Ts, "frames_valid": int(v.sum()),
                    "frames_excluded": int((~v).sum()),
                    "first_invalid_t": int(t_idx[np.argmax(~v)]) if (~v).any() else -1,
                    "pos_frac_mean_valid": float(frac[v].mean()) if v.any() else float("nan"),
                    "pos_frac_min_valid": float(frac[v].min()) if v.any() else float("nan"),
                    "pos_frac_max_valid": float(frac[v].max()) if v.any() else float("nan"),
                    "sigma_mean": float(sigma[n].mean()),
                })
            tot["traj"] += N; tot["frames"] += N * Ts; tot["valid"] += int(valid.sum())
            tot["pos_px"] += int(mask[valid].sum())

            if do_cmp and (args.compare_raw / split / fp.name).exists():
                r, mism, nfr = compare_with_raw(args.compare_raw / split / fp.name, t_idx, factor,
                                                mask, sigma, valid, args.k, args.sigma_floor)
                mism_tot += mism; nfr_tot += nfr
                for key in cmp_all:
                    cmp_all[key].append(r[key])
                if len(r["iou"]):
                    cmp_rows.append({"file": fp.name, "frames_compared": len(r["iou"]),
                                     "iou_mean": r["iou"].mean(), "iou_min": r["iou"].min(),
                                     "dice_mean": r["dice"].mean(), "pixel_agreement_mean": r["agree"].mean(),
                                     "sigma_ratio_128_over_256_median": float(np.median(r["ratio"])),
                                     "valid_flag_mismatches": mism})

        # ---- resumo do split
        with open(out / f"{split}_gt_summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        exc = tot["frames"] - tot["valid"]
        pm = float(np.nanmean([r["pos_frac_mean_valid"] for r in rows]))
        print(f"\n=== Split '{split}' ({VERSION}) ===")
        print(f"{len(files)} arquivos, {tot['traj']} trajetórias, {tot['frames']} quadros")
        print(f"Critério: d > {args.k} * sigma_frame ; sigma_floor = {args.sigma_floor:g}")
        print(f"Quadros excluídos: {exc} ({100 * exc / tot['frames']:.1f}%) | "
              f"trajetórias com algum excluído: {sum(r['frames_excluded'] > 0 for r in rows)} de {len(rows)}")
        print(f"Fração positiva média (quadros válidos): {pm:.3f}")
        manifest["per_split"][split] = {
            "n_files": len(files), "n_trajectories": tot["traj"], "n_frames": tot["frames"],
            "n_valid_frames": tot["valid"], "n_excluded_frames": exc,
            "pos_frac_mean_valid": round(pm, 4), "positive_pixels_valid_frames": tot["pos_px"]}

        # ---- comparação com 256 px
        if cmp_rows:
            allv = {k_: np.concatenate(v) for k_, v in cmp_all.items()}
            with open(out / f"{split}_gt_resolution_check.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(cmp_rows[0].keys()))
                w.writeheader(); w.writerows(cmp_rows)
            res = {"frames_compared": int(len(allv["iou"])),
                   "iou_mean": float(allv["iou"].mean()), "iou_p5": float(np.percentile(allv["iou"], 5)),
                   "iou_min": float(allv["iou"].min()), "dice_mean": float(allv["dice"].mean()),
                   "pixel_agreement_mean": float(allv["agree"].mean()),
                   "sigma_ratio_median": float(np.median(allv["ratio"])),
                   "valid_flag_mismatches": mism_tot, "frames_total_checked": nfr_tot}
            manifest["per_split"][split]["resolution_check_vs_256_nearest"] = {k_: round(v, 4) if isinstance(v, float) else v
                                                                                for k_, v in res.items()}
            print("\nComparação GT 128 px (gerado após a redução) x GT 256 px reduzido por nearest-neighbor:")
            print(f"  quadros comparados: {res['frames_compared']}")
            print(f"  IoU médio {res['iou_mean']:.3f} | p5 {res['iou_p5']:.3f} | mínimo {res['iou_min']:.3f}")
            print(f"  Dice médio {res['dice_mean']:.3f} | concordância de pixels {res['pixel_agreement_mean']:.4f}")
            print(f"  razão sigma128/sigma256 (mediana): {res['sigma_ratio_median']:.3f}")
            print(f"  divergências na flag de validade: {mism_tot} de {nfr_tot} quadros")

        # ---- figura de validação: menor, mediana e maior amplitude
        order = sorted(rows, key=lambda r: r["sigma_mean"])
        picks = [(r["file"], r["traj"]) for r in (order[0], order[len(order) // 2], order[-1])]
        print(f"Figura de validação: {validation_figure(split, picks, out, args.processed, out)}")

    (out / "ground_truth_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    print(f"\nManifesto do Ground Truth: {out / 'ground_truth_manifest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("data/ground_truth"))
    ap.add_argument("--splits", nargs="+", default=["valid"], choices=["train", "valid", "test"])
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--sigma-floor", type=float, default=1e-5)
    ap.add_argument("--compare-raw", type=Path, default=None,
                    help="pasta com os originais de 256 px (ex.: ./data/raw/datasets/active_matter/data)")
    main(ap.parse_args())
