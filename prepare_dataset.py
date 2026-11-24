"""
Etapa 4 - Seleção do subconjunto, redução espacial e manifesto (protocolo do professor).

O que faz, para cada arquivo HDF5 do active_matter:
  1. Seleciona TODAS as trajetórias e 1 quadro a cada `stride` (padrão 3): t = 0, 3, 6, ..., 78.
     Seleção determinística (sem sorteio, seed não se aplica).
  2. Reduz os campos de 256 x 256 para 128 x 128 por média de blocos (2 x 2). É uma média de área
     (antialiasing), sem aumentar a resolução e preservando a proporção.
  3. Salva um arquivo compacto por arquivo de origem em <out>/<split>/<mesmo nome>.hdf5, com:
        fields        (N, Ts, 10, h, w)  canais de entrada: vel_x, vel_y, D_xx..D_yy, E_xx..E_yy
        concentration (N, Ts, h, w)      usada SOMENTE para gerar o Ground Truth
        time_index    (Ts,)              índices de tempo originais selecionados
        traj_index    (N,)               índices de trajetória originais
     + atributos (origem, parâmetros L/zeta/alpha, método, erro de reconstrução etc.).
  4. Gera o manifesto <out>/manifest.json e <out>/manifest_files.csv (reconstruído a partir dos
     arquivos já processados, então funciona com retomada após interrupções).

Dois modos de origem:
  --source local : lê arquivos já baixados em <raw-root>/<split>/*.hdf5 (não apaga nada).
  --source hf    : baixa cada arquivo do Hugging Face (polymathic-ai/active_matter), processa e apaga
                   o original (use --keep-raw para manter). Evita guardar os 55,8 GB de uma vez.

Arquivos já processados são pulados (use --overwrite para refazer), o que permite retomar no Colab.

Exemplos:
    python prepare_dataset.py --splits valid --source local --raw-root ./data/raw/datasets/active_matter/data
    python prepare_dataset.py --splits train test --source hf --tmp ./data/tmp_download
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

REPO_ID = "polymathic-ai/active_matter"
CHANNELS = ["vel_x", "vel_y", "D_xx", "D_xy", "D_yx", "D_yy", "E_xx", "E_xy", "E_yx", "E_yy"]
TARGET_GB = 25.0


# ----------------------------------------------------------------------------- redução espacial
def block_mean(x, f):
    """Média de blocos f x f nos eixos espaciais 1 e 2 de x com forma (T, H, W, C)."""
    if f == 1:
        return x.astype(np.float32, copy=False)
    T, H, W, C = x.shape
    assert H % f == 0 and W % f == 0, f"Dimensões {H}x{W} não divisíveis por {f}"
    y = x.astype(np.float32, copy=False).reshape(T, H // f, f, W // f, f, C)
    return y.mean(axis=(2, 4), dtype=np.float32)


def upsample_nn(x, f):
    """Repete cada pixel f x f vezes (apenas para medir o erro de reconstrução, não é upscaling de dados)."""
    return np.repeat(np.repeat(x, f, axis=1), f, axis=2)


def reduction_factor(H, W, target):
    m = max(H, W)
    return 1 if m <= target else math.ceil(m / target)


# ----------------------------------------------------------------------------- processamento de um arquivo
def process_file(src: Path, dst: Path, split: str, stride: int, target: int):
    tmp = dst.with_suffix(".tmp")
    with h5py.File(src, "r") as f:
        conc_ds = f["t0_fields/concentration"]
        N, T, H, W = conc_ds.shape
        factor = reduction_factor(H, W, target)
        h, w = H // factor, W // factor
        sl = slice(0, T, stride)
        t_idx = np.arange(0, T, stride, dtype=np.int32)
        Ts = len(t_idx)

        num = np.zeros(11, dtype=np.float64)   # erro de reconstrução: 10 canais + concentração (c-1)
        den = np.zeros(11, dtype=np.float64)

        with h5py.File(tmp, "w") as g:
            dfields = g.create_dataset("fields", shape=(N, Ts, 10, h, w), dtype="float32",
                                       chunks=(1, 1, 10, h, w))
            dconc = g.create_dataset("concentration", shape=(N, Ts, h, w), dtype="float32",
                                     chunks=(1, 1, h, w))
            g.create_dataset("time_index", data=t_idx)
            g.create_dataset("traj_index", data=np.arange(N, dtype=np.int32))

            for n in range(N):                       # uma trajetória por vez: pouca memória
                c = conc_ds[n, sl]                                   # (Ts, H, W)
                vel = f["t1_fields/velocity"][n, sl]                 # (Ts, H, W, 2)
                D = f["t2_fields/D"][n, sl].reshape(Ts, H, W, 4)
                E = f["t2_fields/E"][n, sl].reshape(Ts, H, W, 4)
                x = np.concatenate([vel, D, E], axis=-1)             # (Ts, H, W, 10)

                xd = block_mean(x, factor)                           # (Ts, h, w, 10)
                cd = block_mean(c[..., None], factor)[..., 0]        # (Ts, h, w)

                dfields[n] = np.moveaxis(xd, -1, 1)                  # (Ts, 10, h, w)
                dconc[n] = cd

                # erro relativo de reconstrução (quanta estrutura fina se perde na redução)
                up = upsample_nn(xd, factor)
                num[:10] += ((x.astype(np.float64) - up) ** 2).sum(axis=(0, 1, 2))
                den[:10] += (x.astype(np.float64) ** 2).sum(axis=(0, 1, 2))
                dev, devd = c.astype(np.float64) - 1.0, cd.astype(np.float64) - 1.0
                num[10] += ((dev - upsample_nn(devd[..., None], factor)[..., 0]) ** 2).sum()
                den[10] += (dev ** 2).sum()

            rel_err = np.sqrt(num / np.maximum(den, 1e-30))
            g.attrs.update({
                "source_file": src.name, "split": split,
                "L": float(f.attrs["L"]), "zeta": float(f.attrs["zeta"]), "alpha": float(f.attrs["alpha"]),
                "n_traj": N, "orig_T": T, "orig_H": H, "orig_W": W, "h": h, "w": w,
                "stride": stride, "factor": factor, "method": f"block_mean_{factor}x{factor}",
                "orig_size_bytes": os.path.getsize(src),
                "channels": json.dumps(CHANNELS),
                "rel_err": rel_err, "rel_err_names": json.dumps(CHANNELS + ["conc_minus_1"]),
            })
    os.replace(tmp, dst)
    return N, rel_err


# ----------------------------------------------------------------------------- origem dos arquivos
def hf_list(split):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"data/{split}/") and f.endswith((".hdf5", ".h5")))


def hf_download(repo_path, tmp_dir: Path):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=repo_path, local_dir=str(tmp_dir)))


# ----------------------------------------------------------------------------- manifesto
def build_manifest(out: Path, args):
    rows, per_split = [], {}
    for split in ["train", "valid", "test"]:
        files = sorted((out / split).glob("*.hdf5"))
        if not files:
            continue
        sp = {"n_files": 0, "n_trajectories": 0, "n_frames": 0, "orig_equiv_gb": 0.0, "processed_gb": 0.0}
        for fp in files:
            with h5py.File(fp, "r") as g:
                a = dict(g.attrs)
                t_idx = g["time_index"][...].tolist()
            n, Ts = int(a["n_traj"]), len(t_idx)
            equiv = float(a["orig_size_bytes"]) * Ts / int(a["orig_T"]) / 1e9
            size = os.path.getsize(fp) / 1e9
            rel = np.asarray(a["rel_err"], dtype=float)
            rows.append({
                "split": split, "file": a["source_file"], "L": a["L"], "zeta": a["zeta"], "alpha": a["alpha"],
                "n_traj": n, "frames_per_traj": Ts, "orig_size_gb": float(a["orig_size_bytes"]) / 1e9,
                "orig_equiv_gb_selected": equiv, "processed_gb": size,
                "rel_err_max_channel": float(rel[:10].max()), "rel_err_conc": float(rel[10]),
            })
            sp["n_files"] += 1; sp["n_trajectories"] += n; sp["n_frames"] += n * Ts
            sp["orig_equiv_gb"] += equiv; sp["processed_gb"] += size
        per_split[split] = sp

    total_equiv = sum(s["orig_equiv_gb"] for s in per_split.values())
    manifest = {
        "dataset": "active_matter (The Well)", "repo_id": REPO_ID,
        "created": datetime.now().isoformat(timespec="seconds"),
        "selection": {
            "trajectories": "todas as trajetórias de cada arquivo",
            "frames": f"1 a cada {args.stride}: t = 0, {args.stride}, {2 * args.stride}, ... (ao longo de todo o tempo)",
            "method": "determinístico (seed não se aplica)", "splits": "oficiais do The Well (train/valid/test)",
        },
        "reduction": {"target_max_dim": args.target, "method": "média de blocos (antialiasing), sem upscaling",
                      "masks": "Ground Truth será gerado a partir da concentração já reduzida"},
        "channels": CHANNELS,
        "budget": {"target_gb": TARGET_GB, "subset_orig_equiv_gb": round(total_equiv, 2),
                   "within_budget": bool(total_equiv <= TARGET_GB)},
        "per_split": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in per_split.items()},
        "software": {"numpy": np.__version__, "h5py": h5py.__version__},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with open(out / "manifest_files.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return manifest, rows


# ----------------------------------------------------------------------------- main
def main(args):
    out = args.out
    for split in args.splits:
        (out / split).mkdir(parents=True, exist_ok=True)
        if args.source == "local":
            files = sorted((args.raw_root / split).glob("*.hdf5"))
            items = [(p.name, p) for p in files]
        else:
            items = [(Path(r).name, r) for r in hf_list(split)]
        if args.limit_files:
            items = items[:args.limit_files]
        if not items:
            raise FileNotFoundError(f"Nenhum arquivo encontrado para o split '{split}'")
        print(f"\n=== Split '{split}': {len(items)} arquivos ===")

        for i, (name, ref) in enumerate(items, 1):
            dst = out / split / name
            if dst.exists() and not args.overwrite:
                print(f"[{i}/{len(items)}] {name}: já processado, pulando")
                continue
            t0 = time.time()
            src, downloaded = (ref, False) if args.source == "local" else (hf_download(ref, args.tmp), True)
            try:
                N, rel = process_file(src, dst, split, args.stride, args.target)
            finally:
                if downloaded and not args.keep_raw and Path(src).exists():
                    Path(src).unlink()
            print(f"[{i}/{len(items)}] {name}: {N} traj, erro de reconstrução máx. {rel[:10].max():.3f} "
                  f"(conc {rel[10]:.3f}), {time.time() - t0:.1f}s")

    manifest, rows = build_manifest(out, args)
    print("\n=== Manifesto ===")
    for k, v in manifest["per_split"].items():
        print(f"{k:6s}: {v['n_files']} arquivos, {v['n_trajectories']} trajetórias, {v['n_frames']} quadros, "
              f"{v['orig_equiv_gb']:.2f} GB (equivalente original), {v['processed_gb']:.2f} GB em disco")
    b = manifest["budget"]
    print(f"Subconjunto total (equivalente original): {b['subset_orig_equiv_gb']} GB | teto {b['target_gb']} GB | "
          f"dentro do teto: {b['within_budget']}")
    if rows:
        print(f"Erro de reconstrução (maior canal) entre arquivos: máx = "
              f"{max(r['rel_err_max_channel'] for r in rows):.3f}, concentração máx = "
              f"{max(r['rel_err_conc'] for r in rows):.3f}")
    print(f"Manifesto salvo em {out / 'manifest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["valid"], choices=["train", "valid", "test"])
    ap.add_argument("--source", choices=["local", "hf"], default="local")
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/datasets/active_matter/data"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--tmp", type=Path, default=Path("data/tmp_download"))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--target", type=int, default=128)
    ap.add_argument("--keep-raw", action="store_true", help="modo hf: não apagar o arquivo baixado")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-files", type=int, default=None, help="processa só os N primeiros (teste)")
    main(ap.parse_args())
