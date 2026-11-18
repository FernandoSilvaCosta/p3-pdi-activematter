"""
Etapa 3 - Pipeline de dados do projeto active_matter.

Lê, de forma preguiçosa (sem carregar os 51 GB na memória):
  - os campos físicos do The Well:  <data_root>/<split>/*.hdf5
  - o Ground Truth gerado por ground_truth.py: <gt_root>/<split>/<mesmo nome>.hdf5

Cada amostra é UM quadro (trajetória n, instante t):
    x     (10, 256, 256) float32   canais físicos, SEM pré-processamento (baseline)
    y     (1, 256, 256)  float32   máscara Ground Truth (0/1)
    meta  dict                     arquivo, traj, t, sigma, índice

Ordem dos 10 canais (a concentração NÃO entra):
    0 vel_x   1 vel_y
    2 D_xx    3 D_xy    4 D_yx    5 D_yy      (tensor de orientação)
    6 E_xx    7 E_xy    8 E_yx    9 E_yy      (tensor strain-rate)

Pré-processamentos P1-P10 entram pelo argumento `transform` (função que recebe e devolve
um array/tensor (C,H,W)). Com transform=None obtém-se a condição de baseline.

Quadros com valid=False no Ground Truth são excluídos por padrão (valid_only=True).
Os índices ficam em ordem temporal por arquivo e trajetória (necessário para o Q10).

Uso como teste/estatísticas:
    python data_loader.py --data-root ./data/raw/datasets/active_matter/data \
                          --gt-root ./data/ground_truth --split valid --stats-out ./data/channel_stats_valid.json
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CHANNEL_NAMES = ["vel_x", "vel_y", "D_xx", "D_xy", "D_yx", "D_yy", "E_xx", "E_xy", "E_yx", "E_yy"]


class ActiveMatterSeg(Dataset):
    def __init__(self, data_root, gt_root, split, transform=None, valid_only=True,
                 max_files=None, include_concentration=False):
        """
        include_concentration=True acrescenta um 11o canal (c - 1). Serve APENAS ao experimento
        complementar de referência (Alternativa A). O experimento principal usa False.
        """
        self.data_dir = Path(data_root) / split
        self.gt_dir = Path(gt_root) / split
        self.transform = transform
        self.include_conc = include_concentration
        self.split = split

        files = sorted(self.data_dir.glob("*.hdf5"))
        if max_files:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(f"Nenhum HDF5 em {self.data_dir}")

        self.files = files
        self.index = []          # lista de (file_idx, traj, t), em ordem temporal
        self.sigma = {}          # (file_idx, traj, t) -> sigma do quadro
        for fi, fp in enumerate(files):
            gp = self.gt_dir / fp.name
            if not gp.exists():
                raise FileNotFoundError(f"Ground Truth ausente: {gp}. Rode ground_truth.py para o split '{split}'.")
            with h5py.File(gp, "r") as g, h5py.File(fp, "r") as f:
                valid = g["valid"][...]
                sig = g["sigma"][...]
                shape_gt = g["mask"].shape
                shape_c = f["t0_fields/concentration"].shape
                assert shape_gt == shape_c, f"Shapes diferentes em {fp.name}: GT {shape_gt} vs dados {shape_c}"
            N, T = valid.shape
            for n in range(N):
                for t in range(T):
                    if valid_only and not valid[n, t]:
                        continue
                    self.index.append((fi, n, t))
                    self.sigma[(fi, n, t)] = float(sig[n, t])

        self.n_channels = 10 + (1 if include_concentration else 0)
        self._h_data, self._h_gt = {}, {}     # handles abertos por processo/worker

    # --- handles preguiçosos (seguros com num_workers > 0)
    def _handles(self, fi):
        if fi not in self._h_data:
            self._h_data[fi] = h5py.File(self.files[fi], "r")
            self._h_gt[fi] = h5py.File(self.gt_dir / self.files[fi].name, "r")
        return self._h_data[fi], self._h_gt[fi]

    def __len__(self):
        return len(self.index)

    def read_fields(self, fi, n, t):
        """Retorna os 10 canais (e opcionalmente c-1) como array (C, H, W) float32."""
        f, _ = self._handles(fi)
        vel = f["t1_fields/velocity"][n, t]                      # (H, W, 2)
        D = f["t2_fields/D"][n, t].reshape(*vel.shape[:2], 4)    # (H, W, 4)
        E = f["t2_fields/E"][n, t].reshape(*vel.shape[:2], 4)    # (H, W, 4)
        chans = [vel, D, E]
        if self.include_conc:
            chans.append(f["t0_fields/concentration"][n, t][..., None] - 1.0)
        x = np.concatenate(chans, axis=-1)                       # (H, W, C)
        return np.ascontiguousarray(np.moveaxis(x, -1, 0)).astype(np.float32)

    def __getitem__(self, i):
        fi, n, t = self.index[i]
        _, g = self._handles(fi)
        x = self.read_fields(fi, n, t)
        y = g["mask"][n, t].astype(np.float32)[None]             # (1, H, W)
        if self.transform is not None:
            x = self.transform(x)
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        meta = {"file": self.files[fi].name, "traj": n, "t": t,
                "sigma": self.sigma[(fi, n, t)], "idx": i}
        return x, torch.from_numpy(y), meta

    def __del__(self):
        for h in list(self._h_data.values()) + list(self._h_gt.values()):
            try:
                h.close()
            except Exception:
                pass


def make_loader(data_root, gt_root, split, batch_size=8, shuffle=None, num_workers=0,
                transform=None, **kw):
    """DataLoader. Para treino, shuffle=True. Para salvar predições (Q10), shuffle=False (ordem temporal)."""
    ds = ActiveMatterSeg(data_root, gt_root, split, transform=transform, **kw)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def compute_channel_stats(ds, t_stride=10, pixel_stride=32, seed=0):
    """
    Estatísticas por canal a partir de quadros válidos (um a cada t_stride) e de uma amostra de pixels.
    Devem ser calculadas SOMENTE no split train, e servem aos pré-processamentos P1-P5
    (média/desvio, mín/máx, mediana/IQR, percentis de corte).
    """
    rng = np.random.default_rng(seed)
    pool = []
    for (fi, n, t) in ds.index:
        if t % t_stride:
            continue
        x = ds.read_fields(fi, n, t).reshape(ds.n_channels, -1)
        sel = rng.choice(x.shape[1], x.shape[1] // pixel_stride, replace=False)
        pool.append(x[:, sel])
    X = np.concatenate(pool, axis=1)
    names = CHANNEL_NAMES + (["c_minus_1"] if ds.include_conc else [])
    q = lambda p: np.percentile(X, p, axis=1)
    stats = {
        "split": ds.split, "n_frames_sampled": len(pool), "n_pixels": int(X.shape[1]),
        "channels": names,
        "mean": X.mean(1).tolist(), "std": X.std(1).tolist(),
        "min": X.min(1).tolist(), "max": X.max(1).tolist(),
        "median": q(50).tolist(), "q25": q(25).tolist(), "q75": q(75).tolist(),
        "p1": q(1).tolist(), "p99": q(99).tolist(),
    }
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--gt-root", default=Path("data/ground_truth"), type=Path)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--max-files", default=None, type=int)
    ap.add_argument("--stats-out", default=None, type=Path)
    a = ap.parse_args()

    ds = ActiveMatterSeg(a.data_root, a.gt_root, a.split, max_files=a.max_files)
    print(f"Split '{a.split}': {len(ds.files)} arquivos, {len(ds)} quadros válidos")
    x, y, meta = ds[0]
    print(f"x: {tuple(x.shape)} {x.dtype} | y: {tuple(y.shape)} {y.dtype} | meta: {meta}")
    print(f"fração positiva do 1º quadro: {y.mean():.3f}")

    loader = make_loader(a.data_root, a.gt_root, a.split, batch_size=4, shuffle=False, num_workers=0,
                         max_files=a.max_files)
    xb, yb, mb = next(iter(loader))
    print(f"batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}")
    assert xb.shape[1] == 10 and not torch.isnan(xb).any(), "Problema nos canais de entrada"

    st = compute_channel_stats(ds)
    print("\nEstatísticas por canal (amostra):")
    print(f"{'canal':8s} {'média':>9s} {'desvio':>9s} {'mín':>9s} {'máx':>9s}")
    for i, nm in enumerate(st["channels"]):
        print(f"{nm:8s} {st['mean'][i]:9.3f} {st['std'][i]:9.3f} {st['min'][i]:9.3f} {st['max'][i]:9.3f}")
    if a.stats_out:
        a.stats_out.parent.mkdir(parents=True, exist_ok=True)
        a.stats_out.write_text(json.dumps(st, indent=2))
        print(f"\nEstatísticas salvas em {a.stats_out}")
        if a.split != "train":
            print("Aviso: para os pré-processamentos finais, calcule estas estatísticas no split 'train'.")
