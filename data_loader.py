"""
Etapa 6 - Pipeline de dados (versão 2): dados processados em 128 px + Ground Truth v2.

Lê, de forma preguiçosa, os arquivos gerados por prepare_dataset.py e ground_truth.py:
    <processed_root>/<split>/<arquivo>.hdf5   (fields, concentration, time_index)
    <gt_root>/<split>/<arquivo>.hdf5          (mask, sigma, valid, time_index)

Cada amostra é UM quadro (trajetória n, instante t):
    x     (10, 128, 128) float32   canais físicos reduzidos, SEM normalização (a normalização do
                                   baseline = Z-score = P1 e os demais P2-P10 entram via `transform`)
    y     (1, 128, 128)  float32   máscara Ground Truth (0/1)
    meta  dict                     arquivo, traj, t (índice de tempo ORIGINAL), sigma, idx

Ordem dos canais (a concentração NÃO entra, salvo include_concentration=True):
    0 vel_x  1 vel_y  2 D_xx  3 D_xy  4 D_yx  5 D_yy  6 E_xx  7 E_xy  8 E_yx  9 E_yy

Quadros com valid=False no Ground Truth são excluídos por padrão. A ordem dos índices é temporal por
arquivo e trajetória (necessário para o Q5 e para a estabilidade temporal): use `sequences()` para obter,
por trajetória, a lista ordenada de índices do dataset. O espaçamento entre quadros é `time_step` (=3).

Uso (teste e estatísticas de canal):
    python data_loader.py --split valid --stats-out ./data/channel_stats_valid.json
As estatísticas que valem para o projeto são as do split 'train' (--split train).
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
    def __init__(self, processed_root, gt_root, split, transform=None, valid_only=True,
                 include_concentration=False, max_files=None):
        """
        include_concentration=True acrescenta um 11o canal (c - 1), apenas para o experimento
        complementar de referência (Alternativa A). O experimento principal usa False.
        """
        self._hp, self._hg = {}, {}          # handles abertos por processo/worker (definidos primeiro: __del__ seguro)
        self.split = split
        self.proc_dir = Path(processed_root) / split
        self.gt_dir = Path(gt_root) / split
        self.transform = transform
        self.include_conc = include_concentration

        files = sorted(self.proc_dir.glob("*.hdf5"))
        if max_files:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo processado em {self.proc_dir}. Rode prepare_dataset.py.")
        self.files = files

        self.index, self.sigmas, self.time_index = [], [], []     # index: (file_idx, traj, pos_tempo)
        for fi, fp in enumerate(files):
            gp = self.gt_dir / fp.name
            if not gp.exists():
                raise FileNotFoundError(f"Ground Truth ausente: {gp}. Rode ground_truth.py para '{split}'.")
            with h5py.File(fp, "r") as p, h5py.File(gp, "r") as g:
                t_p, t_g = p["time_index"][...], g["time_index"][...]
                assert np.array_equal(t_p, t_g), f"time_index diferente entre dados e GT em {fp.name}"
                assert g["mask"].shape == p["concentration"].shape, f"Shapes diferentes em {fp.name}"
                valid, sig = g["valid"][...], g["sigma"][...]
            self.time_index.append(t_p)
            N, Ts = valid.shape
            for n in range(N):
                for ti in range(Ts):
                    if valid_only and not valid[n, ti]:
                        continue
                    self.index.append((fi, n, ti))
                    self.sigmas.append(float(sig[n, ti]))

        t0 = self.time_index[0]
        self.time_step = int(t0[1] - t0[0]) if len(t0) > 1 else 1
        self.n_channels = 10 + (1 if include_concentration else 0)

    def _handles(self, fi):
        if fi not in self._hp:
            self._hp[fi] = h5py.File(self.files[fi], "r")
            self._hg[fi] = h5py.File(self.gt_dir / self.files[fi].name, "r")
        return self._hp[fi], self._hg[fi]

    def __len__(self):
        return len(self.index)

    def read_fields(self, fi, n, ti):
        """Canais de entrada de um quadro, array (C, h, w) float32, sem normalização."""
        p, _ = self._handles(fi)
        x = p["fields"][n, ti]
        if self.include_conc:
            x = np.concatenate([x, (p["concentration"][n, ti] - 1.0)[None]], axis=0)
        return x.astype(np.float32, copy=False)

    def __getitem__(self, i):
        fi, n, ti = self.index[i]
        _, g = self._handles(fi)
        x = self.read_fields(fi, n, ti)
        y = g["mask"][n, ti].astype(np.float32)[None]
        if self.transform is not None:
            x = self.transform(x)
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        meta = {"file": self.files[fi].name, "traj": n, "t": int(self.time_index[fi][ti]),
                "sigma": self.sigmas[i], "idx": i}
        return x, torch.from_numpy(y), meta

    def sequences(self):
        """{(arquivo, trajetória): [índices do dataset em ordem temporal]} (para Q5 e estabilidade temporal)."""
        seqs = {}
        for i, (fi, n, _) in enumerate(self.index):
            seqs.setdefault((self.files[fi].name, n), []).append(i)
        return seqs

    def __del__(self):
        for h in list(self._hp.values()) + list(self._hg.values()):
            try:
                h.close()
            except Exception:
                pass


def make_loader(processed_root, gt_root, split, batch_size=16, shuffle=None, num_workers=0,
                transform=None, **kw):
    """Treino: shuffle=True. Para salvar predições (Q5), shuffle=False mantém a ordem temporal."""
    ds = ActiveMatterSeg(processed_root, gt_root, split, transform=transform, **kw)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def compute_channel_stats(ds, pixel_stride=16, seed=0):
    """
    Estatísticas por canal sobre os quadros do dataset (devem vir do split 'train').
    Exatas (todos os pixels): média, desvio, rms, mínimo, máximo.
    Aproximadas (amostra de 1 a cada `pixel_stride` pixels): mediana, quartis, percentis 1 e 99.
    Servem aos pré-processamentos P1 (Z-score), P2 (RMS), P3 (Robust), P4 (Min-Max), P5 (clipping).
    """
    rng = np.random.default_rng(seed)
    C = ds.n_channels
    s1, s2 = np.zeros(C), np.zeros(C)
    mn, mx = np.full(C, np.inf), np.full(C, -np.inf)
    npx, pool = 0, []
    for (fi, n, ti) in ds.index:
        x = ds.read_fields(fi, n, ti).reshape(C, -1).astype(np.float64)
        s1 += x.sum(1); s2 += (x ** 2).sum(1)
        mn, mx = np.minimum(mn, x.min(1)), np.maximum(mx, x.max(1))
        npx += x.shape[1]
        sel = rng.choice(x.shape[1], x.shape[1] // pixel_stride, replace=False)
        pool.append(x[:, sel].astype(np.float32))
    X = np.concatenate(pool, axis=1)
    mean = s1 / npx
    var = np.maximum(s2 / npx - mean ** 2, 0.0)
    q = lambda p: np.percentile(X, p, axis=1)
    names = CHANNEL_NAMES + (["c_minus_1"] if ds.include_conc else [])
    return {
        "split": ds.split, "n_frames": len(ds.index), "n_pixels": int(npx), "channels": names,
        "mean": mean.tolist(), "std": np.sqrt(var).tolist(), "rms": np.sqrt(s2 / npx).tolist(),
        "min": mn.tolist(), "max": mx.tolist(),
        "median": q(50).tolist(), "q25": q(25).tolist(), "q75": q(75).tolist(),
        "p1": q(1).tolist(), "p99": q(99).tolist(),
        "note": "mean/std/rms/min/max exatos; median/q25/q75/p1/p99 aproximados (amostra de pixels)",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=Path("data/processed"), type=Path)
    ap.add_argument("--gt-root", default=Path("data/ground_truth"), type=Path)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--max-files", default=None, type=int)
    ap.add_argument("--stats-out", default=None, type=Path)
    a = ap.parse_args()

    ds = ActiveMatterSeg(a.processed, a.gt_root, a.split, max_files=a.max_files)
    seqs = ds.sequences()
    print(f"Split '{a.split}': {len(ds.files)} arquivos, {len(seqs)} trajetórias, "
          f"{len(ds)} quadros válidos, espaçamento temporal = {ds.time_step}")
    x, y, meta = ds[0]
    print(f"x: {tuple(x.shape)} {x.dtype} | y: {tuple(y.shape)} {y.dtype} | meta: {meta}")
    print(f"fração positiva do 1º quadro: {y.mean():.3f}")
    k0 = next(iter(seqs))
    print(f"1ª sequência {k0}: {len(seqs[k0])} quadros, t = "
          f"{[ds[i][2]['t'] for i in seqs[k0][:4]]} ...")

    loader = make_loader(a.processed, a.gt_root, a.split, batch_size=4, shuffle=False, max_files=a.max_files)
    xb, yb, mb = next(iter(loader))
    print(f"batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}")
    assert xb.shape[1] == 10 and not torch.isnan(xb).any(), "Problema nos canais de entrada"

    st = compute_channel_stats(ds)
    print("\nEstatísticas por canal:")
    print(f"{'canal':8s} {'média':>9s} {'desvio':>9s} {'rms':>9s} {'mín':>9s} {'máx':>9s}")
    for i, nm in enumerate(st["channels"]):
        print(f"{nm:8s} {st['mean'][i]:9.3f} {st['std'][i]:9.3f} {st['rms'][i]:9.3f} "
              f"{st['min'][i]:9.3f} {st['max'][i]:9.3f}")
    if a.stats_out:
        a.stats_out.parent.mkdir(parents=True, exist_ok=True)
        a.stats_out.write_text(json.dumps(st, indent=2), encoding="utf-8")
        print(f"\nEstatísticas salvas em {a.stats_out}")
        if a.split != "train":
            print("Aviso: para os pré-processamentos finais, calcule estas estatísticas no split 'train'.")
