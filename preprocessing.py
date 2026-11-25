"""
Etapa 9 - Pré-processamentos P1-P10 (compartilhados pelas três arquiteturas).

Cada pré-processamento é um objeto chamável: recebe o quadro (C, h, w) float32 do data_loader e devolve
(C, h, w) float32. Uso:
    stats = load_stats("data/channel_stats_train.json")      # estatísticas SÓ do train
    T = build_transform("P6", stats)
    ds = ActiveMatterSeg(processed, gt_root, "train", transform=T)
    T.params   # dicionário com o id, o nome e os parâmetros, para registrar nos resultados

Definições (por canal, estatísticas do split train):
    P1  Z-score               (x - média) / desvio              <- também é o BASELINE
    P2  RMS                   x / rms                            (sem centralizar)
    P3  Robust scaling        (x - mediana) / (q75 - q25)
    P4  Min-Max               (x - mín) / (máx - mín)
    P5  Percentile clipping   clip em [p1, p99] e depois Z-score com a média/desvio dos dados já cortados
    P6-P10 (filtros espaciais): aplicados DEPOIS do Z-score (P1), canal a canal, 2D:
    P6  Gaussian smoothing    sigma = 1,0 px
    P7  Median filtering      janela 3 x 3
    P8  Bilateral filtering   janela 5 x 5, sigma_espacial 1,5 px, sigma_cor 0,5 (unidades Z-score)
    P9  Anisotropic diffusion Perona-Malik, 10 iterações, kappa 0,5 (unidades Z-score), passo 0,2
    P10 Wavelet denoising     db2, 2 níveis, limiar universal (VisuShrink) suave, modo periódico

O domínio da simulação é PERIÓDICO (condições de contorno periódicas no HDF5), então todos os filtros
tratam as bordas por wrap-around (np.roll, mode="wrap", pywt "periodization"), sem criar artefatos de borda.

Os parâmetros de P6-P10 são valores iniciais razoáveis; o protocolo pede seleção de hiperparâmetros na
VALIDAÇÃO e registro. Eles ficam em DEFAULT_PARAMS e podem ser sobrescritos via build_transform(params=...).

Comandos:
    python preprocessing.py --add-clip-stats --stats data/channel_stats_train.json     (uma vez, para o P5)
    python preprocessing.py                                                            (autoteste)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, median_filter

EPS = 1e-8
PRE_IDS = [f"P{i}" for i in range(1, 11)]
PRE_NAMES = {
    "P1": "Z-score", "P2": "RMS", "P3": "Robust scaling", "P4": "Min-Max", "P5": "Percentile clipping",
    "P6": "Gaussian smoothing", "P7": "Median filtering", "P8": "Bilateral filtering",
    "P9": "Anisotropic diffusion", "P10": "Wavelet denoising",
}
DEFAULT_PARAMS = {
    "P5": {"percentiles": [1, 99]},
    "P6": {"sigma": 1.0},
    "P7": {"size": 3},
    "P8": {"radius": 2, "sigma_spatial": 1.5, "sigma_color": 0.5},
    "P9": {"n_iter": 10, "kappa": 0.5, "step": 0.2},
    "P10": {"wavelet": "db2", "level": 2, "thr_scale": 1.0},
}


def load_stats(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _col(values, n_channels=None):
    a = np.asarray(values, dtype=np.float32)[:, None, None]
    return a if n_channels is None else a[:n_channels]


# ----------------------------------------------------------------------------- filtros (domínio periódico)
def gaussian_periodic(x, sigma):
    return gaussian_filter(x, sigma=(0, sigma, sigma), mode="wrap")


def median_periodic(x, size):
    return median_filter(x, size=(1, size, size), mode="wrap")


def bilateral_periodic(x, radius, sigma_spatial, sigma_color):
    num = np.zeros_like(x)
    den = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sh = np.roll(x, shift=(dy, dx), axis=(1, 2))
            w = np.exp(-(dy * dy + dx * dx) / (2.0 * sigma_spatial ** 2)) * \
                np.exp(-((sh - x) ** 2) / (2.0 * sigma_color ** 2))
            num += w * sh
            den += w
    return num / den


def anisotropic_diffusion_periodic(x, n_iter, kappa, step):
    u = x.copy()
    for _ in range(n_iter):
        n = np.roll(u, 1, axis=1) - u
        s = np.roll(u, -1, axis=1) - u
        e = np.roll(u, -1, axis=2) - u
        w = np.roll(u, 1, axis=2) - u
        u = u + step * (np.exp(-(n / kappa) ** 2) * n + np.exp(-(s / kappa) ** 2) * s +
                        np.exp(-(e / kappa) ** 2) * e + np.exp(-(w / kappa) ** 2) * w)
    return u


def wavelet_denoise(x, wavelet, level, thr_scale):
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        coeffs = pywt.wavedec2(x[c], wavelet, mode="periodization", level=level)
        sigma = np.median(np.abs(coeffs[-1][2])) / 0.6745           # ruído estimado pelo detalhe diagonal fino
        thr = thr_scale * sigma * np.sqrt(2.0 * np.log(x[c].size))
        new = [coeffs[0]] + [tuple(pywt.threshold(d, thr, mode="soft") for d in lvl) for lvl in coeffs[1:]]
        out[c] = pywt.waverec2(new, wavelet, mode="periodization")[: x.shape[1], : x.shape[2]]
    return out


# ----------------------------------------------------------------------------- transformação
class PreTransform:
    def __init__(self, pre_id, stats, params=None):
        if pre_id not in PRE_IDS:
            raise ValueError(f"pre_id deve ser um de {PRE_IDS}")
        self.pre_id = pre_id
        self.p = {**DEFAULT_PARAMS.get(pre_id, {}), **(params or {})}
        s = stats
        self.mean, self.std = _col(s["mean"]), _col(s["std"])
        self.rms = _col(s["rms"])
        self.median, self.iqr = _col(s["median"]), _col(np.asarray(s["q75"]) - np.asarray(s["q25"]))
        self.mn, self.mx = _col(s["min"]), _col(s["max"])
        self.lo, self.hi = _col(s["p1"]), _col(s["p99"])
        if pre_id == "P5":
            if "clip_mean" not in s:
                raise ValueError("P5 precisa das estatísticas dos dados cortados. Rode uma vez: "
                                 "python preprocessing.py --add-clip-stats --stats <channel_stats_train.json>")
            self.cmean, self.cstd = _col(s["clip_mean"]), _col(s["clip_std"])
        self.stats_split = s.get("split", "?")

    @property
    def params(self):
        return {"id": self.pre_id, "name": PRE_NAMES[self.pre_id], "stats_split": self.stats_split, **self.p}

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        pid = self.pre_id
        if pid == "P1":
            y = (x - self.mean) / (self.std + EPS)
        elif pid == "P2":
            y = x / (self.rms + EPS)
        elif pid == "P3":
            y = (x - self.median) / (self.iqr + EPS)
        elif pid == "P4":
            y = (x - self.mn) / (self.mx - self.mn + EPS)
        elif pid == "P5":
            y = (np.clip(x, self.lo, self.hi) - self.cmean) / (self.cstd + EPS)
        else:
            z = (x - self.mean) / (self.std + EPS)                  # filtros atuam sobre dados já em Z-score
            if pid == "P6":
                y = gaussian_periodic(z, self.p["sigma"])
            elif pid == "P7":
                y = median_periodic(z, self.p["size"])
            elif pid == "P8":
                y = bilateral_periodic(z, self.p["radius"], self.p["sigma_spatial"], self.p["sigma_color"])
            elif pid == "P9":
                y = anisotropic_diffusion_periodic(z, self.p["n_iter"], self.p["kappa"], self.p["step"])
            else:
                y = wavelet_denoise(z, self.p["wavelet"], self.p["level"], self.p["thr_scale"])
        return np.ascontiguousarray(y, dtype=np.float32)


def build_transform(pre_id, stats, params=None):
    return PreTransform(pre_id, stats, params)


# ----------------------------------------------------------------------------- estatísticas dos dados cortados (P5)
def add_clip_stats(stats_path, processed="data/processed", gt_root="data/ground_truth", split="train"):
    """Calcula média/desvio por canal DEPOIS do corte em [p1, p99] (somente no train) e grava no JSON."""
    from data_loader import ActiveMatterSeg
    stats = load_stats(stats_path)
    ds = ActiveMatterSeg(processed, gt_root, split)
    C = ds.n_channels
    lo, hi = _col(stats["p1"]).astype(np.float64), _col(stats["p99"]).astype(np.float64)
    s1, s2, npx = np.zeros(C), np.zeros(C), 0
    for (fi, n, ti) in ds.index:
        x = np.clip(ds.read_fields(fi, n, ti).astype(np.float64), lo, hi)
        s1 += x.sum(axis=(1, 2)); s2 += (x ** 2).sum(axis=(1, 2)); npx += x.shape[1] * x.shape[2]
    mean = s1 / npx
    std = np.sqrt(np.maximum(s2 / npx - mean ** 2, 0.0))
    stats["clip_mean"], stats["clip_std"] = mean.tolist(), std.tolist()
    stats["clip_percentiles"] = [1, 99]
    Path(stats_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"clip_mean/clip_std gravados em {stats_path} ({len(ds)} quadros do split '{split}')")
    return stats


# ----------------------------------------------------------------------------- autoteste
def _synthetic_stats(X):
    """X: (C, n_pixels). Mesmas chaves do data_loader.compute_channel_stats, + estatísticas do corte."""
    q = lambda p: np.percentile(X, p, axis=1)
    st = {"split": "sintético", "mean": X.mean(1), "std": X.std(1), "rms": np.sqrt((X ** 2).mean(1)),
          "min": X.min(1), "max": X.max(1), "median": q(50), "q25": q(25), "q75": q(75), "p1": q(1), "p99": q(99)}
    Xc = np.clip(X, st["p1"][:, None], st["p99"][:, None])
    st["clip_mean"], st["clip_std"] = Xc.mean(1), Xc.std(1)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()}


def _self_test():
    rng = np.random.default_rng(0)
    C, H = 10, 128
    scale = np.linspace(0.3, 4.0, C)[:, None, None]
    offs = np.linspace(-1.0, 2.0, C)[:, None, None]
    smooth = lambda: gaussian_filter(rng.standard_normal((C, H, H)), sigma=(0, 6, 6), mode="wrap")
    data = (smooth() * 8 * scale + offs).astype(np.float32)
    stats = _synthetic_stats(data.reshape(C, -1).astype(np.float64))
    x = data

    # forma, tipo e ausência de NaN em todos
    for pid in PRE_IDS:
        y = build_transform(pid, stats)(x)
        assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all(), pid
    # normalizações
    y = build_transform("P1", stats)(x).reshape(C, -1)
    assert np.allclose(y.mean(1), 0, atol=1e-4) and np.allclose(y.std(1), 1, atol=1e-4)
    y = build_transform("P2", stats)(x).reshape(C, -1)
    assert np.allclose(np.sqrt((y ** 2).mean(1)), 1, atol=1e-4)
    y = build_transform("P3", stats)(x).reshape(C, -1)
    assert np.allclose(np.median(y, 1), 0, atol=1e-3) and np.allclose(np.percentile(y, 75, 1) - np.percentile(y, 25, 1), 1, atol=1e-3)
    y = build_transform("P4", stats)(x).reshape(C, -1)
    assert np.allclose(y.min(1), 0, atol=1e-4) and np.allclose(y.max(1), 1, atol=1e-4)
    y = build_transform("P5", stats)(x).reshape(C, -1)
    assert np.allclose(y.mean(1), 0, atol=1e-3) and np.allclose(y.std(1), 1, atol=1e-3)
    # P5 sem estatísticas de corte deve falhar com mensagem clara
    bad = {k: v for k, v in stats.items() if not k.startswith("clip_")}
    try:
        build_transform("P5", bad); raise AssertionError("P5 deveria exigir clip_mean")
    except ValueError:
        pass

    # filtros: equivariância a deslocamentos periódicos (domínio periódico, sem efeito de borda)
    sh = (8, 12)
    for pid in ["P6", "P7", "P8", "P9", "P10"]:
        T = build_transform(pid, stats)
        a = T(np.roll(x, sh, axis=(1, 2)))
        b = np.roll(T(x), sh, axis=(1, 2))
        assert np.allclose(a, b, atol=1e-4), f"{pid} não respeita a periodicidade"

    # filtros: redução de ruído (em unidades Z-score, com estatísticas identidade)
    ident = {"mean": [0.0] * C, "std": [1.0] * C, "rms": [1.0] * C, "median": [0.0] * C, "q25": [-1.0] * C,
             "q75": [1.0] * C, "min": [-5.0] * C, "max": [5.0] * C, "p1": [-3.0] * C, "p99": [3.0] * C}
    clean = gaussian_filter(rng.standard_normal((C, H, H)), sigma=(0, 6, 6), mode="wrap")
    clean = (clean / clean.std()).astype(np.float32)
    noisy = clean + 0.2 * rng.standard_normal(clean.shape).astype(np.float32)
    base = np.mean((noisy - clean) ** 2)
    for pid in ["P6", "P7", "P8", "P9", "P10"]:
        err = np.mean((build_transform(pid, ident)(noisy) - clean) ** 2)
        assert err < base, f"{pid} não reduziu o ruído ({err:.4f} >= {base:.4f})"

    # P8 preserva bordas melhor que o Gaussiano
    step = np.zeros((C, H, H), np.float32); step[:, :, H // 2:] = 3.0
    g = np.abs(np.diff(build_transform("P6", ident)(step), axis=2)).max()
    b = np.abs(np.diff(build_transform("P8", ident)(step), axis=2)).max()
    assert b > g, "bilateral deveria preservar a borda melhor que o gaussiano"
    print("preprocessing.py: todos os testes passaram")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-clip-stats", action="store_true")
    ap.add_argument("--stats", type=Path, default=Path("data/channel_stats_train.json"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--gt-root", type=Path, default=Path("data/ground_truth"))
    ap.add_argument("--split", default="train", help="deve ser 'train' (estatísticas oficiais)")
    a = ap.parse_args()
    if a.add_clip_stats:
        add_clip_stats(a.stats, a.processed, a.gt_root, a.split)
    else:
        _self_test()
