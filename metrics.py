"""
Etapa 7 - Métricas de avaliação (compartilhadas pelas três arquiteturas).

Segue o protocolo do professor:
  principais:     IoU e Dice (classe "alta concentração" = 1)
  complementares: Precision, Recall, latência, memória máxima, estabilidade temporal,
                  intervalos de confiança e variação entre seeds
  por classe:     além do primeiro plano, reporta-se o IoU do fundo (iou_background) e o mIoU

O F1 não é reportado porque, em segmentação binária por pixel, é numericamente igual ao Dice.

Convenções:
  - pred e target são máscaras binárias (bool/0-1). Formatos aceitos: (H,W), (B,H,W) ou (B,1,H,W).
  - Quadros inválidos já são excluídos pelo data_loader, então não há padding a ignorar (128 x 128 não
    precisa de padding).
  - Dois modos de agregação: "pooled" (somando os pixels de todos os quadros) e "frame_mean" (média das
    métricas por quadro). O IC é calculado por bootstrap sobre TRAJETÓRIAS (quadros da mesma trajetória
    são correlacionados, então reamostrar quadros subestimaria a incerteza).
  - Casos indefinidos: se não há pixels positivos nem no alvo nem na predição, IoU/Dice/Precision/Recall = 1.

Este módulo só usa numpy, exceto measure_inference_cost (importa torch quando chamada).
Teste rápido:  python metrics.py
"""
import numpy as np


def _div(num, den, default):
    return float(num) / float(den) if den > 0 else float(default)


def confusion_counts(pred, target):
    """Retorna (tp, fp, fn, tn) somados sobre todos os pixels."""
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    assert p.shape == t.shape, f"formas diferentes: {p.shape} vs {t.shape}"
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    return tp, fp, fn, tn


def scores_from_counts(tp, fp, fn, tn):
    iou = _div(tp, tp + fp + fn, 1.0)
    dice = _div(2 * tp, 2 * tp + fp + fn, 1.0)
    precision = _div(tp, tp + fp, 1.0 if fn == 0 else 0.0)
    recall = _div(tp, tp + fn, 1.0 if fp == 0 else 0.0)
    iou_bg = _div(tn, tn + fn + fp, 1.0)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall,
            "iou_background": iou_bg, "miou": 0.5 * (iou + iou_bg)}


def cluster_bootstrap_ci(values, groups, n_boot=2000, alpha=0.05, seed=0):
    """
    Média e intervalo de confiança (percentil) por bootstrap, reamostrando GRUPOS (trajetórias).
    values: valor por quadro; groups: identificador da trajetória de cada quadro.
    """
    values = np.asarray(values, dtype=np.float64)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    n_groups = inv.max() + 1
    sums = np.bincount(inv, weights=values, minlength=n_groups)
    counts = np.bincount(inv, minlength=n_groups).astype(np.float64)
    mean = float(values.mean())
    if n_groups < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return mean, float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


class MetricAccumulator:
    """Acumula contagens (pooled) e, opcionalmente, métricas por quadro (para IC)."""

    def __init__(self, keep_frames=False):
        self.tp = self.fp = self.fn = self.tn = 0
        self.keep_frames = keep_frames
        self._frames, self._groups = [], []

    def update_counts(self, tp, fp, fn, tn):
        """Atalho para quando as contagens já foram calculadas (ex.: na GPU, durante o treino)."""
        self.tp += int(tp); self.fp += int(fp); self.fn += int(fn); self.tn += int(tn)

    def update(self, pred, target, groups=None):
        p = np.asarray(pred).astype(bool)
        t = np.asarray(target).astype(bool)
        if p.ndim == 2:
            p, t = p[None], t[None]
        if p.ndim == 4 and p.shape[1] == 1:
            p, t = p[:, 0], t[:, 0]
        assert p.shape == t.shape and p.ndim == 3, f"formato inesperado: {p.shape}"
        B = p.shape[0]
        pf, tf = p.reshape(B, -1), t.reshape(B, -1)
        tp = (pf & tf).sum(1); fp = (pf & ~tf).sum(1)
        fn = (~pf & tf).sum(1); tn = (~pf & ~tf).sum(1)
        self.update_counts(tp.sum(), fp.sum(), fn.sum(), tn.sum())
        if self.keep_frames:
            if groups is None:
                groups = [-1] * B
            elif np.isscalar(groups):
                groups = [groups] * B
            for b in range(B):
                self._frames.append(scores_from_counts(int(tp[b]), int(fp[b]), int(fn[b]), int(tn[b])))
                self._groups.append(groups[b])

    def compute(self):
        """Métricas pooled (todos os pixels somados)."""
        s = scores_from_counts(self.tp, self.fp, self.fn, self.tn)
        s.update({"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                  "n_pixels": self.tp + self.fp + self.fn + self.tn})
        return s

    def frame_scores(self):
        assert self.keep_frames, "use MetricAccumulator(keep_frames=True)"
        out = {k: np.array([f[k] for f in self._frames]) for k in self._frames[0]} if self._frames else {}
        out["groups"] = np.array(self._groups)
        return out

    def summary_with_ci(self, n_boot=2000, alpha=0.05, seed=0):
        """Média por quadro e IC (bootstrap por trajetória) para as quatro métricas principais."""
        fs = self.frame_scores()
        res = {}
        for k in ("iou", "dice", "precision", "recall"):
            m, lo, hi = cluster_bootstrap_ci(fs[k], fs["groups"], n_boot, alpha, seed)
            res[k] = {"frame_mean": m, "ci_low": lo, "ci_high": hi}
        res["pooled"] = self.compute()
        return res


def temporal_stability(pred_seq, gt_seq):
    """
    Consistência entre quadros consecutivos de UMA trajetória (arrays (T,H,W) em ordem temporal).
    Compara a predição com o próprio Ground Truth, pois a máscara real também muda no tempo:
      pred_consec_iou / gt_consec_iou : IoU médio entre máscaras de quadros vizinhos
      pred_change_rate / gt_change_rate : fração média de pixels que mudam entre quadros vizinhos
      delta_iou, delta_change         : predição - GT (perto de 0 = dinâmica temporal igual à real;
                                        delta_iou muito negativo / delta_change muito positivo = "tremulação")
    O espaçamento entre quadros é o do data_loader (time_step = 3).
    """
    def consec(m):
        m = np.asarray(m).astype(bool)
        if m.shape[0] < 2:
            return np.nan, np.nan
        a, b = m[:-1], m[1:]
        inter = (a & b).sum(axis=(1, 2)).astype(float)
        union = (a | b).sum(axis=(1, 2)).astype(float)
        iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        return float(iou.mean()), float((a != b).mean())

    pi, pc = consec(pred_seq)
    gi, gc = consec(gt_seq)
    return {"pred_consec_iou": pi, "gt_consec_iou": gi, "delta_iou": pi - gi,
            "pred_change_rate": pc, "gt_change_rate": gc, "delta_change": pc - gc}


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def measure_inference_cost(model, input_shape=(1, 10, 128, 128), device="cuda", warmup=10, runs=50, amp=False):
    """
    Latência de inferência (ms por lote) e pico de memória de GPU (MB), com o mesmo tamanho de lote e
    hardware para todos os modelos. Na CPU a memória não é medida (None).
    O modelo pode expor predict_proba(x); caso contrário usa sigmoid(model(x)).
    """
    import time
    import torch

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_cuda = dev.type == "cuda"
    model = model.to(dev).eval()
    x = torch.randn(*input_shape, device=dev)
    fn = model.predict_proba if hasattr(model, "predict_proba") else (lambda z: torch.sigmoid(model(z)))

    if use_cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    times = []
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=bool(amp and use_cuda)):
        for _ in range(warmup):
            fn(x)
        for _ in range(runs):
            if use_cuda:
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            fn(x)
            if use_cuda:
                torch.cuda.synchronize(dev)
            times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {"latency_ms_mean": float(times.mean()), "latency_ms_std": float(times.std()),
            "latency_ms_median": float(np.median(times)), "batch_size": int(input_shape[0]),
            "device": str(dev), "amp": bool(amp and use_cuda),
            "peak_memory_mb": (torch.cuda.max_memory_allocated(dev) / 2 ** 20) if use_cuda else None}


# ----------------------------------------------------------------------------- autoteste
def _self_test():
    rng = np.random.default_rng(0)
    # 1) predição perfeita
    gt = rng.random((4, 32, 32)) > 0.8
    acc = MetricAccumulator()
    acc.update(gt, gt)
    r = acc.compute()
    assert r["iou"] == r["dice"] == r["precision"] == r["recall"] == 1.0
    # 2) contagens conhecidas: 100 pixels, alvo = 40 positivos, predição acerta 30 e erra 10 (fp)
    t = np.zeros(100, bool); t[:40] = True
    p = np.zeros(100, bool); p[10:50] = True            # tp=30, fp=10, fn=10, tn=50
    assert confusion_counts(p, t) == (30, 10, 10, 50)
    s = scores_from_counts(30, 10, 10, 50)
    assert abs(s["iou"] - 30 / 50) < 1e-12 and abs(s["dice"] - 60 / 80) < 1e-12
    assert abs(s["precision"] - 0.75) < 1e-12 and abs(s["recall"] - 0.75) < 1e-12
    assert abs(s["iou_background"] - 50 / 70) < 1e-12
    # 3) Dice = F1 (justifica não reportar F1)
    f1 = 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"])
    assert abs(f1 - s["dice"]) < 1e-12
    # 4) formatos (B,1,H,W) e acumulação em lotes = pooled direto
    P = rng.random((6, 1, 16, 16)) > 0.5
    T = rng.random((6, 1, 16, 16)) > 0.5
    a1, a2 = MetricAccumulator(), MetricAccumulator()
    a1.update(P, T); a2.update(P[:3], T[:3]); a2.update(P[3:], T[3:])
    assert a1.compute() == a2.compute()
    # 5) casos vazios
    z = np.zeros((1, 8, 8), bool)
    assert scores_from_counts(*confusion_counts(z, z))["dice"] == 1.0
    # 6) IC por bootstrap por trajetória
    acc = MetricAccumulator(keep_frames=True)
    for traj in range(8):
        base = rng.random((5, 16, 16)) > 0.8
        noise = rng.random((5, 16, 16)) > 0.95
        acc.update(base ^ noise, base, groups=f"traj{traj}")
    sm = acc.summary_with_ci(n_boot=500)
    d = sm["dice"]
    assert d["ci_low"] <= d["frame_mean"] <= d["ci_high"] and d["ci_high"] - d["ci_low"] > 0
    # 7) estabilidade temporal: sequência idêntica ao GT => delta 0; predição "tremendo" => delta pior
    seq = rng.random((10, 16, 16)) > 0.8
    st = temporal_stability(seq, seq)
    assert st["delta_iou"] == 0 and st["delta_change"] == 0
    shaky = np.stack([rng.random((16, 16)) > 0.8 for _ in range(10)])
    assert temporal_stability(shaky, np.repeat(seq[:1], 10, axis=0))["delta_iou"] < 0
    print("metrics.py: todos os testes passaram")


if __name__ == "__main__":
    _self_test()
