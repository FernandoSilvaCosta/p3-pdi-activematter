"""
Etapa 12 - Pós-processamentos Q1-Q5 (aplicados às probabilidades salvas de cada modelo, sem retreinar).

    Q1  Threshold optimization             limiar de binarização escolhido na validação (maximiza o Dice)
    Q2  Morphological opening              abertura com elemento em disco de raio r
    Q3  Morphological closing              fechamento com elemento em disco de raio r
    Q4  Connected-component filtering      remove componentes com área < min_area (pixels)
    Q5  Temporal consistency filtering     média móvel das probabilidades ao longo do tempo (janela w)

Convenções:
  - Cada Q é aplicado INDIVIDUALMENTE sobre a binarização padrão (limiar 0,5), exceto o Q1, que troca o limiar.
    A condição "sem pós-processamento" (id None) é a binarização em 0,5.
  - O domínio é PERIÓDICO: morfologia e componentes conexos tratam as bordas por wrap-around (um objeto que
    sai por um lado e entra pelo oposto é um único componente).
  - Q5 atua sobre sequências temporais por trajetória, usando só os quadros válidos consecutivos; o espaçamento
    entre quadros é o do data_loader (time_step = 3).
  - Os parâmetros de cada Q são escolhidos NA VALIDAÇÃO (maior Dice agregado; em empate, o menos agressivo) e
    congelados antes de aplicar ao teste (select_params).

Este módulo usa só numpy, scipy e h5py. Autoteste:  python postprocessing.py
"""
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

Q_IDS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_NAMES = {"Q1": "Threshold optimization", "Q2": "Morphological opening", "Q3": "Morphological closing",
           "Q4": "Connected-component filtering", "Q5": "Temporal consistency filtering"}
BASE_THRESHOLD = 0.5
GRIDS = {"Q2": [1, 2, 3], "Q3": [1, 2, 3], "Q4": [4, 10, 25, 50, 100, 200, 400], "Q5": [3, 5]}
PARAM_NAME = {"Q2": "radius", "Q3": "radius", "Q4": "min_area", "Q5": "window"}


# ----------------------------------------------------------------------------- operações (domínio periódico)
def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _periodic_morph(mask, op, r):
    pad = 2 * r + 1                                           # folga para o efeito de borda de erosão + dilatação
    out = op(np.pad(mask, pad, mode="wrap"), structure=_disk(r))
    return out[pad:-pad, pad:-pad]


def opening(mask, radius):
    return _periodic_morph(mask, ndimage.binary_opening, radius)


def closing(mask, radius):
    return _periodic_morph(mask, ndimage.binary_closing, radius)


def remove_small_components(mask, min_area):
    """Remove componentes (4-conectados, com wrap-around) com área menor que min_area."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask.copy()
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    for i in range(mask.shape[0]):                            # une componentes que atravessam a borda esquerda/direita
        union(lab[i, 0], lab[i, -1])
    for j in range(mask.shape[1]):                            # e a borda superior/inferior
        union(lab[0, j], lab[-1, j])
    roots = np.array([find(i) for i in range(n + 1)])
    areas = np.bincount(roots[lab.ravel()], minlength=n + 1)
    keep = areas[roots] >= min_area
    keep[0] = False
    return keep[lab]


def temporal_smooth(prob, window):
    return ndimage.uniform_filter1d(prob, size=window, axis=0, mode="nearest")


def apply_post(qid, prob, params=None):
    """prob: (T, h, w) probabilidades de UMA sequência temporal -> máscara booleana (T, h, w)."""
    if qid is None:
        return prob >= BASE_THRESHOLD
    if qid == "Q1":
        return prob >= params["threshold"]
    if qid == "Q5":
        return temporal_smooth(prob, params["window"]) >= BASE_THRESHOLD
    m = prob >= BASE_THRESHOLD
    if qid == "Q2":
        return np.stack([opening(f, params["radius"]) for f in m])
    if qid == "Q3":
        return np.stack([closing(f, params["radius"]) for f in m])
    if qid == "Q4":
        return np.stack([remove_small_components(f, params["min_area"]) for f in m])
    raise ValueError(f"pós-processamento desconhecido: {qid}")


# ----------------------------------------------------------------------------- leitura das predições
class Sequence:
    """Quadros válidos consecutivos de uma trajetória: probabilidades, Ground Truth e instantes de tempo."""

    def __init__(self, file, traj, prob, gt, t_index):
        self.file, self.traj, self.prob, self.gt, self.t_index = file, traj, prob, gt, t_index
        self.group = f"{file}|{traj}"


def load_sequences(pred_dir, gt_dir):
    """Lê <pred_dir>/<arquivo>.hdf5 (prob, valid) e <gt_dir>/<arquivo>.hdf5 (mask) e devolve a lista de Sequence."""
    seqs = []
    for fp in sorted(Path(pred_dir).glob("*.hdf5")):
        gp = Path(gt_dir) / fp.name
        if not gp.exists():
            raise FileNotFoundError(f"Ground Truth ausente: {gp}")
        with h5py.File(fp, "r") as p, h5py.File(gp, "r") as g:
            prob, valid, t_idx = p["prob"][...], p["valid"][...], p["time_index"][...]
            gt = g["mask"][...].astype(bool)
            assert prob.shape == gt.shape, f"formas diferentes em {fp.name}: {prob.shape} vs {gt.shape}"
            assert np.array_equal(valid, g["valid"][...]), f"flags de validade diferentes em {fp.name}"
        for n in range(prob.shape[0]):
            idx = np.where(valid[n])[0]
            if len(idx) == 0:
                continue
            for run in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                pr = prob[n, run]
                if np.isnan(pr).any():
                    raise ValueError(f"predição ausente (NaN) em quadros válidos de {fp.name}, trajetória {n}")
                seqs.append(Sequence(fp.name, n, pr, gt[n, run], t_idx[run]))
    if not seqs:
        raise FileNotFoundError(f"Nenhuma predição encontrada em {pred_dir}")
    return seqs


# ----------------------------------------------------------------------------- seleção de parâmetros (validação)
def _dice_from(tp, fp, fn):
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)


def pooled_dice(seqs, qid, params):
    tp = fp = fn = 0
    for s in seqs:
        pred = apply_post(qid, s.prob, params)
        tp += int(np.logical_and(pred, s.gt).sum())
        fp += int(np.logical_and(pred, ~s.gt).sum())
        fn += int(np.logical_and(~pred, s.gt).sum())
    return _dice_from(tp, fp, fn)


def optimize_threshold(seqs, bins=1000):
    """Limiar que maximiza o Dice agregado (via histogramas das probabilidades dos pixels positivos e negativos)."""
    pos, neg = np.zeros(bins), np.zeros(bins)
    for s in seqs:
        pos += np.histogram(s.prob[s.gt], bins=bins, range=(0, 1))[0]
        neg += np.histogram(s.prob[~s.gt], bins=bins, range=(0, 1))[0]
    tp = pos[::-1].cumsum()[::-1]
    fp = neg[::-1].cumsum()[::-1]
    fn = pos.sum() - tp
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    ks = np.where(dice >= dice.max() - 1e-12)[0]              # em caso de platô de empate, usa o limiar do meio
    k = int(ks[len(ks) // 2])
    return k / bins, float(dice[k])


def select_params(seqs):
    """Escolhe, na validação, o parâmetro de cada Q. Retorna um dicionário pronto para gravar em JSON."""
    out = {"baseline": {"params": {"threshold": BASE_THRESHOLD}, "val_dice": pooled_dice(seqs, None, None)}}
    thr, d = optimize_threshold(seqs)
    out["Q1"] = {"params": {"threshold": thr}, "val_dice": d}
    for q in ("Q2", "Q3", "Q4", "Q5"):
        cand = {v: pooled_dice(seqs, q, {PARAM_NAME[q]: v}) for v in GRIDS[q]}
        best = max(cand, key=lambda v: (cand[v], -v))         # empate: o parâmetro menos agressivo
        out[q] = {"params": {PARAM_NAME[q]: best}, "val_dice": cand[best],
                  "candidates": {str(k): v for k, v in cand.items()}}
    return out


# ----------------------------------------------------------------------------- autoteste
def _self_test():
    rng = np.random.default_rng(0)
    H = 64
    # 1) abertura/fechamento equivariantes a deslocamentos periódicos
    m = rng.random((H, H)) > 0.7
    for op in (opening, closing):
        a = op(np.roll(m, (9, 13), axis=(0, 1)), 1)
        b = np.roll(op(m, 1), (9, 13), axis=(0, 1))
        assert np.array_equal(a, b), "morfologia não respeita a periodicidade"
    # abertura remove pixels isolados; fechamento fecha um buraco de 1 pixel
    iso = np.zeros((H, H), bool); iso[10, 10] = True
    assert not opening(iso, 1).any()
    hole = np.ones((H, H), bool); hole[20, 20] = False
    assert closing(hole, 1)[20, 20]
    # 2) componentes conexos: um objeto que cruza a borda conta como UM componente (área somada)
    cc = np.zeros((H, H), bool)
    cc[10:14, 0:3] = True; cc[10:14, H - 3:] = True            # 12 + 12 = 24 pixels, ligados pela borda
    cc[40, 40] = True                                          # 1 pixel isolado
    kept = remove_small_components(cc, 20)
    assert kept[10, 0] and kept[10, H - 1] and not kept[40, 40], "união periódica de componentes falhou"
    kept = remove_small_components(cc, 30)
    assert not kept.any(), "componente de 24 px deveria ser removido com min_area=30"
    # 3) limiar ótimo: probabilidades separáveis em 0,3 => limiar entre as classes
    gt = rng.random((2, 4, 32, 32)) > 0.8
    prob = np.where(gt, 0.35, 0.25) + 0.01 * rng.random(gt.shape)
    seqs = [Sequence("f", n, prob[n].astype(np.float32), gt[n], np.arange(4)) for n in range(2)]
    thr, d = optimize_threshold(seqs)
    assert 0.26 < thr <= 0.35 and d > 0.99, (thr, d)
    # 4) Q5: a média temporal reduz ruído independente entre quadros
    base = rng.random((H, H)) > 0.8
    seq_gt = np.repeat(base[None], 8, axis=0)
    noisy = np.clip(seq_gt * 0.6 + 0.2 + 0.35 * rng.standard_normal(seq_gt.shape), 0, 1).astype(np.float32)
    s = Sequence("g", 0, noisy, seq_gt, np.arange(8))
    d0, d5 = pooled_dice([s], None, None), pooled_dice([s], "Q5", {"window": 5})
    assert d5 > d0, (d0, d5)
    # 5) seleção de parâmetros devolve todos os Q e respeita o empate (parâmetro menos agressivo)
    sel = select_params([s])
    assert set(sel) == {"baseline", "Q1", "Q2", "Q3", "Q4", "Q5"} and sel["Q5"]["val_dice"] >= d5 - 1e-12
    print("postprocessing.py: todos os testes passaram")


if __name__ == "__main__":
    _self_test()
