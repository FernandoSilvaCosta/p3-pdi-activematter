"""
Etapa 8 - Treinamento genérico (compartilhado por U-Net, SegFormer e Mask2Former).

Regras do protocolo do professor implementadas aqui:
  - NO MÁXIMO 20 épocas por execução (MAX_EPOCHS); mesma regra para todas as arquiteturas e condições.
  - A cada época avalia o Dice (ou IoU) no conjunto de VALIDAÇÃO e salva o melhor checkpoint
    (best_model.pt), registrando em que época ocorreu o pico (não se usa necessariamente a última época).
  - Early stopping opcional (cfg.patience); o limite continua 20 e a época de parada fica registrada.
  - Seed, hiperparâmetros e checkpoints registrados em disco.
  - Retomada automática (last.pt) para sessões do Colab que caem.

Contrato do modelo (o que cada arquitetura precisa oferecer):
  Opção simples (U-Net, SegFormer):  nn.Module com forward(x) -> logits (B,1,H,W), com H,W iguais aos da
      entrada. A perda padrão é BCE + Dice.
  Opção personalizada (Mask2Former):  além do nn.Module, definir
      model.training_loss(x, y) -> perda escalar   (y: (B,1,H,W) com 0/1)
      model.predict_proba(x)    -> probabilidades de primeiro plano (B,1,H,W) em [0,1]
  Se predict_proba não existir, usa-se sigmoid(forward(x)).

Saídas:
  checkpoints/<arq>/<P_id>/seed<k>/{best_model.pt, last.pt, history.csv, train_summary.json}
  predictions/<arq>/<P_id>/seed<k>/<split>/<arquivo>.hdf5   (probabilidades alinhadas ao Ground Truth)

Teste de fumaça (valida o fluxo com uma rede minúscula, NÃO produz resultados do trabalho):
    python training.py --smoke-test
"""
import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics import MetricAccumulator

MAX_EPOCHS = 20


@dataclass
class TrainConfig:
    arch: str = "unet"
    pre_id: str = "P1"                 # condição de pré-processamento (P1 = baseline)
    seed: int = 0
    epochs: int = 20                   # máximo permitido pelo protocolo
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True                   # precisão mista (só tem efeito em GPU)
    grad_clip: float = 1.0
    num_workers: int = 0
    threshold: float = 0.5             # binarização usada na seleção do melhor modelo
    select_metric: str = "dice"        # "dice" ou "iou"
    patience: Optional[int] = None     # None = sem early stopping
    resume: bool = True
    ckpt_root: str = "checkpoints"
    pred_root: str = "predictions"


# ----------------------------------------------------------------------------- utilidades
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(cfg):
    return Path(cfg.ckpt_root) / cfg.arch / cfg.pre_id / f"seed{cfg.seed}"


def prediction_dir(cfg):
    return Path(cfg.pred_root) / cfg.arch / cfg.pre_id / f"seed{cfg.seed}"


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def bce_dice_loss(logits, target):
    logits = logits.float()
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)


def model_loss(model, x, y):
    if hasattr(model, "training_loss"):
        return model.training_loss(x, y)
    return bce_dice_loss(model(x), y)


def model_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    return torch.sigmoid(model(x).float())


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- avaliação
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, amp=False):
    """Métricas pooled (IoU, Dice, Precision, Recall...) sobre todo o loader, com limiar fixo."""
    model.eval()
    acc = MetricAccumulator()
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            p = model_proba(model, x)
        pred = p.float() >= threshold
        tgt = y > 0.5
        acc.update_counts((pred & tgt).sum().item(), (pred & ~tgt).sum().item(),
                          (~pred & tgt).sum().item(), (~pred & ~tgt).sum().item())
    return acc.compute()


# ----------------------------------------------------------------------------- treino
def train(model, train_loader, val_loader, cfg: TrainConfig):
    if cfg.epochs > MAX_EPOCHS:
        raise ValueError(f"O protocolo limita o treino a {MAX_EPOCHS} épocas (recebido {cfg.epochs}).")
    if cfg.select_metric not in ("dice", "iou"):
        raise ValueError("select_metric deve ser 'dice' ou 'iou'")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = make_scaler(use_amp)

    history, best, bad, start_epoch, stopped_epoch = [], {"score": -1.0, "epoch": 0, "val": None}, 0, 1, None
    last_path = out / "last.pt"
    if cfg.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        history, best, bad = ck["history"], ck["best"], ck["bad"]
        start_epoch = ck["epoch"] + 1
        print(f"Retomando de {last_path} (época {ck['epoch']} concluída; melhor até aqui: época {best['epoch']})")

    n_params = int(sum(p.numel() for p in model.parameters()))
    print(f"{cfg.arch} | {cfg.pre_id} | seed {cfg.seed} | {n_params:,} parâmetros | dispositivo: {device} | "
          f"AMP: {use_amp} | épocas: {cfg.epochs}")
    t_start = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model_loss(model, x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Perda não finita na época {epoch}. Reduza o lr ou verifique a entrada.")
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        lr_now = opt.param_groups[0]["lr"]
        sched.step()

        val = evaluate(model, val_loader, device, cfg.threshold, use_amp)
        score = val[cfg.select_metric]
        improved = score > best["score"] + 1e-6
        if improved:
            best = {"score": float(score), "epoch": epoch, "val": {k: float(v) for k, v in val.items()}}
            _atomic_save({"model": model.state_dict(), "epoch": epoch, "val": best["val"],
                          "select_metric": cfg.select_metric, "cfg": asdict(cfg)}, out / "best_model.pt")
            bad = 0
        else:
            bad += 1

        history.append({"epoch": epoch, "train_loss": run_loss / max(n, 1), "val_dice": val["dice"],
                        "val_iou": val["iou"], "val_precision": val["precision"], "val_recall": val["recall"],
                        "lr": lr_now, "epoch_time_s": time.time() - t0})
        with open(out / "history.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "history": history,
                      "best": best, "bad": bad, "cfg": asdict(cfg)}, last_path)

        h = history[-1]
        print(f"Época {epoch:02d}/{cfg.epochs} | loss {h['train_loss']:.4f} | val Dice {h['val_dice']:.4f} "
              f"IoU {h['val_iou']:.4f} | lr {lr_now:.2e} | {h['epoch_time_s']:.0f}s" + (" | * melhor" if improved else ""))

        if cfg.patience is not None and bad >= cfg.patience:
            stopped_epoch = epoch
            print(f"Early stopping na época {epoch} (paciência {cfg.patience}).")
            break

    summary = {"arch": cfg.arch, "pre_id": cfg.pre_id, "seed": cfg.seed,
               "epochs_max": cfg.epochs, "epochs_run": history[-1]["epoch"] if history else 0,
               "best_epoch": best["epoch"], "select_metric": cfg.select_metric,
               f"best_val_{cfg.select_metric}": best["score"], "val_at_best": best["val"],
               "early_stopping": cfg.patience is not None, "patience": cfg.patience,
               "stopped_epoch": stopped_epoch, "n_params": n_params, "device": str(device), "amp": use_amp,
               "train_time_s_this_session": time.time() - t_start, "torch": torch.__version__, "cfg": asdict(cfg)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Melhor época: {best['epoch']} ({cfg.select_metric} de validação = {best['score']:.4f}) | "
          f"resumo em {out / 'train_summary.json'}")
    return summary


def load_best(model, cfg, device=None):
    """Carrega best_model.pt (para predição/avaliação final). Retorna o dicionário do checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run_dir(cfg) / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return ck


# ----------------------------------------------------------------------------- predições
@torch.no_grad()
def predict_split(model, dataset, out_dir, batch_size=32, amp=False, num_workers=0, attrs=None):
    """
    Salva as probabilidades de um split (valid ou test) em <out_dir>/<split>/<arquivo>.hdf5, no mesmo
    layout do Ground Truth:  prob (N,Ts,h,w) float32 (NaN nos quadros inválidos excluídos),
    valid (N,Ts), time_index (Ts,). As probabilidades salvas alimentam os pós-processamentos (Q1-Q5)
    sem retreinar. Use um dataset sem embaralhar (a ordem não importa aqui, o índice garante o alinhamento).
    """
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    shapes = {}
    for fi, fp in enumerate(dataset.files):
        with h5py.File(dataset.gt_dir / fp.name, "r") as g:
            shapes[fi] = g["mask"].shape                       # (N, Ts, h, w)
    probs = {fi: np.full(s, np.nan, dtype=np.float32) for fi, s in shapes.items()}

    for x, _, meta in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            p = model_proba(model, x)
        p = p.float().cpu().numpy()[:, 0]                      # (B, h, w)
        for b, idx in enumerate(meta["idx"].tolist()):
            fi, n, ti = dataset.index[idx]
            probs[fi][n, ti] = p[b]

    out_split = Path(out_dir) / dataset.split
    out_split.mkdir(parents=True, exist_ok=True)
    for fi, fp in enumerate(dataset.files):
        with h5py.File(dataset.gt_dir / fp.name, "r") as g:
            valid, t_idx = g["valid"][...], g["time_index"][...]
        with h5py.File(out_split / fp.name, "w") as f:
            f.create_dataset("prob", data=probs[fi], compression="gzip", compression_opts=1)
            f.create_dataset("valid", data=valid)
            f.create_dataset("time_index", data=t_idx)
            f.attrs["source_file"] = fp.name
            for k, v in (attrs or {}).items():
                f.attrs[k] = v
    print(f"Probabilidades salvas em {out_split} ({len(dataset.files)} arquivos)")
    return out_split


# ----------------------------------------------------------------------------- teste de fumaça
class TinyNet(nn.Module):
    """Rede minúscula só para validar o fluxo (não é uma das arquiteturas do trabalho)."""

    def __init__(self, c_in=10):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        return self.net(x)


def _smoke_test(a):
    from data_loader import ActiveMatterSeg
    from metrics import count_parameters, measure_inference_cost

    def standardize(x):                                        # só para o teste: normalização por amostra
        return (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)

    cfg = TrainConfig(arch="tiny", pre_id="P1", seed=0, epochs=a.epochs, batch_size=16, amp=True,
                      resume=False, ckpt_root="checkpoints/_smoke_test", pred_root="predictions/_smoke_test")
    ds = ActiveMatterSeg(a.processed, a.gt_root, "valid", transform=standardize, max_files=a.max_files)
    print(f"[smoke] {len(ds)} quadros (o mesmo conjunto serve de treino e validação: é só um teste de fluxo)")
    mk = lambda shuffle: DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0)

    model = TinyNet(10)
    summary = train(model, mk(True), mk(False), cfg)
    load_best(model, cfg)
    out = predict_split(model, ds, prediction_dir(cfg), batch_size=32,
                        attrs={"arch": cfg.arch, "pre_id": cfg.pre_id, "seed": cfg.seed,
                               "best_epoch": summary["best_epoch"]})
    f = next(Path(out).glob("*.hdf5"))
    with h5py.File(f, "r") as h:
        prob = h["prob"][...]
        print(f"[smoke] arquivo de predição {f.name}: prob {prob.shape}, "
              f"fração NaN (quadros inválidos) = {np.isnan(prob).mean():.3f}, "
              f"min/max = {np.nanmin(prob):.3f}/{np.nanmax(prob):.3f}")
    cost = measure_inference_cost(model, device="cuda", warmup=2, runs=5)
    print(f"[smoke] parâmetros: {count_parameters(model):,} | latência: {cost['latency_ms_mean']:.2f} ms/lote | "
          f"memória de pico: {cost['peak_memory_mb']}")
    print("[smoke] fluxo concluído. Os valores de Dice/IoU acima NÃO são resultados do trabalho.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--processed", default=Path("data/processed"), type=Path)
    ap.add_argument("--gt-root", default=Path("data/ground_truth"), type=Path)
    ap.add_argument("--epochs", default=2, type=int)
    ap.add_argument("--max-files", default=4, type=int)
    a = ap.parse_args()
    if a.smoke_test:
        _smoke_test(a)
    else:
        print("Este módulo é usado pelos scripts de experimento. Para validar o fluxo: python training.py --smoke-test")
