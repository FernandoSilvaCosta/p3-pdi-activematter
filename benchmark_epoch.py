"""
Benchmark de velocidade: estima o tempo de uma época de cada arquitetura e o custo de cada pré-processamento,
para planejar os 30 treinos (3 arquiteturas x 10 condições) e o número de sementes.

Mede (no hardware em que for executado, de preferência a GPU de treino):
  - segundos por passo de treino (inclui leitura de dados e pré-processamento P1), pico de memória de GPU;
  - tempo de uma passada de validação;
  - estimativa de uma época e de 20 épocas por arquitetura, e de um conjunto de 10 condições;
  - tempo do pré-processamento por quadro (P1..P10) em CPU e o que ele soma a uma época com 1 processo de leitura.

Não grava resultados do trabalho, só imprime o quadro e salva results/benchmark.json.

Uso:
    python benchmark_epoch.py                      # as 3 arquiteturas, 30 passos cada
    python benchmark_epoch.py --archs unet --steps 50
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import ActiveMatterSeg
from models import ARCHS, build_model, load_arch_config
from models.common import count_parameters
from preprocessing import PRE_IDS, build_transform, load_stats
from training import make_scaler, model_loss, model_proba, set_seed


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def bench_arch(arch, args, stats, dev):
    cfg = load_arch_config(arch, args.configs)
    tc, bs = cfg.get("train", {}), args.batch_size or cfg.get("train", {}).get("batch_size", 16)
    amp_cfg = {"on": True, "off": False}.get(args.amp, bool(tc.get("amp", True)))      # --amp on/off sobrescreve a config
    amp = bool(amp_cfg and dev.type == "cuda")
    T = build_transform("P1", stats)
    train_ds = ActiveMatterSeg(args.processed, args.gt_root, "train", transform=T)
    val_ds = ActiveMatterSeg(args.processed, args.gt_root, "valid", transform=T)
    nw = args.num_workers
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    set_seed(0)
    model_kw = {**cfg.get("model", {}), **(json.loads(args.model_kw) if args.model_kw else {})}
    model = build_model(arch, in_channels=10, **model_kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.get("lr", 1e-3), weight_decay=tc.get("weight_decay", 1e-4))
    scaler = make_scaler(amp)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    model.train()
    times, losses, it = [], [], iter(tl)
    for i in range(args.warmup + args.steps):
        x, y, _ = next(it)
        sync(dev)
        t0 = time.perf_counter()
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, enabled=amp):
            loss = model_loss(model, x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
        scaler.step(opt)
        scaler.update()
        sync(dev)
        losses.append(float(loss.detach()))
        if i >= args.warmup:
            times.append(time.perf_counter() - t0)
    s_step = float(np.mean(times))
    loss_ok = bool(np.all(np.isfinite(losses)))
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else None

    model.eval()
    vt = []
    with torch.no_grad():
        for i, (x, _, _) in enumerate(vl):
            if i >= args.val_batches:
                break
            sync(dev)
            t0 = time.perf_counter()
            with torch.autocast(device_type=dev.type, enabled=amp):
                model_proba(model, x.to(dev))
            sync(dev)
            vt.append(time.perf_counter() - t0)
    s_val_batch = float(np.mean(vt))

    steps_epoch = len(train_ds) // bs
    epoch_s = s_step * steps_epoch + s_val_batch * math.ceil(len(val_ds) / bs)
    return {"arch": arch, "params": count_parameters(model), "batch": bs, "amp": amp, "s_per_step": s_step,
            "steps_per_epoch": steps_epoch, "val_s_per_batch": s_val_batch, "epoch_s": epoch_s,
            "run20_min": epoch_s * 20 / 60, "ten_conditions_h": epoch_s * 20 * 10 / 3600, "peak_mem_gb": peak,
            "n_train": len(train_ds), "n_val": len(val_ds), "loss_finite": loss_ok,
            "loss_first": float(losses[0]), "loss_last": float(losses[-1])}


def bench_preprocessing(args, stats):
    ds = ActiveMatterSeg(args.processed, args.gt_root, "train", max_files=6)
    idx = np.linspace(0, len(ds.index) - 1, args.pre_frames).astype(int)
    raws = [ds.read_fields(*ds.index[i]) for i in idx]
    n_train = len(ActiveMatterSeg(args.processed, args.gt_root, "train"))
    out = {}
    for pid in PRE_IDS:
        T = build_transform(pid, stats)
        T(raws[0])
        t0 = time.perf_counter()
        for r in raws:
            T(r)
        ms = (time.perf_counter() - t0) / len(raws) * 1000
        out[pid] = {"ms_per_frame": ms, "s_per_epoch_one_worker": ms / 1000 * n_train}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--pre-frames", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--model-kw", type=str, default=None,
                    help='JSON que sobrescreve a seção model da config, ex.: \'{"decoder_layers": 4}\'')
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                    help="precisão mista (auto = valor do configs/<arq>.yaml)")
    ap.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    ap.add_argument("--stats", type=Path, default=Path("data/channel_stats_train.json"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--gt-root", type=Path, default=Path("data/ground_truth"))
    ap.add_argument("--configs", default="configs")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU (tempos NÃO representativos do treino em GPU)"
    print(f"dispositivo: {gpu}\n")
    stats = load_stats(a.stats)

    rows = []
    for arch in a.archs:
        r = bench_arch(arch, a, stats, dev)
        rows.append(r)
        mem = f"{r['peak_mem_gb']:.1f} GB" if r["peak_mem_gb"] is not None else "n/d"
        print(f"{arch:12s} | {r['params'] / 1e6:6.1f} M param | lote {r['batch']:>2d} | {r['s_per_step']:.3f} s/passo "
              f"| época ~ {r['epoch_s'] / 60:5.1f} min | 20 épocas ~ {r['run20_min']:5.1f} min | "
              f"10 condições ~ {r['ten_conditions_h']:4.1f} h | pico de memória {mem} | AMP: {'sim' if r['amp'] else 'não'}"
              f" | perda {r['loss_first']:.3f} -> {r['loss_last']:.3f}" + ("" if r["loss_finite"] else "  [ATENÇÃO: perda não finita]"))

    pre = bench_preprocessing(a, stats)
    print("\nPré-processamento por quadro (CPU, 1 processo):")
    for pid, v in pre.items():
        print(f"  {pid:3s}: {v['ms_per_frame']:6.2f} ms/quadro  (+{v['s_per_epoch_one_worker']:5.0f} s por época com 1 leitor)")

    tot = sum(r["ten_conditions_h"] for r in rows)
    print(f"\nEstimativa para 1 seed das condições P1..P10 nas arquiteturas medidas: ~ {tot:.1f} h (sem os filtros acima)")
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps({"device": gpu, "archs": rows, "preprocessing": pre}, indent=2),
                                              encoding="utf-8")
    print("Resultados salvos em results/benchmark.json")
