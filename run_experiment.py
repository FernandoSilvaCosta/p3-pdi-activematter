"""
Etapa 11 - Execução de experimentos: liga dados + pré-processamento + modelo + treino + predições.

Cada execução (arquitetura, pré-processamento, seed) faz:
  1. monta os datasets (train/valid/test) com o pré-processamento escolhido (estatísticas do train);
  2. treina até 20 épocas, guardando o melhor checkpoint pelo Dice de validação (training.py);
  3. carrega o melhor checkpoint e salva as probabilidades de valid e test (para os pós-processamentos);
  4. mede latência e memória do modelo e o tempo do pré-processamento por quadro;
  5. grava o registro da execução em results/runs/<arq>_<P>_seed<k>.json (+ histórico por época).

O split TEST nunca é avaliado aqui: só se salvam suas probabilidades. As métricas de teste saem do
evaluation.py, depois de congelados modelos e parâmetros.

Execuções já concluídas (existe o .json) são puladas; uma execução interrompida retoma de last.pt.

Exemplos:
    python run_experiment.py --arch unet --pre P1 --seeds 0 --smoke      # teste rápido do fluxo (1 época, poucos arquivos)
    python run_experiment.py --arch unet segformer mask2former --pre P1  # Experimento 1 (baseline nas 3 arquiteturas)
    python run_experiment.py --arch unet --pre all                       # U-Net com P1..P10
    python run_experiment.py --status                                    # quadro das execuções feitas/pendentes
"""
import argparse
import json
import os
import platform
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import ActiveMatterSeg
from metrics import count_parameters, measure_inference_cost
from models import ARCHS, build_model, load_arch_config
from preprocessing import PRE_IDS, build_transform, load_stats
from training import TrainConfig, load_best, predict_split, prediction_dir, run_dir, set_seed, train


def time_preprocessing(ds, n=20):
    """Tempo médio (ms) do pré-processamento por quadro, em CPU, sobre n quadros espalhados pelo dataset."""
    idx = np.linspace(0, len(ds.index) - 1, n).astype(int)
    raws = [ds.read_fields(*ds.index[i]) for i in idx]
    t0 = time.perf_counter()
    for r in raws:
        ds.transform(r)
    return (time.perf_counter() - t0) / n * 1000.0


def run_one(args, arch, pre, seed):
    run_id = f"{arch}_{pre}_seed{seed}"
    runs_dir = args.results / "runs"
    rec_path = runs_dir / f"{run_id}.json"
    if rec_path.exists() and not args.force:
        print(f"[{run_id}] já concluída, pulando")
        return
    runs_dir.mkdir(parents=True, exist_ok=True)

    arch_cfg = load_arch_config(arch, args.configs)
    if args.model_kw:
        arch_cfg["model"] = {**arch_cfg.get("model", {}), **json.loads(args.model_kw)}
    tc = arch_cfg.get("train", {})
    cfg = TrainConfig(
        arch=arch, pre_id=pre, seed=seed, epochs=args.epochs,
        batch_size=args.batch_size or tc.get("batch_size", 16), lr=args.lr or tc.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-4), amp=bool((tc.get("amp", True) or args.amp) and not args.no_amp),
        grad_clip=tc.get("grad_clip", 1.0), num_workers=args.num_workers, patience=args.patience,
        ckpt_root=str(args.ckpt_root), pred_root=str(args.pred_root))

    stats = load_stats(args.stats)
    pre_params = json.loads(args.pre_params).get(pre) if args.pre_params else None
    T = build_transform(pre, stats, pre_params)
    mk = lambda split: ActiveMatterSeg(args.processed, args.gt_root, split, transform=T, max_files=args.max_files)
    train_ds, val_ds = mk("train"), mk("valid")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=torch.cuda.is_available())
    print(f"\n=== {run_id} | {len(train_ds)} quadros de treino, {len(val_ds)} de validação | pré: {T.params['name']} ===")

    set_seed(seed)                                   # a inicialização dos pesos também depende da seed
    model = build_model(arch, in_channels=10, **arch_cfg.get("model", {}))
    t0 = time.time()
    summary = train(model, train_loader, val_loader, cfg)
    train_time = time.time() - t0

    ck = load_best(model, cfg)
    attrs = {"arch": arch, "pre_id": pre, "seed": seed, "best_epoch": int(ck["epoch"])}
    predict_split(model, val_ds, prediction_dir(cfg), batch_size=32, amp=cfg.amp, attrs=attrs)
    if not args.no_test_pred:
        predict_split(model, mk("test"), prediction_dir(cfg), batch_size=32, amp=cfg.amp, attrs=attrs)

    cost = measure_inference_cost(model, input_shape=(1, 10, 128, 128), device="cuda", amp=False)
    record = {
        "run_id": run_id, "arch": arch, "pre_id": pre, "seed": seed, "pre_params": T.params,
        "train_config": asdict(cfg), "model_config": arch_cfg.get("model", {}),
        "n_params": count_parameters(model), "best_epoch": summary["best_epoch"],
        "epochs_run": summary["epochs_run"], "early_stopping": summary["early_stopping"],
        f"best_val_{cfg.select_metric}": summary[f"best_val_{cfg.select_metric}"],
        "val_at_best": summary["val_at_best"], "train_time_s_total": train_time,
        "cost": cost, "preprocessing_ms_per_frame": time_preprocessing(val_ds),
        "n_train_frames": len(train_ds), "n_val_frames": len(val_ds),
        "device": cost["device"], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    shutil.copy(run_dir(cfg) / "history.csv", runs_dir / f"{run_id}_history.csv")
    rec_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")        # marca de conclusão
    print(f"[{run_id}] concluída: melhor época {record['best_epoch']}, "
          f"Dice de validação {summary['val_at_best']['dice']:.4f}, IoU {summary['val_at_best']['iou']:.4f} | "
          f"latência {cost['latency_ms_mean']:.1f} ms/quadro | registro em {rec_path}")


def print_status(args):
    runs = args.results / "runs"
    print(f"{'arquitetura':12s} " + " ".join(f"{p:>4s}" for p in PRE_IDS))
    for arch in ARCHS:
        row = []
        for p in PRE_IDS:
            seeds_done = sorted(int(f.stem.split("seed")[-1]) for f in runs.glob(f"{arch}_{p}_seed*.json"))
            row.append(f"{len(seeds_done):>4d}" if seeds_done else "   -")
        print(f"{arch:12s} " + " ".join(row))
    print("(número de seeds concluídas por condição; '-' = pendente)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", choices=list(ARCHS))
    ap.add_argument("--pre", nargs="+", default=["P1"], help="P1..P10 ou 'all'")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--amp", action="store_true", help="força precisão mista mesmo se a config a desligar")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--patience", type=int, default=None, help="early stopping (o limite continua 20 épocas)")
    ap.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    ap.add_argument("--model-kw", type=str, default=None, help="JSON que sobrescreve a seção model da config")
    ap.add_argument("--pre-params", type=str, default=None, help='JSON, ex.: \'{"P6": {"sigma": 1.5}}\'')
    ap.add_argument("--stats", type=Path, default=Path("data/channel_stats_train.json"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--gt-root", type=Path, default=Path("data/ground_truth"))
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--ckpt-root", type=Path, default=Path("checkpoints"))
    ap.add_argument("--pred-root", type=Path, default=Path("predictions"))
    ap.add_argument("--max-files", type=int, default=None, help="usa só os N primeiros arquivos de cada split (teste)")
    ap.add_argument("--no-test-pred", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="teste rápido: 1 época, 3 arquivos, saídas em pastas _smoke")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.epochs, a.max_files = 1, 3
        a.results, a.ckpt_root, a.pred_root = Path("results/_smoke"), Path("checkpoints/_smoke"), Path("predictions/_smoke")
    if a.status:
        print_status(a)
    else:
        if not a.arch:
            ap.error("informe --arch (ou use --status)")
        pres = PRE_IDS if a.pre == ["all"] else a.pre
        for p in pres:
            if p not in PRE_IDS:
                ap.error(f"pré-processamento inválido: {p}")
        for arch in a.arch:
            for pre in pres:
                for seed in a.seeds:
                    run_one(a, arch, pre, seed)
