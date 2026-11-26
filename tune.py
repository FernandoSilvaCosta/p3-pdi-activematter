"""
Ajuste de hiperparâmetros na VALIDAÇÃO (o protocolo manda selecionar hiperparâmetros na validação, nunca no teste).

Dois ajustes, ambos com seed 0 e o critério "maior Dice de validação no melhor checkpoint":
  lr       taxa de aprendizado de cada arquitetura, na condição baseline (P1)
  filters  parâmetros dos filtros P6..P10, usando uma arquitetura de referência (padrão: U-Net, lr de configs/)

Cada configuração testada é uma execução do run_experiment.py com --tag, gravada em results/_tune/<tag>/
(sem predições de teste). Configurações já feitas são puladas, então o comando pode ser repetido após uma queda.

Uso:
    python tune.py lr --arch unet segformer               # 3 valores de lr por arquitetura
    python tune.py lr --arch mask2former --epochs 8       # o Mask2Former é mais lento: versão curta
    python tune.py filters --arch unet --epochs 10        # parâmetros de P6..P10
    python tune.py summary                                # tabela com o que já foi rodado
    python tune.py lr --arch unet --dry-run               # só mostra os comandos

Depois de escolher: atualize o lr em configs/<arq>.yaml e os filtros em configs/preprocessing.yaml, e commite.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LR_GRID = {"unet": [3e-4, 1e-3, 3e-3], "segformer": [1e-4, 2e-4, 6e-4], "mask2former": [5e-5, 1e-4, 2e-4]}
FILTER_GRID = {
    "P6": [{"sigma": 0.5}, {"sigma": 1.0}, {"sigma": 2.0}],
    "P7": [{"size": 3}, {"size": 5}],
    "P8": [{"sigma_color": 0.25}, {"sigma_color": 0.5}, {"sigma_color": 1.0}],
    "P9": [{"n_iter": 5}, {"n_iter": 10}, {"n_iter": 20}],
    "P10": [{"thr_scale": 0.5}, {"thr_scale": 1.0}, {"thr_scale": 2.0}],
}


def slug(text):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", text)


def jobs(mode, archs, pres):
    out = []
    if mode == "lr":
        for arch in archs:
            for lr in LR_GRID[arch]:
                out.append({"arch": arch, "pre": "P1", "tag": slug(f"lr_{arch}_{lr:g}"), "lr": lr, "pre_params": None})
    else:
        for arch in archs:
            for pre in pres:
                for params in FILTER_GRID[pre]:
                    k, v = next(iter(params.items()))
                    out.append({"arch": arch, "pre": pre, "tag": slug(f"filt_{arch}_{pre}_{k}{v:g}"), "lr": None,
                                "pre_params": {pre: params}})
    return out


def command(job, args):
    cmd = [sys.executable, "run_experiment.py", "--arch", job["arch"], "--pre", job["pre"], "--seeds", "0",
           "--epochs", str(args.epochs), "--tag", job["tag"]]
    if job["lr"] is not None:
        cmd += ["--lr", str(job["lr"])]
    if job["pre_params"]:
        cmd += ["--pre-params", json.dumps(job["pre_params"])]
    for flag, val in (("--stats", args.stats), ("--processed", args.processed), ("--gt-root", args.gt_root),
                      ("--configs", args.configs), ("--results", args.results), ("--ckpt-root", args.ckpt_root),
                      ("--pred-root", args.pred_root)):
        if val is not None:
            cmd += [flag, str(val)]
    return cmd


def summary(results):
    rows = []
    for f in sorted(Path(results, "_tune").glob("*/runs/*_seed*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        v = r["val_at_best"]
        pp = {k: x for k, x in r["pre_params"].items() if k not in ("id", "name", "stats_split")}
        rows.append((r["arch"], r["pre_id"], r["tag"], r["train_config"]["lr"], pp,
                     r["train_config"]["epochs"], r["best_epoch"], v["dice"], v["iou"]))
    if not rows:
        print("Nenhuma execução de ajuste encontrada em", Path(results, "_tune"))
        return
    rows.sort(key=lambda x: (x[0], x[1], -x[7]))
    print(f"{'arq':12s} {'pré':4s} {'lr':>8s} {'épocas':>6s} {'melhor':>6s} {'Dice val':>9s} {'IoU val':>8s}  parâmetros do filtro / tag")
    prev = None
    for arch, pre, tag, lr, pp, ep, be, dice, iou in rows:
        if prev and prev != (arch, pre):
            print()
        prev = (arch, pre)
        print(f"{arch:12s} {pre:4s} {lr:8.0e} {ep:6d} {be:6d} {dice:9.4f} {iou:8.4f}  {pp if pp else ''} [{tag}]")
    print("\n(ordenado por Dice de validação dentro de cada arquitetura/pré-processamento; a primeira linha de cada bloco é a melhor)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lr", "filters", "summary"])
    ap.add_argument("--arch", nargs="+", default=["unet"], choices=list(LR_GRID))
    ap.add_argument("--pre", nargs="+", default=list(FILTER_GRID), help="filtros a ajustar (modo filters)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--gt-root", default=None)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--results", default=None, help="raiz dos resultados (padrão: results)")
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--pred-root", default=None)
    a = ap.parse_args()

    if a.mode == "summary":
        summary(a.results or "results")
    else:
        todo = jobs(a.mode, a.arch, a.pre)
        print(f"{len(todo)} configurações ({a.mode}), {a.epochs} épocas cada\n")
        for i, job in enumerate(todo, 1):
            cmd = command(job, a)
            print(f"[{i}/{len(todo)}] {job['tag']}")
            if a.dry_run:
                print("   ", " ".join(cmd))
                continue
            subprocess.run(cmd, check=False)
        if not a.dry_run:
            print("\n=== Resumo ===")
            summary(a.results or "results")
