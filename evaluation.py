"""
Etapa 13 - Avaliação e CSV de resultados (Experimentos 3 e 4: pós-processamento sobre as predições salvas).

Para cada execução (arquitetura, pré-processamento, seed) concluída pelo run_experiment.py:
  1. lê as probabilidades salvas (predictions/<arq>/<P>/seed<k>/<split>/) e o Ground Truth;
  2. escolhe os parâmetros de Q1-Q5 NA VALIDAÇÃO e os congela em results/postproc/<run_id>.json;
  3. avalia a condição sem pós-processamento ("none") e Q1..Q5 no split pedido;
  4. grava uma linha por (execução, split, pós-processamento) em results/results_<arq>.csv.

Experimento 3 = condição P1 (baseline) com cada Q; Experimento 4 = P2..P10 com cada Q. As linhas "none" de P1..P10
são os Experimentos 1 e 2.

ISOLAMENTO DO TESTE: --split test exige --final. Use só depois de congelar modelos, lr, filtros e pós-processamentos
(os parâmetros dos Q SEMPRE são escolhidos na validação, mesmo ao avaliar o teste).

Colunas (as do protocolo + extras): dataset, dataset_variant, trajectory_split, resolution_original, resolution_used,
sample_budget_gb, preprocessing_id, preprocessing_params, architecture, epochs, best_epoch, seed, postprocessing_id,
postprocessing_params, IoU, Dice, Precision, Recall, latency_ms, + IoU do fundo, mIoU, média por quadro com IC 95%
(bootstrap por trajetória), estabilidade temporal, memória, custo do pré-processamento, parâmetros, commit.
IoU/Dice/Precision/Recall são agregados sobre todos os pixels; os IC são da média por quadro.

Uso:
    python evaluation.py --arch unet segformer --pre all --seeds 0                 # validação (livre)
    python evaluation.py --arch unet segformer --pre all --seeds 0 --split test --final
    python evaluation.py --merge                                                    # junta results_<arq>.csv
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from metrics import MetricAccumulator, temporal_stability
from postprocessing import Q_IDS, apply_post, load_sequences, select_params

ARCHS = ("unet", "segformer", "mask2former")
PRE_IDS = [f"P{i}" for i in range(1, 11)]
COLUMNS = ["dataset", "dataset_variant", "trajectory_split", "resolution_original", "resolution_used",
           "sample_budget_gb", "preprocessing_id", "preprocessing_params", "architecture", "epochs", "best_epoch",
           "seed", "postprocessing_id", "postprocessing_params", "IoU", "Dice", "Precision", "Recall", "latency_ms",
           "IoU_background", "mIoU", "IoU_frame_mean", "IoU_ci_low", "IoU_ci_high", "Dice_frame_mean", "Dice_ci_low",
           "Dice_ci_high", "temporal_pred_consec_iou", "temporal_gt_consec_iou", "temporal_delta_iou",
           "temporal_delta_change", "peak_memory_mb", "preprocessing_ms", "n_params", "n_frames", "n_trajectories",
           "git_commit", "notes"]
TEMPORAL = ("pred_consec_iou", "gt_consec_iou", "delta_iou", "delta_change")


def evaluate_variant(seqs, qid, params, n_boot):
    acc = MetricAccumulator(keep_frames=True)
    tsum, wsum = {k: 0.0 for k in TEMPORAL}, 0
    for s in seqs:
        pred = apply_post(qid, s.prob, params)
        acc.update(pred, s.gt, groups=s.group)
        if len(pred) >= 2:
            ts = temporal_stability(pred, s.gt)
            w = len(pred) - 1
            for k in TEMPORAL:
                tsum[k] += ts[k] * w
            wsum += w
    res = acc.summary_with_ci(n_boot=n_boot)
    pooled = res["pooled"]
    temporal = {k: (tsum[k] / wsum if wsum else float("nan")) for k in TEMPORAL}
    return pooled, res, temporal, len({s.group for s in seqs})


def make_row(args, rec, arch, pre, seed, split, qid, params, pooled, res, temporal, n_traj, n_frames, budget):
    cost = (rec or {}).get("cost", {})
    return {
        "dataset": "active_matter", "dataset_variant": "The Well, 1 quadro a cada 3, 128 px, Ground Truth v2",
        "trajectory_split": split, "resolution_original": "256x256", "resolution_used": "128x128",
        "sample_budget_gb": budget, "preprocessing_id": pre,
        "preprocessing_params": json.dumps((rec or {}).get("pre_params", {}), sort_keys=True, default=str),
        "architecture": arch, "epochs": (rec or {}).get("train_config", {}).get("epochs", ""),
        "best_epoch": (rec or {}).get("best_epoch", ""), "seed": seed,
        "postprocessing_id": qid or "none", "postprocessing_params": json.dumps(params or {}, sort_keys=True),
        "IoU": pooled["iou"], "Dice": pooled["dice"], "Precision": pooled["precision"], "Recall": pooled["recall"],
        "latency_ms": cost.get("latency_ms_mean", ""), "IoU_background": pooled["iou_background"],
        "mIoU": pooled["miou"], "IoU_frame_mean": res["iou"]["frame_mean"], "IoU_ci_low": res["iou"]["ci_low"],
        "IoU_ci_high": res["iou"]["ci_high"], "Dice_frame_mean": res["dice"]["frame_mean"],
        "Dice_ci_low": res["dice"]["ci_low"], "Dice_ci_high": res["dice"]["ci_high"],
        "temporal_pred_consec_iou": temporal["pred_consec_iou"], "temporal_gt_consec_iou": temporal["gt_consec_iou"],
        "temporal_delta_iou": temporal["delta_iou"], "temporal_delta_change": temporal["delta_change"],
        "peak_memory_mb": cost.get("peak_memory_mb", ""),
        "preprocessing_ms": (rec or {}).get("preprocessing_ms_per_frame", ""), "n_params": (rec or {}).get("n_params", ""),
        "n_frames": n_frames, "n_trajectories": n_traj, "git_commit": (rec or {}).get("git_commit", ""),
        "notes": "" if rec else "registro da execução (results/runs) não encontrado"}


def write_csv(path, rows):
    """Substitui as linhas com a mesma chave (arquitetura, pré, seed, split, pós) e acrescenta as novas."""
    key = lambda r: (r["architecture"], r["preprocessing_id"], str(r["seed"]), r["trajectory_split"],
                     r["postprocessing_id"])
    existing = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    new_keys = {key(r) for r in rows}
    kept = [r for r in existing if key(r) not in new_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(kept + rows)


def merge(results):
    files = sorted(f for f in Path(results).glob("results_*.csv") if f.name != "results_all.csv")
    rows = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    with open(Path(results) / "results_all.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} linhas de {len(files)} arquivos -> {Path(results) / 'results_all.csv'}")


def run_eval(args, arch, pre, seed, budget):
    run_id = f"{arch}_{pre}_seed{seed}"
    pdir = args.pred_root / arch / pre / f"seed{seed}"
    if not (pdir / "valid").exists():
        print(f"[{run_id}] sem predições de validação em {pdir / 'valid'}, pulando")
        return []
    rec_path = args.results / "runs" / f"{run_id}.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else None

    val_seqs = load_sequences(pdir / "valid", args.gt_root / "valid")
    pp_path = args.results / "postproc" / f"{run_id}.json"
    if pp_path.exists() and not args.reselect:
        selection = json.loads(pp_path.read_text(encoding="utf-8"))
    else:
        selection = select_params(val_seqs)                      # SEMPRE na validação
        selection["created"] = datetime.now().isoformat(timespec="seconds")
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        pp_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    if args.split == "valid":
        seqs = val_seqs
    else:
        if not (pdir / "test").exists():
            print(f"[{run_id}] sem predições de teste em {pdir / 'test'}, pulando")
            return []
        seqs = load_sequences(pdir / "test", args.gt_root / "test")
    n_frames = int(sum(len(s.prob) for s in seqs))

    rows, line = [], []
    base_dice = None
    for qid in [None] + Q_IDS:
        params = selection["baseline"]["params"] if qid is None else selection[qid]["params"]
        pooled, res, temporal, n_traj = evaluate_variant(seqs, qid, params, args.n_boot)
        rows.append(make_row(args, rec, arch, pre, seed, args.split, qid, params, pooled, res, temporal,
                             n_traj, n_frames, budget))
        base_dice = pooled["dice"] if qid is None else base_dice
        line.append(f"{qid or 'none'} {pooled['dice']:.4f}" + ("" if qid is None else f" ({pooled['dice'] - base_dice:+.4f})"))
    print(f"[{run_id}] {args.split}: Dice  " + " | ".join(line))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", choices=list(ARCHS))
    ap.add_argument("--pre", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--final", action="store_true", help="confirma a avaliação no TESTE (use só ao final)")
    ap.add_argument("--reselect", action="store_true", help="reescolhe os parâmetros dos Q na validação")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--pred-root", type=Path, default=Path("predictions"))
    ap.add_argument("--gt-root", type=Path, default=Path("data/ground_truth"))
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.merge:
        merge(a.results)
    else:
        if not a.arch:
            ap.error("informe --arch (ou use --merge)")
        if a.split == "test" and not a.final:
            ap.error("o split de TESTE só deve ser avaliado ao final, com modelos e parâmetros congelados: "
                     "acrescente --final para confirmar")
        pres = PRE_IDS if a.pre == ["all"] else a.pre
        budget = ""
        if a.manifest.exists():
            budget = json.loads(a.manifest.read_text(encoding="utf-8")).get("budget", {}).get("subset_orig_equiv_gb", "")
        for arch in a.arch:
            rows = []
            for pre in pres:
                for seed in a.seeds:
                    r = run_eval(a, arch, pre, seed, budget)
                    if r:
                        write_csv(a.results / f"results_{arch}.csv", r)       # grava a cada execução (progresso salvo)
                        rows += r
            print(f"\n{arch}: {len(rows)} linhas gravadas em {a.results / f'results_{arch}.csv'}\n")
