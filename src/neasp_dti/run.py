import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.nn import functional as F

from .data import DATASETS, OpaqueBundle, load_dataset, make_smoke_inputs, sha256_file
from .model import ModelConfig, NEASPDTI, multi_positive_infonce


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics(labels, scores):
    labels = labels.cpu().numpy()
    scores = scores.detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite prediction probabilities")
    return {"acc": float(accuracy_score(labels, scores >= 0.5)),
            "auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores)),
            "f1": float(f1_score(labels, scores >= 0.5, zero_division=0))}


@torch.no_grad()
def evaluate(model, drug_features, target_features, edge_index, split):
    model.eval()
    z = model.encode(drug_features, target_features, edge_index)
    scores = torch.cat([model.score(z, pairs).softmax(1)[:, 1]
                        for pairs in split.pairs.to(z.device).split(8192)])
    return metrics(split.labels, scores), scores.cpu()


def save_prepared(directory, data):
    torch.save({"drug_features": data.drug_features, "target_features": data.target_features,
                "edge_index": data.edge_index, "drug_ids": data.drug_ids,
                "target_ids": data.target_ids,
                "splits": {key: {"pairs": value.pairs, "labels": value.labels}
                           for key, value in data.splits.items()}}, directory / "prepared.pt")
    write_json(directory / "data.json", {"identity": data.identity,
                                          "prepared_sha256": sha256_file(directory / "prepared.pt")})


def run_trial(data, seed, directory, args, config):
    seed_all(seed)
    device = torch.device(args.device)
    model = NEASPDTI(len(data.drug_ids), len(data.target_ids), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    drugs, targets = data.drug_features.to(device), data.target_features.to(device)
    edges = data.edge_index.to(device)
    positives = data.positive_pairs.to(device)
    pairs = data.splits["train"].pairs.to(device)
    labels = data.splits["train"].labels.to(device)
    best_auroc, best_epoch, stale = -math.inf, 0, 0
    best_state, best_validation = None, None
    directory.mkdir(parents=True, exist_ok=args.resume)
    with (directory / "history.csv").open("w", newline="") as handle:
        fields = ["epoch", "loss", "classification_loss", "infonce_loss",
                  "validation_acc", "validation_auroc", "validation_aupr", "validation_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(drugs, targets, edges)
            ce = F.cross_entropy(model.score(z, pairs), labels)
            nce = multi_positive_infonce(z[:len(data.drug_ids)], z[len(data.drug_ids):],
                                         positives, args.temperature)
            loss = ce + args.contrastive_weight * nce
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at {data.name}/seed-{seed}/epoch-{epoch}")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError(f"invalid gradient at {data.name}/seed-{seed}/epoch-{epoch}")
            optimizer.step()
            validation, _ = evaluate(model, drugs, targets, edges, data.splits["validation"])
            writer.writerow({"epoch": epoch, "loss": loss.item(), "classification_loss": ce.item(),
                             "infonce_loss": nce.item(),
                             **{f"validation_{key}": value for key, value in validation.items()}})
            handle.flush()
            if validation["auroc"] > best_auroc:
                best_auroc, best_epoch, best_validation, stale = validation["auroc"], epoch, validation, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"{data.name} seed={seed} epoch={epoch} loss={loss.item():.4f} "
                      f"val_AUROC={validation['auroc']:.4f}", flush=True)
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint = directory / "best.pt"
    torch.save({"model_config": asdict(config), "model_state": best_state,
                "num_drugs": len(data.drug_ids), "num_targets": len(data.target_ids),
                "seed": seed, "best_epoch": best_epoch,
                "prepared_sha256": args.prepared_sha256}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    test, scores = evaluate(model, drugs, targets, edges, data.splits["test"])
    predictions = directory / "test-predictions.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("drugid", "targetid", "label", "probability"))
        writer.writerows((data.drug_ids[d], data.target_ids[t], label, probability)
                         for (d, t), label, probability in zip(data.splits["test"].pairs.tolist(),
                                                               data.splits["test"].labels.tolist(),
                                                               scores.tolist()))
    result = {
        "dataset": data.name, "seed": seed, "synthetic": data.identity["synthetic"],
        "epochs_run": epoch, "best_epoch": best_epoch, "validation": best_validation,
        "test": test, "checkpoint_sha256": sha256_file(checkpoint),
        "predictions_sha256": sha256_file(predictions)}
    write_json(directory / "metrics.json", result)
    print(f"{data.name} seed={seed} best_epoch={best_epoch} test={test}", flush=True)
    return result


def summarize(output, results):
    fields = ["dataset", "synthetic", "n_seeds"] + [f"{metric}_{stat}"
             for metric in ("acc", "auroc", "aupr", "f1") for stat in ("mean", "std")]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in dict.fromkeys(result["dataset"] for result in results):
            runs = [result for result in results if result["dataset"] == dataset]
            row = {"dataset": dataset, "synthetic": runs[0]["synthetic"], "n_seeds": len(runs)}
            for metric in ("acc", "auroc", "aupr", "f1"):
                values = [result["test"][metric] for result in runs]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
            writer.writerow(row)


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if min(args.threads, args.epochs, args.patience) < 1:
        raise ValueError("threads, epochs and patience must be positive")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError("seeds must be in [0, 2**32)")
    if any(not math.isfinite(value) or value <= 0
           for value in (args.lr, args.temperature, args.contrastive_weight)):
        raise ValueError("lr, temperature and contrastive_weight must be positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.sparse.check_sparse_tensor_invariants.enable()
    names = ["Davis" if args.dataset == "davis" else args.dataset]
    if names[0] not in DATASETS:
        raise ValueError(f"unknown dataset: {args.dataset}")
    bundle = OpaqueBundle(args.opaque_embeddings)
    if args.command == "run" and bundle.metadata.get("synthetic"):
        raise ValueError("synthetic embeddings are only accepted by smoke")
    if any(name not in bundle.datasets for name in names):
        raise ValueError("requested dataset missing from embedding bundle")
    datasets = [load_dataset(args.data_root, name, bundle, args.split_seed) for name in names]
    config = ModelConfig()
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    identity = {"model": asdict(config), "datasets": [data.identity for data in datasets],
                "training": {key: getattr(args, key) for key in
                             ("seeds", "epochs", "patience", "lr", "weight_decay",
                              "temperature", "contrastive_weight", "device", "threads")},
                "source_sha256": source_hash.hexdigest(), "torch": str(torch.__version__),
                "numpy": np.__version__, "sklearn": sklearn.__version__,
                "python": sys.version.split()[0], "cuda": torch.version.cuda}
    output = Path(args.output)
    if output.exists():
        if not args.resume or not (output / "experiment.json").is_file():
            raise FileExistsError(f"{output} already exists")
        if json.loads((output / "experiment.json").read_text()) != identity:
            raise ValueError("resume rejected: inputs or settings changed")
    else:
        output.mkdir(parents=True)
        write_json(output / "experiment.json", identity)
    results = []
    for data in datasets:
        dataset_dir = output / data.name
        dataset_dir.mkdir(exist_ok=True)
        prepared = dataset_dir / "prepared.pt"
        if not prepared.exists():
            save_prepared(dataset_dir, data)
        args.prepared_sha256 = sha256_file(prepared)
        record = json.loads((dataset_dir / "data.json").read_text())
        if record != {"identity": data.identity, "prepared_sha256": args.prepared_sha256}:
            raise ValueError(f"prepared input changed: {dataset_dir}")
        for seed in args.seeds:
            directory = dataset_dir / f"seed-{seed}"
            completed = directory / "metrics.json"
            if args.resume and completed.exists():
                result = json.loads(completed.read_text())
                if result["checkpoint_sha256"] != sha256_file(directory / "best.pt"):
                    raise ValueError(f"checkpoint changed: {directory}")
                if result["predictions_sha256"] != sha256_file(directory / "test-predictions.csv"):
                    raise ValueError(f"predictions changed: {directory}")
                results.append(result)
            else:
                results.append(run_trial(data, seed, directory, args, config))
            summarize(output, results)
    print(f"Completed {len(results)} trials. Summary: {output / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "smoke"):
        parser_run = subparsers.add_parser(command)
        parser_run.add_argument("--output", type=Path, default=Path("results") / command)
        parser_run.add_argument("--device", default="cpu" if command == "smoke" else "cuda:0")
        parser_run.add_argument("--seed", "--seeds", dest="seeds", type=int, nargs="+", required=True,
                                help="one or more training seeds")
        parser_run.add_argument("--epochs", type=int, default=3 if command == "smoke" else 100)
        parser_run.add_argument("--patience", type=int, default=20)
        parser_run.add_argument("--threads", type=int, default=2 if command == "smoke" else 4)
        parser_run.add_argument("--split-seed", type=int, default=114514)
        parser_run.add_argument("--lr", type=float, default=0.005)
        parser_run.add_argument("--weight-decay", type=float, default=0.01)
        parser_run.add_argument("--temperature", type=float, default=0.2)
        parser_run.add_argument("--contrastive-weight", type=float, default=0.1)
        parser_run.add_argument("--dataset", required=True)
        parser_run.add_argument("--opaque-embeddings", type=Path,
                                default=Path("pretrained/opaque-dti-embeddings.pt"))
        parser_run.add_argument("--resume", action="store_true")
        if command == "run":
            parser_run.add_argument("--data-root", type=Path, default=Path("data"))
        else:
            parser_run.add_argument("--inputs", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "smoke":
        if args.output.exists():
            parser.error(f"{args.output} already exists")
        inputs = args.inputs or args.output.with_name(args.output.name + "-inputs")
        args.dataset = "Davis" if args.dataset == "davis" else args.dataset
        args.data_root, args.opaque_embeddings = make_smoke_inputs(inputs, args.dataset)
        args.resume = False
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
