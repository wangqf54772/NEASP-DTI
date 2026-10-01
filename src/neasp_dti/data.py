import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATASETS = ("BindingDB", "Davis", "DrugBank", "KIBA")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path, columns):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns are {columns}")
        result = list(reader)
    if not result:
        raise ValueError(f"{path}: empty CSV")
    return result


class OpaqueBundle:
    def __init__(self, path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        datasets = payload.get("datasets") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
            raise ValueError(f"{path}: invalid opaque bundle")
        if not isinstance(datasets, dict):
            raise ValueError(f"{path}: missing datasets")
        self.metadata = metadata
        self.sha256 = sha256_file(self.path)
        self.datasets = {}
        for name, values in datasets.items():
            if not isinstance(values, dict):
                raise ValueError(f"{path}: invalid dataset {name}")
            drug = values.get("drug_embeddings")
            target = values.get("target_embeddings")
            if (not isinstance(drug, torch.Tensor) or drug.ndim != 2 or drug.shape[1] != 512
                    or not drug.is_floating_point() or not torch.isfinite(drug).all()):
                raise ValueError(f"{path}: invalid drug embeddings for {name}")
            if (not isinstance(target, torch.Tensor) or target.ndim != 2 or target.shape[1] != 1152
                    or not target.is_floating_point() or not torch.isfinite(target).all()):
                raise ValueError(f"{path}: invalid target embeddings for {name}")
            drug, target = drug.float().contiguous(), target.float().contiguous()
            if not torch.isfinite(drug).all() or not torch.isfinite(target).all():
                raise ValueError(f"{path}: embeddings overflow float32")
            self.datasets[name] = (drug, target)


@dataclass
class PairSplit:
    pairs: torch.Tensor
    labels: torch.Tensor


@dataclass
class Dataset:
    name: str
    drug_ids: list[str]
    target_ids: list[str]
    drug_features: torch.Tensor
    target_features: torch.Tensor
    splits: dict[str, PairSplit]
    edge_index: torch.Tensor
    positive_pairs: torch.Tensor
    identity: dict


def load_dataset(root, name, bundle, split_seed=114514):
    path = Path(root) / f"{name}.csv"
    interaction_rows = rows(path, ("drugid", "targetid", "label"))
    drug_features, target_features = bundle.datasets[name]
    drug_ids = list(map(str, range(len(drug_features))))
    target_ids = list(map(str, range(len(target_features))))
    drug_index = {value: index for index, value in enumerate(drug_ids)}
    target_index = {value: index for index, value in enumerate(target_ids)}
    interactions = {}
    duplicates = 0
    for row in interaction_rows:
        if row["label"] not in ("0", "1"):
            raise ValueError(f"{path}: labels must be 0 or 1")
        if row["drugid"] not in drug_index or row["targetid"] not in target_index:
            raise ValueError(f"{path}: IDs must be zero-based embedding row indices")
        pair = (drug_index[row["drugid"]], target_index[row["targetid"]])
        label = int(row["label"])
        if pair in interactions:
            if interactions[pair] != label:
                raise ValueError(f"{path}: conflicting labels for {pair}")
            duplicates += 1
        interactions[pair] = label
    positives = sorted(pair for pair, label in interactions.items() if label == 1)
    negatives = sorted(pair for pair, label in interactions.items() if label == 0)
    if not positives or len(negatives) > len(positives):
        raise ValueError(f"{path}: need positives and no more supplied negatives than positives")
    num_drugs, num_targets = len(drug_ids), len(target_ids)
    pool_mask = np.ones(num_drugs * num_targets, dtype=bool)
    pool_mask[[drug * num_targets + target for drug, target in interactions]] = False
    pool = np.flatnonzero(pool_mask)
    needed = len(positives) - len(negatives)
    if len(pool) < needed:
        raise ValueError(f"{path}: insufficient unobserved pairs for 1:1 sampling")
    rng = np.random.RandomState(split_seed)
    sampled = pool[rng.choice(len(pool), size=needed, replace=False)]
    negatives.extend((int(value // num_targets), int(value % num_targets)) for value in sampled)
    pairs = np.asarray(positives + negatives, dtype=np.int64)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int64)
    order = rng.permutation(len(labels))
    pairs, labels = pairs[order], labels[order]
    train_end = int(len(labels) * 0.7)
    validation_end = train_end + int(len(labels) * 0.1)
    splits = {}
    for split_name, start, end in (("train", 0, train_end),
                                   ("validation", train_end, validation_end),
                                   ("test", validation_end, len(labels))):
        if set(labels[start:end]) != {0, 1}:
            raise ValueError(f"{path}: {split_name} lacks both classes")
        splits[split_name] = PairSplit(torch.from_numpy(pairs[start:end].copy()),
                                       torch.from_numpy(labels[start:end].copy()))
    train_positive = splits["train"].pairs[splits["train"].labels == 1]
    forward = train_positive.clone().T
    forward[1] += num_drugs
    edge_index = torch.cat((forward, forward.flip(0)), dim=1).contiguous()
    identity = {
        "name": name,
        "split_seed": split_seed,
        "split_ratio": [0.7, 0.1, 0.2],
        "num_drugs": num_drugs,
        "num_targets": num_targets,
        "positives": len(positives),
        "negatives": len(negatives),
        "sampled_negatives": needed,
        "deduplicated_rows": duplicates,
        "graph_directed_edges": edge_index.shape[1],
        "split_sizes": {key: len(value.labels) for key, value in splits.items()},
        "synthetic": bool(bundle.metadata.get("synthetic")),
        "interaction_csv_sha256": sha256_file(path),
        "embedding_bundle_sha256": bundle.sha256,
    }
    return Dataset(name, drug_ids, target_ids, drug_features, target_features,
                   splits, edge_index, train_positive, identity)


def make_smoke_inputs(destination, dataset_name):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.RandomState(2026)
    datasets = {}
    directory = root / "data"
    directory.mkdir()
    num_drugs, num_targets = 32, 24
    drug_latent = rng.randn(num_drugs, 8)
    target_latent = rng.randn(num_targets, 8)
    selected = np.argsort((drug_latent @ target_latent.T).ravel())[-128:]
    with (directory / f"{dataset_name}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("drugid", "targetid", "label"))
        writer.writerows((str(value // num_targets), str(value % num_targets), 1)
                         for value in sorted(selected))
    datasets[dataset_name] = {
        "drug_embeddings": torch.tensor(drug_latent @ rng.randn(8, 512) / np.sqrt(8), dtype=torch.float32),
        "target_embeddings": torch.tensor(target_latent @ rng.randn(8, 1152) / np.sqrt(8), dtype=torch.float32),
    }
    bundle = root / "opaque-embeddings.pt"
    torch.save({"metadata": {"format_version": 1, "kind": "opaque-per-dataset-fixed-embeddings",
                               "synthetic": True}, "datasets": datasets}, bundle)
    return root / "data", bundle
