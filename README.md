# NEASP-DTI

## Model Architecture

NEASP-DTI addresses drug–target interaction prediction by considering that entities with similar local neighborhoods may still exhibit different interaction patterns. It projects pretrained molecular representations of drugs and protein sequence representations into a shared latent space. Propagation is performed only on the bipartite graph formed by positive training interactions: the Multimodal Entity Encoding module (MEE) unifies the attribute information of the two entity types; the Node-Adaptive Structure Modulation module (NASM) learns an entity-specific balance between self-information and neighborhood information; the Spectral-Response Calibration module (SRC) adjusts the spectral positions of the propagation operator through an identity term; and the Edge-Conditioned Multi-Order Propagation and Prediction module (ECMOPP) recalibrates existing edge weights according to the attributes of their endpoint entities and uses learnable multi-order Chebyshev propagation to integrate relational information at different ranges. After normalization and gated residual fusion, the model concatenates drug and target representations, estimates the interaction probability of each candidate pair with a classifier, and is trained jointly with binary cross-entropy and bidirectional multi-positive InfoNCE.

## Directory Structure

```text
data/
├── BindingDB.csv
├── Davis.csv
├── DrugBank.csv
└── KIBA.csv
pretrained/
└── opaque-dti-embeddings.pt
src/
└── neasp_dti/
    ├── __init__.py
    ├── data.py
    ├── model.py
    └── run.py
.python-version
pyproject.toml
uv.lock
```

## Command Example

```bash
uv run --extra cuda neasp run \
  --data-root data \
  --opaque-embeddings pretrained/opaque-dti-embeddings.pt \
  --dataset BindingDB \
  --seed 0 1 2 \
  --device cuda:0 \
  --output results/BindingDB-seeds-0-1-2
```

`--seed`/`--seeds` accepts one or more seeds. Each seed is run independently and
stored under `seed-<seed>/`; `summary.csv` reports the mean and sample standard
deviation across the requested seeds. For example, `--seeds 0 1 2` creates
three trials. Use `--resume` to reuse completed trials.

## Precomputed paper embeddings

The ESM-C and Uni-Mol encoder components used for the paper are provided here as
precomputed embeddings. NEASP-DTI consumes the fixed features directly from
`pretrained/opaque-dti-embeddings.pt`:

- **Drug features:** Uni-Mol v1 (all-H), 512 dimensions per drug.
- **Target features:** ESM-C 600M, layer-24 post-layer-normalization mean-pooled
  representations, 1152 dimensions per target.

The bundle contains embeddings for BindingDB, Davis, DrugBank, and KIBA. Its rows
follow the local zero-based entity IDs referenced by the corresponding CSV files
(`data/*.csv`), so the supplied embeddings can be used directly with the included
training commands. 
