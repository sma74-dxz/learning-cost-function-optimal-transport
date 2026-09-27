# Version Audit

This repository was consolidated from the uploaded historical research files.

## Core discrete algorithm

| Original file | Role | Decision |
|---|---|---|
| `iot_log.py` | Early compact Sinkhorn + inverse Sinkhorn implementation | Consolidated into `src/inverse_ot/discrete.py` |
| `iot_log(1).py` | Near-duplicate of `iot_log.py` | Removed as duplicate |
| `discrete_synthetic.ipynb` | Synthetic discrete IOT experiment | Main source for clean synthetic example |
| `discrete_synthetic-figure4.ipynb` | Figure-specific synthetic experiment | Historical / plotting branch |
| `discrete_synthetic-last_two_figures.ipynb` | Figure-specific experiment | Historical / plotting branch |
| `continuous_synthetic(1).ipynb` | Despite filename, contains the same matrix-scaling synthetic lineage | Historical / duplicate branch |

## Continuous neural inverse OT

| Original file | Role | Decision |
|---|---|---|
| `net_struct.py` | TensorFlow definitions of dual and cost networks | Architecture retained; TF duplication removed |
| `cts_iot_train.py` | TensorFlow training implementation of the continuous IOT loss | Objective consolidated into PyTorch `continuous.py` |
| `cts_iot_eval.py` | TensorFlow checkpoint evaluation / recovered-cost plotting | Replaced by reusable evaluation utilities |
| `continuous_synthetic.ipynb` | Newer PyTorch continuous synthetic experiments | **Primary implementation source** |
| `iot_colortransfer_generate.ipynb` | PyTorch color-transfer application | Kept as application template; image files external |

## Marriage / structured real-data branch

| Original file | Role | Decision |
|---|---|---|
| `marriage_dataset-revise_for_iot.ipynb` | Marriage-data preprocessing / clustering | Dataset-specific preprocessing; not included in core |
| `iot_discrete_marriage.ipynb` | Discrete IOT + marriage affinity experiment | Paper application; represented by clean template |
| `iot_log_marriage.py` | Compact marriage-specific inverse-OT variant | Superseded by general discrete implementation |

## Migration / later exploratory branch

| Original file | Role | Decision |
|---|---|---|
| `iot_discrete_migration.ipynb` | Migration matching experiment | Exploratory, not part of the core paper presentation |
| `iot_discrete_migration(1).ipynb` | Near-duplicate migration notebook | Removed as duplicate |
| `iot_log_migration.py` | Compact migration-specific variant | Superseded by general implementation |
| `iot_discrete_migration_learn_weights.ipynb` | Larger weight-learning extension | Kept out of paper-aligned core |

## External / comparison implementations

| Original file | Role | Decision |
|---|---|---|
| `inverse_OT_implementation.ipynb` | Alternative / baseline inverse-OT implementation | Historical comparison only |
| `invot_stuart_implementation.ipynb` | Stuart-Wolfram comparison implementation | Historical comparison only |

## Consolidation Principles

1. Keep one implementation of each algorithm.
2. Preserve the paper's separation between discrete matrix scaling and continuous neural IOT.
3. Prefer the newer PyTorch continuous implementation over duplicate TensorFlow 1.x code.
4. Remove hard-coded local paths and notebook-only state.
5. Do not redistribute external marriage or image datasets.
6. Keep migration and later weight-learning work conceptually separate from the paper-aligned core.
