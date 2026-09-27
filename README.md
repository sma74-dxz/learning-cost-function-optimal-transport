# Learning Cost Function for Optimal Transport

Clean research-code repository accompanying:

**Shaojun Ma, Haodong Sun, Xiaojing Ye, Hongyuan Zha, Haomin Zhou**  
*Learning Cost Function for Optimal Transport*  
arXiv:2002.09650

Paper: https://arxiv.org/abs/2002.09650

## Overview

This repository consolidates multiple historical research notebooks and scripts
for **inverse optimal transport (IOT)**: recovering the transport cost from an
observed optimal transport plan or samples from that plan.

The paper develops two complementary algorithms:

1. **Discrete inverse OT**  
   A matrix-scaling algorithm derived from entropy-regularized OT.

2. **Continuous inverse OT**  
   A mesh-free learning formulation in which the dual potentials and transport
   cost are parameterized by neural networks and optimized directly from samples.

A central advantage of the formulation is that cost learning does **not** require
solving a full forward OT problem inside every outer optimization iteration.

---

## Mathematical Setup

For entropy-regularized OT, the inverse problem observes a coupling `pi_hat`
with marginals `mu` and `nu` and seeks a cost `c`.

### Discrete inverse OT

The matrix-scaling iteration used in the paper is

```text
K = exp(-c / epsilon)
u = mu / (K v)
v = nu / (K^T u)
K = pi_hat / (u v^T)
c = projection_or_prox(-epsilon * log(K))
```

The default projection in the synthetic example enforces:

```text
c = c^T
diag(c) = 0
c >= 0
```

### Continuous inverse OT

Given paired observations `(x, y) ~ pi_hat`, the continuous formulation uses
neural networks for

```text
alpha_theta(x)
beta_theta(y)
c_eta(x, y)
```

and minimizes the empirical version of

```text
L(theta, eta)
    =
R(c_eta)
- E_mu[alpha_theta]
- E_nu[beta_theta]
+ E_pi_hat[c_eta]
+ epsilon * Integral exp(
      (alpha_theta(x) + beta_theta(y) - c_eta(x,y)) / epsilon
  ) d lambda(x,y)
```

The final integral is approximated using collocation samples.  The paper uses
uniform sampling in the reported experiments.

---

## Repository Structure

```text
learning_cost_optimal_transport/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── discrete_synthetic.json
│   └── continuous_synthetic.json
├── src/
│   └── inverse_ot/
│       ├── __init__.py
│       ├── discrete.py
│       ├── continuous.py
│       ├── networks.py
│       ├── sampling.py
│       └── metrics.py
├── experiments/
│   ├── discrete_synthetic.py
│   ├── continuous_synthetic.py
│   ├── marriage_template.py
│   └── color_transfer_template.py
├── docs/
│   └── VERSION_AUDIT.md
├── data/
│   └── README.md
└── figures/
```

---

## Discrete Synthetic Example

The discrete experiment recovers a structured cost matrix such as

```text
c_ij = |(i-j)/n|^p
```

from an entropy-regularized optimal coupling.

Run:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python experiments/discrete_synthetic.py
```

The script prints the relative Frobenius error of the recovered cost.

---

## Continuous Synthetic Example

The continuous experiment follows the paper's neural IOT formulation.  A
synthetic observed coupling is first sampled from a discretized forward OT
problem.  Neural networks then recover the continuous cost from paired samples.

Run:

```bash
export PYTHONPATH=src
python experiments/continuous_synthetic.py
```

The default script is intentionally smaller than the full research run so that
the repository provides a practical demonstration.  Increase the iteration and
sample counts through `configs/continuous_synthetic.json` for larger experiments.

The historical code used both TensorFlow 1.x and PyTorch implementations.  The
clean repository uses **PyTorch** for the continuous model to avoid maintaining
two duplicate neural implementations.

---

## Neural Architecture

For the one-dimensional continuous synthetic experiments, the paper uses a
5-layer cost network:

- input layer,
- 3 hidden layers,
- output layer,
- 20 neurons per hidden layer,
- `tanh` hidden activations,
- nonnegative output for the cost.

The clean implementation exposes the architecture through configuration rather
than duplicating it in every notebook.

---

## Real Marriage-Data Experiment

The paper also applies discrete inverse OT to marriage-market data using a
linear affinity model

```text
c = G^T A D
```

where `G` and `D` contain features for the two populations and `A` is the
interaction / affinity matrix.

The historical notebooks contain substantial preprocessing and local file-path
logic.  Those dataset files are not redistributed in this repository.

`experiments/marriage_template.py` contains the clean entry point expected after
the feature matrices and observed matching matrix have been prepared.

---

## Color-Transfer Experiment

The continuous IOT formulation is also used for RGB color transfer.  The cost is
represented through a neural function of RGB differences.

The original image files are not redistributed here.  A lightweight template is
provided in `experiments/color_transfer_template.py` documenting the expected
input format and learned-cost interface.

---

## Historical Code Consolidation

The uploaded research directory contained several generations of the same
algorithms:

- repeated discrete Sinkhorn/IOT implementations,
- multiple notebooks used only for individual paper figures,
- early baseline implementations,
- TensorFlow and PyTorch continuous variants,
- marriage-data preprocessing,
- migration experiments,
- color-transfer experiments,
- later experiments that learned additional weight structures.

These were **not** copied verbatim into the public repository.

Instead:

- the discrete matrix-scaling method is centralized in `discrete.py`;
- the continuous neural objective is centralized in `continuous.py`;
- neural architectures live in `networks.py`;
- forward-OT sampling utilities live in `sampling.py`;
- paper applications are represented by small experiment entry points;
- machine-specific paths, notebook outputs, duplicated plotting code, and
  external datasets are excluded.

See `docs/VERSION_AUDIT.md` for the detailed mapping.

---

## Reproducibility Notes

The clean repository is designed to make the algorithmic structure readable and
runnable.  It is not a claim that every historical notebook cell is required to
reproduce every figure exactly.

For exact historical figure reproduction, the archived research notebooks would
still be the appropriate reference.

---

## Citation

```bibtex
@article{ma2021learning,
  title={Learning Cost Function for Optimal Transport},
  author={Ma, Shaojun and Sun, Haodong and Ye, Xiaojing and Zha, Hongyuan and Zhou, Haomin},
  year={2021},
  note={arXiv:2002.09650}
}
```
