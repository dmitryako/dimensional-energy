# dimensional-energy

Atomic spectra, field laws, and quantum systems across changing dimensions.

## Layout

- `src/qmbase/` — core solver library: Laguerre basis, LCR
  (logarithmic-coordinate representation) quadrature grids, Hamiltonian
  assembly, and diagonalization.
- `papers/2026_he_1d_halfline/` — self-contained scripts that reproduce the
  energies, tables, and figures of the paper below.

## Paper

This repository reproduces the calculations in:

> D. A. Konovalov, *Bound States of One-Dimensional Helium on the Half-Line*,
> submitted to The European Physical Journal Plus (2026).
<!-- Add the DOI here on acceptance. -->

## Requirements

Python 3.10+ and the packages listed in [`requirements.txt`](requirements.txt):
`matplotlib`, `numpy`, `pandas`, `scipy`, `torch`, `tqdm`.

Set up an isolated environment with either conda or `venv`, then install the
dependencies. Both paths end with the same `pip install` step.

### Option A — conda / Miniconda

```bash
conda create -n dimensional-energy python=3.11
conda activate dimensional-energy
pip install -r requirements.txt
```

### Option B — venv

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Each `runmeXX` script adds `src/qmbase` to `sys.path` automatically, so no
`PYTHONPATH` setup is needed — clone, install, and run.

### From the terminal

```bash
cd papers/2026_he_1d_halfline
python runme01_makeFigs_v260710c_He1d_LgrrLcrAnti_Test_OK.py
```

### In PyCharm

Open the repository as a project, then right-click `src/qmbase` and choose
**Mark Directory as → Sources Root**. Run any script under `papers/`.

## Scripts (`papers/2026_he_1d_halfline`)

| Script | Purpose |
| --- | --- |
| `runme01_..._LgrrLcrAnti_Test_OK.py` | Antisymmetric (triplet) wave-function figures, with self-test asserts |
| `runme02_..._LgrrLcrSym_Test_OK.py`  | Symmetric (singlet) wave-function figures, with self-test asserts |
| `runme03_..._antiFinalTable.py`      | Antisymmetric final-table energies, He (Z = 2) |
| `runme04_..._symmFinalTable.py`      | Symmetric final-table energies, He (Z = 2) |
| `runme05_..._Z3_FinalTable.py`       | Li⁺ (Z = 3) |
| `runme06_..._Z1p1_FinalTable.py`     | Hypothetical ion, Z = 1.1 |
| `runme07_..._Z1_FinalTable.py`       | H⁻ (Z = 1); confirms no bound outer electron |

All numerical results are checked at runtime by asserts that catch deviations
to ~1e-13.


## Preprint

This work is available as a preprint:

**Dimensional Energy: A New Framework for Quantum Many-Body Systems**  
Dmitry A. Konovalov  

[![ResearchSquare](https://img.shields.io/badge/ResearchSquare-Preprint-blue)](https://www.researchsquare.com/article/rs-9602948/v1)

**DOI:** [10.21203/rs.3.rs-9602948/v1](https://doi.org/10.21203/rs.3.rs-9602948/v1)

---

## How to Cite

```bibtex
@misc{konovalov2025dimensional,
  title        = {Dimensional Energy: A New Framework for Quantum Many-Body Systems},
  author       = {Dmitry A. Konovalov},
  year         = {2025},
  publisher    = {Research Square},
  doi          = {10.21203/rs.3.rs-9602948/v1},
  url          = {https://doi.org/10.21203/rs.3.rs-9602948/v1}
}


## License

See [`LICENSE`](LICENSE).
