# dimensional-energy

Atomic spectra, field laws, and quantum systems across changing dimensions.

## Layout

- `src/qmbase/` — core solver library: Laguerre basis, LCR
  (logarithmic-coordinate representation) quadrature grids, Hamiltonian
  assembly, and diagonalization.
- `papers/2026b_he_2d_gauss/` — scripts for *Radial models of helium and atomic hydrogen in two and three dimensions*.
- `papers/2026_he_1d_halfline/` — scripts for *Bound States of One-Dimensional Helium on the Half-Line*.

## Papers and Preprints

This repository reproduces the calculations in the following papers and preprints

[//]: # (> D. A. Konovalov, *Bound States of One-Dimensional Helium on the Half-Line*,)

[//]: # (> submitted to The European Physical Journal Plus &#40;2026&#41;.)

[//]: # (<!-- Add the DOI here on acceptance. -->)


**Konovalov, D. Radial Models of Helium and Atomic Hydrogen in Two and Three Dimensions.** 
Few-Body Syst 67, 59 (2026).
**DOI:** [10.1007/s00601-026-02080-5](https://doi.org/10.1007/s00601-026-02080-5)

**Konovalov, D.A. Bound states of one-dimensional helium on the half-line.** 
Eur. Phys. J. Plus 141, 967 (2026).
**DOI:** [10.1140/epjp/s13360-026-08186-3](https://doi.org/10.1140/epjp/s13360-026-08186-3)

**PREPRINT: Radial models of helium and atomic hydrogen in two and three dimensions**  
Dmitry A. Konovalov [![ResearchSquare](https://img.shields.io/badge/ResearchSquare-Preprint-blue)](https://www.researchsquare.com/article/rs-10237331/v1)
**DOI:** [10.21203/rs.3.rs-10237331/v1](https://doi.org/10.21203/rs.3.rs-10237331/v1)

**PREPRINT: Bound States of One-Dimensional Helium on the Half-Line**  
Dmitry A. Konovalov [![ResearchSquare](https://img.shields.io/badge/ResearchSquare-Preprint-blue)](https://www.researchsquare.com/article/rs-9602948/v1)
**DOI:** [10.21203/rs.3.rs-9602948/v1](https://doi.org/10.21203/rs.3.rs-9602948/v1)

---

## How to Cite


```bibtex
@article{konovalov2026radial,
  title        = {Radial Models of Helium and Atomic Hydrogen in Two and Three Dimensions.},
  author       = {Dmitry Konovalov},
  year         = {2026},
  journal      = {Few-Body Syst},
  volume       = {67},
  page         = {59},
  doi          = {https://doi.org/10.1007/s00601-026-02080-5},
  url          = {https://doi.org/10.1007/s00601-026-02080-5}
}
```

```bibtex
@article{konovalov2026He1Dp,
  title        = {Bound States of One-Dimensional Helium on the Half-Line},
  author       = {Dmitry A. Konovalov},
  year         = {2026},
  journal      = {Eur. Phys. J. Plus},
  volume       = {141},
  page         = {967},
  doi          = {https://doi.org/10.1140/epjp/s13360-026-08186-3},
  url          = {https://doi.org/10.1140/epjp/s13360-026-08186-3}
}
```

```bibtex
@article{konovalov2026radial-preprint,
  title={Radial models of helium and atomic hydrogen in two and three dimensions},
  author={Konovalov, Dmitry},
  year={2026},
  publisher    = {Research Square},
  doi          = {https://doi.org/10.21203/rs.3.rs-10237331/v1},
  url          = {https://doi.org/10.21203/rs.3.rs-10237331/v1}
}
```


```bibtex
@article{konovalov2025He1Dp-preprint,
  title        = {PREPRINT: Bound States of One-Dimensional Helium on the Half-Line},
  author       = {Dmitry A. Konovalov},
  year         = {2025},
  publisher    = {Research Square},
  doi          = {https://doi.org/10.21203/rs.3.rs-9602948/v1},
  url          = {https://doi.org/10.21203/rs.3.rs-9602948/v1}
}
```

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

### From the terminal, example:

```bash
cd papers/2026b_he_2d_gauss
python runme01_Table1.py
```

### In PyCharm

Open the repository as a project, then right-click `src/qmbase` and choose
**Mark Directory as → Sources Root**. Run any script under `papers/`.

## Scripts (`papers/2026b_he_2d_gauss`)

| Script              | Purpose                                                                |
|---------------------|------------------------------------------------------------------------|
| `runme01_Table1.py` | Table-1: Validation of the intrinsic 2D logarithmic-potential spectrum |
| `runme02_Table2.py` | Table-2: Lowest one-electron radial energies comparing the ordinary 3D |
| `runme0X_TableN.py` | Table-N: etc                                                           |


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




## License

See [`LICENSE`](LICENSE).
