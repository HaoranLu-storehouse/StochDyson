# Cr(CO)5 LVCM bundle for the 2D Poisson solver

This bundle contains a 2D three-state diabatic linear vibronic coupling model for Cr(CO)5 based on:

- Baihua Wu, Xin He, and Jian Liu, *J. Phys. Chem. Lett.* **2024**, 15, 644-658, Supporting Information Section S1-F.
- The underlying molecular model cited there: Worth, Welch, and Paterson, *Mol. Phys.* **2006**, 104, 1095-1105.

## Model form

The published diabatic LVCM Hamiltonian is

H = sum_k (omega_k / 2) (P_k^2 + R_k^2)
  + sum_n (E_n + sum_k kappa_k^(n) R_k) |n><n|
  + sum_{n!=m} sum_k lambda_k^(nm) R_k |n><m|

This bundle maps the two normal modes to the solver axes as:

- x := mode 2, the tuning / relaxation coordinate
- y := mode 1, the branching / coupling coordinate

The PES NPZ is stored in Hartree so it can be used directly with the solver's Schrödinger propagator.
Coordinates remain the published dimensionless normal-mode coordinates.

## Files

- `examples/generate_crco5_lvcm.py` — builds the NPZ PES and a CSV preview of cuts
- `examples/crco5_lvcm_2d.npz` — solver-ready 2D PES
- `examples/crco5_lvcm_preview_cuts.csv` — human-readable PES cuts
- `configs/crco5_lvcm_relaxation.yaml` — production-style Poisson input for the 1000 fs relaxation window

## Initial packet

The paper uses the second diabatic state as the initial electronic state.
The initial nuclear Wigner distribution is centered at:

- x0 = 14.3514
- y0 = 0.0

The Gaussian widths in the YAML are chosen to reproduce the published Wigner widths of eq. (S28):

- sigma_x = sqrt(2 / alpha_2)
- sigma_y = sqrt(2 / alpha_1)

with `alpha_2 = 0.4586` and `alpha_1 = 0.4501`.

## Usage

Place the bundle inside `poisson_v1_multistate/` or adjust the relative paths in the YAML, then run:

```bash
python examples/generate_crco5_lvcm.py
python run_poisson_v1.py configs/crco5_lvcm_relaxation.yaml
```
