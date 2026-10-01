#!/usr/bin/env python3
"""Generate a 2D 3-state LVCM PES for Cr(CO)5 after photodissociation.

Model source:
- Baihua Wu, Xin He, and Jian Liu, J. Phys. Chem. Lett. 2024, 15, 644-658
  Supporting Information, Section S1-F, eq. (S26), Table S4, eqs. (S28)-(S29).
- The underlying molecular model is the 2-mode/3-state LVCM of
  Worth, Welch, and Paterson, Mol. Phys. 2006, 104, 1095-1105.

Conventions used here for the Poisson solver:
- x := mode 2 (the tuning/relaxation coordinate; initial packet centered at x=14.3514)
- y := mode 1 (the branching/coupling coordinate; initial packet centered at y=0)
- energies in Hartree for direct use with the Schrödinger propagator (ħ = 1)
- coordinates are the dimensionless normal-mode coordinates of the published LVCM
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import numpy as np

EV_TO_HARTREE = 1.0 / 27.211386245988
FS_TO_AUT = 41.3413745758


@dataclass(frozen=True)
class GridSpec:
    xmin: float = -30.0
    xmax: float = 30.0
    nx: int = 800
    ymin: float = -10.0
    ymax: float = 10.0
    ny: int = 200


def build_model(grid: GridSpec) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    # Published parameters in eV (SI Table S4)
    E1_eV, E2_eV, E3_eV = 0.0424, 0.0424, 0.4344
    kappa2_state1_eV = -0.0328
    kappa2_state2_eV = +0.0328
    lambda1_12_eV = +0.0328
    lambda1_23_eV = -0.0978
    lambda2_13_eV = -0.0978
    omega1_eV, omega2_eV = 0.0129, 0.0129

    # Convert energies/frequencies to Hartree.
    E1 = E1_eV * EV_TO_HARTREE
    E2 = E2_eV * EV_TO_HARTREE
    E3 = E3_eV * EV_TO_HARTREE
    kappa2_state1 = kappa2_state1_eV * EV_TO_HARTREE
    kappa2_state2 = kappa2_state2_eV * EV_TO_HARTREE
    lambda1_12 = lambda1_12_eV * EV_TO_HARTREE
    lambda1_23 = lambda1_23_eV * EV_TO_HARTREE
    lambda2_13 = lambda2_13_eV * EV_TO_HARTREE
    omega1 = omega1_eV * EV_TO_HARTREE
    omega2 = omega2_eV * EV_TO_HARTREE

    # x := mode 2, y := mode 1
    x = np.linspace(grid.xmin, grid.xmax, grid.nx, endpoint=False)
    y = np.linspace(grid.ymin, grid.ymax, grid.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    common = 0.5 * omega2 * xx**2 + 0.5 * omega1 * yy**2

    # Diabatic PES matrix elements from eq. (S26) and Table S4.
    V11 = common + E1 + kappa2_state1 * xx
    V22 = common + E2 + kappa2_state2 * xx
    V33 = common + E3

    # Off-diagonal terms after mapping x<->mode2 and y<->mode1.
    V12 = lambda1_12 * yy
    V23 = lambda1_23 * yy
    V13 = lambda2_13 * xx

    mats = np.zeros(xx.shape + (3, 3), dtype=np.float64)
    mats[..., 0, 0] = V11
    mats[..., 1, 1] = V22
    mats[..., 2, 2] = V33
    mats[..., 0, 1] = mats[..., 1, 0] = V12
    mats[..., 1, 2] = mats[..., 2, 1] = V23
    mats[..., 0, 2] = mats[..., 2, 0] = V13
    evals = np.linalg.eigvalsh(mats)

    arrays = {
        'x': x,
        'y': y,
        'V11': V11,
        'V22': V22,
        'V33': V33,
        'V12': V12,
        'V23': V23,
        'V13': V13,
        'A1': evals[..., 0],
        'A2': evals[..., 1],
        'A3': evals[..., 2],
    }
    meta = {
        'omega1_hartree': omega1,
        'omega2_hartree': omega2,
        'mx': 1.0 / omega2,  # because x := mode 2 and kinetic = -(omega2/2) d^2/dx^2
        'my': 1.0 / omega1,  # because y := mode 1 and kinetic = -(omega1/2) d^2/dy^2
        'initial_x0': 14.3514,
        'initial_y0': 0.0,
        # From Wigner distribution eq. (S28): choose sigma^2 = 2/alpha for
        # psi ~ exp[-(q-q0)^2/(2 sigma^2)], which reproduces the published Wigner widths.
        'sigma_x': math.sqrt(2.0 / 0.4586),  # alpha_2 because x := mode 2
        'sigma_y': math.sqrt(2.0 / 0.4501),  # alpha_1 because y := mode 1
        'tmax_1000fs_au': 1000.0 * FS_TO_AUT,
        'tmax_300fs_au': 300.0 * FS_TO_AUT,
        'state1_min_x': -kappa2_state1 / omega2,  # V11 minimum along x
        'state2_min_x': -kappa2_state2 / omega2,  # V22 minimum along x
    }
    return x, y, arrays, meta


def write_preview_csv(path: str, x: np.ndarray, y: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    # Cuts chosen to expose the initial FC region, the JT region, and the lower valleys.
    y_cuts = [0.0]
    x_cuts = [14.3514, 2.5426356589, 0.0, -2.5426356589]

    def nearest_idx(vec: np.ndarray, value: float) -> int:
        return int(np.argmin(np.abs(vec - value)))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'cut_type', 'cut_value', 'scan_coord',
            'V11_hartree', 'V22_hartree', 'V33_hartree',
            'V12_hartree', 'V23_hartree', 'V13_hartree',
            'A1_hartree', 'A2_hartree', 'A3_hartree',
            'V11_eV', 'V22_eV', 'V33_eV',
            'V12_eV', 'V23_eV', 'V13_eV',
            'A1_eV', 'A2_eV', 'A3_eV',
        ])
        for y0 in y_cuts:
            j = nearest_idx(y, y0)
            for i, xv in enumerate(x):
                row_h = [
                    arrays['V11'][i, j], arrays['V22'][i, j], arrays['V33'][i, j],
                    arrays['V12'][i, j], arrays['V23'][i, j], arrays['V13'][i, j],
                    arrays['A1'][i, j], arrays['A2'][i, j], arrays['A3'][i, j],
                ]
                row_ev = [v / EV_TO_HARTREE for v in row_h]
                w.writerow(['y_fixed', y[j], xv, *row_h, *row_ev])
        for x0 in x_cuts:
            i = nearest_idx(x, x0)
            for j, yv in enumerate(y):
                row_h = [
                    arrays['V11'][i, j], arrays['V22'][i, j], arrays['V33'][i, j],
                    arrays['V12'][i, j], arrays['V23'][i, j], arrays['V13'][i, j],
                    arrays['A1'][i, j], arrays['A2'][i, j], arrays['A3'][i, j],
                ]
                row_ev = [v / EV_TO_HARTREE for v in row_h]
                w.writerow(['x_fixed', x[i], yv, *row_h, *row_ev])


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    grid = GridSpec()
    x, y, arrays, meta = build_model(grid)

    npz_path = os.path.join(base, 'crco5_lvcm_2d.npz')
    csv_path = os.path.join(base, 'crco5_lvcm_preview_cuts.csv')

    np.savez_compressed(npz_path, **arrays)
    write_preview_csv(csv_path, x, y, arrays)

    print(f'wrote {npz_path}')
    print(f'wrote {csv_path}')
    print('recommended grid and initial settings:')
    for k, v in meta.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
