#!/usr/bin/env python3
import os
import numpy as np


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, 'three_state_2d_demo.npz')

    x = np.linspace(-6.0, 6.0, 96, endpoint=False)
    y = np.linspace(-4.0, 4.0, 64, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Simple smooth 2D three-state diabatic model for smoke tests.
    V11 = 0.004 * (xx + 2.0) ** 2 + 0.010 * yy ** 2
    V22 = 0.004 * (xx - 0.5) ** 2 + 0.008 * (yy - 1.1) ** 2 + 0.03
    V33 = 0.003 * (xx - 2.8) ** 2 + 0.007 * (yy + 1.0) ** 2 + 0.08

    V12 = 0.010 * np.exp(-((xx + 0.9) ** 2) / 1.5 - (yy ** 2) / 0.8)
    V23 = 0.008 * np.exp(-((xx - 1.3) ** 2) / 1.6 - ((yy - 0.2) ** 2) / 0.9)

    np.savez_compressed(
        out,
        x=x,
        y=y,
        V11=V11,
        V22=V22,
        V33=V33,
        V12=V12,
        V23=V23,
    )
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
