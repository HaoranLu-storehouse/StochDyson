
from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml

from run_activeL_poisson_diagnostics import PoissonDiagnosticSimulator, _norm
from newns_anderson_linear import HybridSettings, model_from_config, FS_TO_AU, group_observables


def run_timing_case(cfg: dict, nmetal: int, L: int, tmax_fs: float) -> Dict[str, float]:
    cfg = json.loads(json.dumps(cfg))
    cfg['model']['nmetal'] = int(nmetal)
    cfg['hybrid']['active_L'] = int(L)
    cfg['hybrid']['tmax_fs'] = float(tmax_fs)
    model = model_from_config(cfg)
    settings = HybridSettings(**cfg['hybrid'])
    diag = cfg.get('poisson_diagnostics', {})
    sim = PoissonDiagnosticSimulator(
        model, settings,
        lambda_scale=float(diag.get('lambda_scale', 8.0)),
        mu_min=float(diag.get('mu_min', 0.05)),
        mu_max=float(diag.get('mu_max', 0.80)),
    )
    dt = settings.block_fs * FS_TO_AU
    nblocks = int(round(settings.tmax_fs / settings.block_fs))
    groups = np.repeat(model.psi0[None, :, :], settings.groups, axis=0)

    stochastic_elapsed = 0.0
    lambda_vals=[]; ess_vals=[]; strength_vals=[]
    t0_total = time.perf_counter()
    for _ in range(nblocks):
        psi_mean = np.mean(groups, axis=0)
        rate = sim.set_block_rate(psi_mean, dt)
        t0 = time.perf_counter()
        for g in range(settings.groups):
            groups[g] = sim.propagate_group_block(groups[g], dt, sim.rngs[g])
        stochastic_elapsed += time.perf_counter() - t0
        lambda_vals.append(rate['lambda_au_inv'])
        strength_vals.append(rate['residual_strength_au'])
        ess_vals.append(sim.block_ess())
    wall_elapsed = time.perf_counter() - t0_total
    obs = group_observables(groups, model)
    return {
        'N': int(nmetal),
        'L': int(L),
        'residual_states': int(sim.residual.size),
        'tmax_fs': float(tmax_fs),
        'block_fs': float(settings.block_fs),
        'groups': int(settings.groups),
        'replicas_per_group': int(settings.replicas_per_group),
        'first_order_samples_per_replica': int(sim.first_order_sample_count()),
        'high_order_paths_per_block': int(settings.groups * settings.replicas_per_group * settings.high_order_samples),
        'stochastic_cpu_s': float(stochastic_elapsed),
        'wall_s': float(wall_elapsed),
        'time_per_block_s': float(stochastic_elapsed / max(nblocks,1)),
        'mean_lambda_au_inv': float(np.mean(lambda_vals) if lambda_vals else 0.0),
        'mean_ess': float(np.mean(ess_vals) if ess_vals else 1.0),
        'mean_residual_strength_au': float(np.mean(strength_vals) if strength_vals else 0.0),
        'max_abs_norm_error': float(abs(obs['norm_u'] - 1.0)),
        'final_population_d': float(obs['population_d_u']),
    }


def fit_global_exponent(rows: List[Dict[str, float]]) -> Dict[str, float]:
    # Fit ln t = a_L + x ln N with a separate intercept for each L and a global slope x.
    L_values = sorted({int(r['L']) for r in rows})
    idx = {L:i for i,L in enumerate(L_values)}
    m = len(rows); p = len(L_values) + 1
    A = np.zeros((m,p), dtype=float)
    y = np.zeros(m, dtype=float)
    for i,r in enumerate(rows):
        A[i, idx[int(r['L'])]] = 1.0
        A[i, -1] = math.log(float(r['N']))
        y[i] = math.log(float(r['stochastic_cpu_s']))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[-1])
    intercepts = {L: float(math.exp(coef[idx[L]])) for L in L_values}
    return {'x': slope, 'prefactors': intercepts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--L', type=int, nargs='+', default=[0,16,32,48,64])
    ap.add_argument('--N', type=int, nargs='+', default=[200,400,600,800,1000,1200,1400,1600,1800,2000])
    ap.add_argument('--tmax-fs', type=float, default=20.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    rows=[]
    for L in args.L:
        for N in args.N:
            print(f'[RUN scaling] L={L} N={N}', flush=True)
            row = run_timing_case(cfg, N, L, args.tmax_fs)
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / 'cpu_scaling_L_vs_N.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fit = fit_global_exponent(rows)
    fit_path = args.outdir / 'cpu_scaling_fit.json'
    fit_path.write_text(json.dumps(fit, indent=2), encoding='utf-8')
    print(json.dumps(fit, indent=2), flush=True)

if __name__ == '__main__':
    main()
