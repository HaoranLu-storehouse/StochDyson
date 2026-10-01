from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from newns_anderson_linear import (
    AU_TO_FS,
    FS_TO_AU,
    HybridNewnsAndersonSimulator,
    HybridSettings,
    NewnsAndersonModel,
    model_from_config,
    group_observables,
    state_observables,
    _poisson_tail_ge2,
)


def _norm(psi: np.ndarray) -> float:
    return float(np.sqrt(np.vdot(psi.ravel(), psi.ravel()).real))


def phase_aligned_error(psi: np.ndarray, ref: np.ndarray) -> float:
    """Normalized phase-aligned L2 wavefunction distance."""
    n1, n2 = _norm(psi), _norm(ref)
    if n1 <= 1e-15 or n2 <= 1e-15:
        return float('nan')
    a = psi / n1
    b = ref / n2
    ov = np.vdot(b.ravel(), a.ravel())
    phase = ov / abs(ov) if abs(ov) > 1e-15 else 1.0 + 0.0j
    return _norm(a - phase * b)


class PoissonDiagnosticSimulator(HybridNewnsAndersonSimulator):
    """Strict Dyson-Poisson simulator with block-adaptive lambda and ESS diagnostics.

    lambda_b = clip(alpha_lambda * ||H_R psi_b||/||psi_b||, mu_min/dt, mu_max/dt)
    is fixed within a block.  Since lambda only changes the sampling law and the
    corresponding Poisson compensation is retained, the conditional n>=2
    estimator remains unbiased for every positive blockwise lambda.
    """

    def __init__(self, model: NewnsAndersonModel, settings: HybridSettings,
                 lambda_scale: float = 8.0, mu_min: float = 0.05, mu_max: float = 0.80):
        super().__init__(model, settings)
        self.lambda_scale = float(lambda_scale)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self._block_path_masses: List[float] = []
        self._block_orders: List[int] = []
        self._current_lambda = 0.0
        self._current_mu = 0.0

    def residual_transition_strength(self, psi: np.ndarray) -> float:
        """State-resolved RMS residual transition strength ||H_R psi||/||psi||.

        For the star residual H_R=sum_j(V_j|d><j|+h.c.), H_R psi can be
        evaluated in O((N-L)K) without building a matrix.
        """
        if self.residual.size == 0:
            return 0.0
        denom = _norm(psi)
        if denom <= 1e-15:
            return 0.0
        v = self.model.couplings[self.residual]
        far = psi[self.residual + 1]
        d_out = np.sum(np.conjugate(v)[:, None] * far, axis=0)
        far_norm2 = float(np.sum(np.abs(v) ** 2) * np.vdot(psi[0], psi[0]).real)
        num2 = float(np.vdot(d_out, d_out).real) + far_norm2
        return math.sqrt(max(0.0, num2)) / denom

    def set_block_rate(self, psi_mean: np.ndarray, dt: float) -> Dict[str, float]:
        strength = self.residual_transition_strength(psi_mean)
        lam_raw = self.lambda_scale * strength
        mu_raw = lam_raw * abs(dt)
        if self.residual.size == 0 or strength <= 0.0:
            mu = 0.0
            lam = 0.0
        else:
            mu = min(self.mu_max, max(self.mu_min, mu_raw))
            lam = mu / max(abs(dt), 1e-15)
        self._current_lambda = lam
        self._current_mu = mu
        # Existing path routines read high_order_mu from settings.
        self.settings.high_order_mu = mu
        self._block_path_masses = []
        self._block_orders = []
        return {
            'residual_strength_au': strength,
            'lambda_au_inv': lam,
            'mu': mu,
            'lambda_raw_au_inv': lam_raw,
            'mu_raw': mu_raw,
        }

    def _one_high_order_path_conditional_star(self, source, dt, p_channel, rng):
        # Reproduce parent algorithm while recording sampled order by comparing profile.
        before_events = self.profile['high_order_events']
        before_paths = self.profile['high_order_paths']
        out = super()._one_high_order_path_conditional_star(source, dt, p_channel, rng)
        if self.profile['high_order_paths'] > before_paths:
            n = int(round(self.profile['high_order_events'] - before_events))
            self._block_orders.append(n)
        return out

    def _one_high_order_path(self, source, dt, p_channel, rng):
        before_events = self.profile['high_order_events']
        before_paths = self.profile['high_order_paths']
        out = super()._one_high_order_path(source, dt, p_channel, rng)
        if self.profile['high_order_paths'] > before_paths:
            n = int(round(self.profile['high_order_events'] - before_events))
            self._block_orders.append(n)
        return out

    def _high_order_batch(self, sources, dt, p_channel, rng):
        # Same estimator as the base class, but retain individual weighted path
        # masses for the ESS diagnostic.
        t0 = time.perf_counter()
        B = sources.shape[0]
        M = int(self.settings.high_order_samples)
        out = np.zeros_like(sources)
        mu = float(self.settings.high_order_mu)
        if M <= 0 or self.residual.size == 0 or mu <= 0.0:
            self.profile['high_order_s'] += time.perf_counter() - t0
            return out
        common_weight = _poisson_tail_ge2(mu) / M
        mode = str(self.settings.high_order_channel_mode).lower()
        for b in range(B):
            acc = np.zeros_like(sources[b])
            for _ in range(M):
                if mode == 'conditional_star':
                    path = self._one_high_order_path_conditional_star(sources[b], dt, p_channel, rng)
                else:
                    path = self._one_high_order_path(sources[b], dt, p_channel, rng)
                acc += path
                # Common factor does not affect normalized ESS, but include it
                # so the saved mass has the physical contribution scale.
                self._block_path_masses.append(abs(common_weight) * _norm(path))
            out[b] = common_weight * acc
        self.profile['high_order_s'] += time.perf_counter() - t0
        return out

    def block_ess(self) -> float:
        m = np.asarray(self._block_path_masses, dtype=float)
        if m.size == 0:
            return 1.0
        den = m.size * float(np.sum(m*m))
        return float((np.sum(m)**2) / den) if den > 0.0 else 1.0

    def block_order_stats(self):
        if not self._block_orders:
            return 0.0, 0
        x = np.asarray(self._block_orders, dtype=float)
        return float(np.mean(x)), int(np.max(x))


def run_case(cfg: dict, L: int, outdir: Path, reference_every_fs: float = 1.0) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg['model']['nmetal'] = int(cfg['model'].get('nmetal', 400))
    h = cfg['hybrid']
    h['active_L'] = int(L)
    model = model_from_config(cfg)
    settings = HybridSettings(**h)
    if settings.propagation_mode != 'strict_dyson_poisson':
        raise ValueError('This diagnostic runner requires propagation_mode=strict_dyson_poisson')
    diag = cfg.get('poisson_diagnostics', {})
    sim = PoissonDiagnosticSimulator(
        model, settings,
        lambda_scale=float(diag.get('lambda_scale', 8.0)),
        mu_min=float(diag.get('mu_min', 0.05)),
        mu_max=float(diag.get('mu_max', 0.80)),
    )

    dt = settings.block_fs * FS_TO_AU
    nblocks = int(round(settings.tmax_fs / settings.block_fs))
    output_every = max(1, int(round(settings.output_fs / settings.block_fs)))

    # Independent stochastic groups for cross-group observables.
    groups = np.repeat(model.psi0[None, :, :], settings.groups, axis=0)

    # Deterministic full-H reference, advanced with the same block size only for
    # wavefunction diagnostics.  Its runtime is measured separately and is not
    # included in stochastic CPU time.
    from newns_anderson_linear import ChebyshevPropagator
    full_prop = ChebyshevPropagator(model.build_full_hamiltonian(), tol=2.0e-12)
    ref = model.psi0.copy()

    records = []
    sqerr_sum = 0.0
    sqerr_count = 0
    stochastic_elapsed = 0.0
    reference_elapsed = 0.0

    # Initial record.
    obs0 = group_observables(groups, model)
    refobs0 = state_observables(ref, model)
    records.append({
        'time_fs': 0.0,
        'lambda_au_inv': 0.0,
        'lambda_fs_inv': 0.0,
        'mu': 0.0,
        'residual_strength_au': sim.residual_transition_strength(np.mean(groups, axis=0)),
        'ess_normalized': 1.0,
        'mean_poisson_order': 0.0,
        'max_poisson_order': 0,
        'population_d': obs0['population_d_u'],
        'population_exact': refobs0['population_d_exact'],
        'norm_u': obs0['norm_u'],
        'group_population_std': obs0['group_population_std'],
        'phase_aligned_error': 0.0,
        'cumulative_rms_wavefunction_error': 0.0,
        'cpu_elapsed_s': 0.0,
    })

    block_lambda = block_mu = block_strength = block_ess = 0.0
    block_mean_order = 0.0; block_max_order = 0

    for b in range(1, nblocks + 1):
        if b % 100 == 0:
            print(f"[progress L={L}] block={b}/{nblocks} t={b*settings.block_fs:.1f} fs cpu={stochastic_elapsed:.2f}s norm_mean={_norm(np.mean(groups,axis=0)):.3e}", flush=True)
        psi_mean = np.mean(groups, axis=0)
        rate = sim.set_block_rate(psi_mean, dt)

        t0 = time.perf_counter()
        for g in range(settings.groups):
            groups[g] = sim.propagate_group_block(groups[g], dt, sim.rngs[g])
        stochastic_elapsed += time.perf_counter() - t0

        tr = time.perf_counter()
        ref = full_prop.apply(ref, dt)
        reference_elapsed += time.perf_counter() - tr

        block_lambda = rate['lambda_au_inv']
        block_mu = rate['mu']
        block_strength = rate['residual_strength_au']
        block_ess = sim.block_ess()
        block_mean_order, block_max_order = sim.block_order_stats()

        if b % output_every == 0 or b == nblocks:
            obs = group_observables(groups, model)
            refobs = state_observables(ref, model)
            err = phase_aligned_error(np.mean(groups, axis=0), ref)
            sqerr_sum += err*err
            sqerr_count += 1
            records.append({
                'time_fs': b * settings.block_fs,
                'lambda_au_inv': block_lambda,
                'lambda_fs_inv': block_lambda / AU_TO_FS,
                'mu': block_mu,
                'residual_strength_au': block_strength,
                'ess_normalized': block_ess,
                'mean_poisson_order': block_mean_order,
                'max_poisson_order': block_max_order,
                'population_d': obs['population_d_u'],
                'population_exact': refobs['population_d_exact'],
                'norm_u': obs['norm_u'],
                'group_population_std': obs['group_population_std'],
                'phase_aligned_error': err,
                'cumulative_rms_wavefunction_error': math.sqrt(sqerr_sum/sqerr_count),
                'cpu_elapsed_s': stochastic_elapsed,
            })

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f'L{L:02d}_poisson_diagnostics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)

    pop = np.asarray([r['population_d'] for r in records])
    pop_ref = np.asarray([r['population_exact'] for r in records])
    lam = np.asarray([r['lambda_au_inv'] for r in records[1:]])
    ess = np.asarray([r['ess_normalized'] for r in records[1:]])
    rms = np.asarray([r['cumulative_rms_wavefunction_error'] for r in records])
    summary = {
        'L': int(L),
        'N': int(model.nmetal),
        'K': int(model.nbasis),
        'residual_states': int(sim.residual.size),
        'tmax_fs': float(settings.tmax_fs),
        'block_fs': float(settings.block_fs),
        'groups': int(settings.groups),
        'replicas_per_group': int(settings.replicas_per_group),
        'first_order_samples_per_residual': float(settings.first_order_samples_per_residual),
        'first_order_samples_per_replica': int(sim.first_order_sample_count()),
        'time_quadrature_order': int(settings.time_quadrature_order),
        'high_order_samples_per_replica': int(settings.high_order_samples),
        'high_order_paths_per_block': int(settings.groups * settings.replicas_per_group * settings.high_order_samples),
        'lambda_scale': sim.lambda_scale,
        'mu_min_setting': sim.mu_min,
        'mu_max_setting': sim.mu_max,
        'lambda_mean_au_inv': float(np.mean(lam)) if lam.size else 0.0,
        'lambda_min_au_inv': float(np.min(lam)) if lam.size else 0.0,
        'lambda_max_au_inv': float(np.max(lam)) if lam.size else 0.0,
        'mu_mean': float(np.mean([r['mu'] for r in records[1:]])),
        'ess_mean': float(np.mean(ess)) if ess.size else 1.0,
        'ess_min': float(np.min(ess)) if ess.size else 1.0,
        'fraction_ess_gt_075': float(np.mean(ess >= 0.75)) if ess.size else 1.0,
        'population_rms': float(np.sqrt(np.mean((pop-pop_ref)**2))),
        'population_max_abs': float(np.max(np.abs(pop-pop_ref))),
        'final_cumulative_rms_wavefunction_error': float(rms[-1]),
        'max_cumulative_rms_wavefunction_error': float(np.max(rms)),
        'max_abs_norm_error': float(np.max(np.abs(np.asarray([r['norm_u'] for r in records])-1.0))),
        'stochastic_cpu_s': float(stochastic_elapsed),
        'reference_cpu_s_excluded': float(reference_elapsed),
        'csv': str(csv_path),
    }
    (outdir / f'L{L:02d}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--L', type=int, nargs='+', default=[0,16,32,48,64])
    ap.add_argument('--outdir', type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    summaries=[]
    for L in args.L:
        print(f'[RUN] L={L}', flush=True)
        s=run_case(cfg,L,args.outdir)
        summaries.append(s)
        print(json.dumps(s, indent=2), flush=True)
    if summaries:
        with (args.outdir/'L_sweep_poisson_summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)

if __name__=='__main__':
    main()
