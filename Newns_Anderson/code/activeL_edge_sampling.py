from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy import sparse

from newns_anderson_linear import (
    AU_TO_FS,
    EV_TO_HARTREE,
    FS_TO_AU,
    ActiveSpace,
    ChebyshevPropagator,
    NewnsAndersonModel,
    NewnsAndersonParameters,
    model_from_config,
    state_observables,
    systematic_sample,
)


def _norm2(psi: np.ndarray) -> float:
    return float(np.vdot(psi.ravel(), psi.ravel()).real)


@dataclass(frozen=True)
class DenseCoupling:
    """Energy-local dense metal-metal coupling used for scaling tests.

    W_ab = gamma * dE/(2 sigma) * exp(-|eps_a-eps_b|/sigma), a != b.
    The edge sampler never enumerates the O(N^2) edges.  It samples the energy
    index distance d first and then a uniformly allowed starting index.
    """

    gamma_au: float = 0.004
    sigma_ev: float = 0.5

    @property
    def sigma_au(self) -> float:
        return self.sigma_ev * EV_TO_HARTREE

    def coupling_by_distance(self, model: NewnsAndersonModel) -> np.ndarray:
        n = model.nmetal
        de = model.metal_energy_spacing()
        d = np.arange(1, n, dtype=float)
        return self.gamma_au * de / (2.0 * self.sigma_au) * np.exp(-d * de / self.sigma_au)

    def distance_distribution(self, model: NewnsAndersonModel) -> Tuple[np.ndarray, np.ndarray, float]:
        h = self.coupling_by_distance(model)
        multiplicity = np.arange(model.nmetal - 1, 0, -1, dtype=float)
        weights = multiplicity * np.abs(h)
        z = float(np.sum(weights))
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError("Invalid dense-edge weight normalization")
        return h, weights / z, z


class GeneralActiveSpace(ActiveSpace):
    """Active space with optional exact metal-metal couplings inside the L block."""

    def __init__(
        self,
        model: NewnsAndersonModel,
        active_metal: Sequence[int],
        dense: Optional[DenseCoupling],
        cheb_tol: float = 1.0e-10,
    ) -> None:
        self._dense_spec = dense
        super().__init__(model=model, active_metal=np.asarray(active_metal, dtype=np.int64), cheb_tol=cheb_tol)
        if dense is not None and self.L > 1:
            self.active_h = self._add_dense_active_edges(self.active_h, dense)
            self.propagator = ChebyshevPropagator(self.active_h, tol=cheb_tol)

    def _add_dense_active_edges(self, h0: sparse.csr_matrix, dense: DenseCoupling) -> sparse.csr_matrix:
        k = self.model.nbasis
        eye = sparse.identity(k, dtype=np.complex128, format="csr")
        active = self.active_metal
        blocks = h0.tolil(copy=True)
        de = self.model.metal_energy_spacing()
        sigma = dense.sigma_au
        pref = dense.gamma_au * de / (2.0 * sigma)
        for ia in range(active.size):
            for ib in range(ia + 1, active.size):
                a, b = int(active[ia]), int(active[ib])
                val = pref * math.exp(-abs(a - b) * de / sigma)
                ra = slice((ia + 1) * k, (ia + 2) * k)
                rb = slice((ib + 1) * k, (ib + 2) * k)
                blocks[ra, rb] = val * eye
                blocks[rb, ra] = val * eye
        return blocks.tocsr()


@dataclass
class EdgeSamplingSettings:
    tmax_fs: float = 200.0
    block_fs: float = 0.1
    output_fs: float = 1.0
    active_L: int = 16
    trajectories: int = 4
    star_samples_per_residual: float = 2.0
    dense_samples_per_state: float = 1.0
    probability_uniform_mix: float = 0.05
    probability_broadening_ev: float = 0.15
    seed: int = 20260806
    cheb_tol: float = 1.0e-10


class ActiveLEdgeSampler:
    """Unitary active-L plus sampled-residual propagator.

    The residual generator is decomposed into Hermitian edges.  Star edges are
    sampled by random-start systematic sampling and applied collectively as an
    exact bright-state rotation.  Dense metal-metal edges are sampled without
    enumerating the O(N^2) edge list and applied by exact two-state rotations.
    Every trajectory is norm preserving to roundoff.
    """

    def __init__(
        self,
        model: NewnsAndersonModel,
        settings: EdgeSamplingSettings,
        dense: Optional[DenseCoupling] = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.dense = dense
        active = model.choose_active_metal_states(settings.active_L)
        self.space = GeneralActiveSpace(model, active, dense=dense, cheb_tol=settings.cheb_tol)
        self.active_set = set(int(x) for x in self.space.active_metal)
        self.residual = self.space.residual_metal
        self.detuning = model.vertical_detunings()
        self.rngs = [np.random.default_rng(settings.seed + 104729 * r) for r in range(settings.trajectories)]
        if dense is not None:
            self.dense_h_by_d, self.dense_p_d, self.dense_z = dense.distance_distribution(model)
        else:
            self.dense_h_by_d = self.dense_p_d = None
            self.dense_z = 0.0

    def star_probabilities(self, dt: float) -> np.ndarray:
        if self.residual.size == 0:
            return np.zeros(0, dtype=float)
        delta = np.abs(self.detuning[self.residual])
        eta = self.settings.probability_broadening_ev * EV_TO_HARTREE
        envelope = np.minimum(abs(dt), 2.0 / np.sqrt(delta * delta + eta * eta))
        score = np.abs(self.model.couplings[self.residual]) * envelope
        if not np.any(score > 0.0):
            score = np.ones_like(score)
        p = score / np.sum(score)
        mix = float(np.clip(self.settings.probability_uniform_mix, 0.0, 1.0))
        p = (1.0 - mix) * p + mix / p.size
        return p / np.sum(p)

    def _apply_star_bright_rotation(self, psi: np.ndarray, dt: float, rng: np.random.Generator) -> None:
        nr = int(self.residual.size)
        if nr == 0 or abs(dt) < 1.0e-16:
            return
        m = max(1, int(math.ceil(self.settings.star_samples_per_residual * nr)))
        p = self.star_probabilities(dt)
        draw = systematic_sample(p, m, rng)
        counts = np.bincount(draw, minlength=nr).astype(float)
        g = counts * self.model.couplings[self.residual] / (m * p)
        gnorm = float(np.linalg.norm(g))
        if gnorm <= 0.0:
            return
        ridx = self.residual + 1
        a = psi[0].copy()
        # |B> = sum_k g_k^*/||g|| |k>, hence <B|psi> = sum_k g_k psi_k/||g||.
        b = np.tensordot(g / gnorm, psi[ridx], axes=(0, 0))
        theta = gnorm * dt
        c, s = math.cos(theta), math.sin(theta)
        anew = c * a - 1j * s * b
        bnew = c * b - 1j * s * a
        psi[0] = anew
        psi[ridx] += (np.conjugate(g) / gnorm)[:, None] * (bnew - b)[None, :]

    @staticmethod
    def _rotate_edge(psi: np.ndarray, ia: int, ib: int, h: complex, dt: float) -> None:
        amp = abs(h)
        if amp <= 0.0 or abs(dt) < 1.0e-16:
            return
        phase = h / amp
        theta = amp * dt
        c, s = math.cos(theta), math.sin(theta)
        a = psi[ia].copy()
        b = psi[ib].copy()
        psi[ia] = c * a - 1j * s * phase * b
        psi[ib] = c * b - 1j * s * np.conjugate(phase) * a

    def _sample_dense_edges(self, rng: np.random.Generator) -> Dict[Tuple[int, int], Tuple[int, float]]:
        if self.dense is None or self.model.nmetal < 2:
            return {}
        m = max(1, int(math.ceil(self.settings.dense_samples_per_state * self.model.nmetal)))
        ddraw = systematic_sample(self.dense_p_d, m, rng) + 1
        edges: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for d in ddraw:
            a = int(rng.integers(0, self.model.nmetal - int(d)))
            b = a + int(d)
            key = (a, b)
            count, hd = edges.get(key, (0, float(self.dense_h_by_d[int(d) - 1])))
            edges[key] = (count + 1, hd)
        return edges

    def _apply_dense_palindrome(self, psi: np.ndarray, dt: float, rng: np.random.Generator) -> None:
        if self.dense is None:
            return
        m = max(1, int(math.ceil(self.settings.dense_samples_per_state * self.model.nmetal)))
        edges = self._sample_dense_edges(rng)
        sequence = []
        for (a, b), (count, hd) in edges.items():
            # Active-active dense edges are already exact in H_A, so their residual edge is zero.
            if a in self.active_set and b in self.active_set:
                continue
            p_edge = abs(hd) / self.dense_z
            heff = count * hd / (m * p_edge)
            sequence.append((a + 1, b + 1, complex(heff)))
        for ia, ib, h in sequence:
            self._rotate_edge(psi, ia, ib, h, 0.5 * dt)
        for ia, ib, h in reversed(sequence):
            self._rotate_edge(psi, ia, ib, h, 0.5 * dt)

    def block(self, psi: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        out = self.space.propagate_full(psi, 0.5 * dt)
        self._apply_star_bright_rotation(out, 0.5 * dt, rng)
        self._apply_dense_palindrome(out, dt, rng)
        self._apply_star_bright_rotation(out, 0.5 * dt, rng)
        out = self.space.propagate_full(out, 0.5 * dt)
        return out

    def run(self) -> Dict[str, np.ndarray]:
        dt = self.settings.block_fs * FS_TO_AU
        nblocks = int(round(self.settings.tmax_fs / self.settings.block_fs))
        out_every = max(1, int(round(self.settings.output_fs / self.settings.block_fs)))
        particles = np.repeat(self.model.psi0[None, :, :], self.settings.trajectories, axis=0)
        rows = []
        start = time.perf_counter()
        for b in range(nblocks + 1):
            if b % out_every == 0:
                obs = [state_observables(p, self.model) for p in particles]
                rows.append((
                    b * self.settings.block_fs,
                    float(np.mean([x["population_d_exact"] for x in obs])),
                    float(np.std([x["population_d_exact"] for x in obs], ddof=1)) if len(obs) > 1 else 0.0,
                    float(np.mean([x["norm_exact"] for x in obs])),
                    float(max(abs(x["norm_exact"] - 1.0) for x in obs)),
                ))
            if b == nblocks:
                break
            for r, rng in enumerate(self.rngs):
                particles[r] = self.block(particles[r], dt, rng)
        elapsed = time.perf_counter() - start
        arr = np.asarray(rows, dtype=float)
        return {
            "time_fs": arr[:, 0],
            "population": arr[:, 1],
            "population_std": arr[:, 2],
            "mean_norm": arr[:, 3],
            "max_norm_error": arr[:, 4],
            "elapsed_s": np.asarray([elapsed]),
        }


def _load_settings(cfg: dict) -> EdgeSamplingSettings:
    return EdgeSamplingSettings(**cfg.get("edge_sampling", {}))


def save_csv(path: Path, result: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["time_fs", "population", "population_std", "mean_norm", "max_norm_error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in zip(*(result[k] for k in keys)):
            w.writerow([f"{float(x):.16e}" for x in row])


def main() -> None:
    ap = argparse.ArgumentParser(description="Active-L near-linear edge-sampled Newns-Anderson dynamics")
    ap.add_argument("config", type=str)
    ap.add_argument("--output", type=str, default="run_output")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = model_from_config(cfg)
    settings = _load_settings(cfg)
    dense_cfg = cfg.get("dense_coupling", {})
    dense = DenseCoupling(**{k: v for k, v in dense_cfg.items() if k != "enabled"}) if dense_cfg.get("enabled", False) else None
    sim = ActiveLEdgeSampler(model, settings, dense=dense)
    result = sim.run()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "trajectory.csv", result)
    meta = {
        "nmetal": model.nmetal,
        "nbasis": model.nbasis,
        "active_L": settings.active_L,
        "dense_enabled": dense is not None,
        "elapsed_s": float(result["elapsed_s"][0]),
        "max_norm_error": float(np.max(result["max_norm_error"])),
        "settings": settings.__dict__,
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
