from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.special import jv

# Atomic-unit conversions (CODATA-compatible values used throughout the package).
EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_EV = 1.0 / EV_TO_HARTREE
FS_TO_AU = 41.3413745758
AU_TO_FS = 1.0 / FS_TO_AU
AMU_TO_ME = 1822.888486209
ANGSTROM_TO_BOHR = 1.889726125


def _complex_norm(x: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, np.vdot(x.ravel(), x.ravel()).real)))


def _poisson_tail_ge2(mu: float) -> float:
    if mu <= 0.0:
        return 0.0
    return math.exp(mu) - 1.0 - mu


def sample_poisson_conditioned_ge2(mu: float, rng: np.random.Generator) -> int:
    """Sample Poisson(mu) conditioned on n >= 2 by inverse recurrence.

    The common exp(-mu) factor cancels after conditioning. Starting from
    w_2=mu^2/2, w_{n+1}=w_n*mu/(n+1) builds the conditional tail without
    rejection, which remains efficient for small adaptive block intensities.
    """
    if mu <= 1.0e-14:
        return 2
    tail = math.exp(mu) - 1.0 - mu
    target = rng.random() * tail
    n = 2
    w = 0.5 * mu * mu
    acc = w
    while target > acc:
        n += 1
        w *= mu / n
        acc += w
        if n > 10000:
            raise RuntimeError(f"Conditioned Poisson inverse sampler stalled at mu={mu}")
    return n

def systematic_sample(prob: np.ndarray, nsamples: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic categorical sampling; lower variance than iid multinomial sampling."""
    if nsamples <= 0:
        return np.zeros(0, dtype=np.int64)
    p = np.asarray(prob, dtype=float)
    p = p / max(np.sum(p), 1.0e-300)
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    points = (rng.random() + np.arange(nsamples)) / nsamples
    return np.searchsorted(cdf, points, side="right").astype(np.int64)


@dataclass(frozen=True)
class NewnsAndersonParameters:
    """Parameters for displaced-harmonic Newns--Anderson models.

    The legacy eV/amu/Angstrom fields preserve the original package.  The
    optional direct atomic-unit fields are used for literature benchmarks
    whose parameters are reported in atomic units (for example the
    Jin--Subotnik Anderson--Holstein benchmark).
    """

    hw_ev: float = 0.1
    mass_amu: float = 8.0
    x0_angstrom: float = 1.06
    epsilon0_ev: float = 2.5
    band_bottom_ev: float = -10.0
    coupling_au: float = 0.01

    omega_au: Optional[float] = None
    mass_au: Optional[float] = None
    x0_bohr: Optional[float] = None
    epsilon0_au: Optional[float] = None
    band_min_au: Optional[float] = None
    band_max_au: Optional[float] = None
    gamma_au: Optional[float] = None
    initial_q_bohr: Optional[float] = None
    initial_p_au: Optional[float] = None
    benchmark_name: str = "legacy"

    @property
    def omega(self) -> float:
        return float(self.omega_au) if self.omega_au is not None else self.hw_ev * EV_TO_HARTREE

    @property
    def mass(self) -> float:
        return float(self.mass_au) if self.mass_au is not None else self.mass_amu * AMU_TO_ME

    @property
    def x0(self) -> float:
        return float(self.x0_bohr) if self.x0_bohr is not None else self.x0_angstrom * ANGSTROM_TO_BOHR

    @property
    def epsilon0(self) -> float:
        return float(self.epsilon0_au) if self.epsilon0_au is not None else self.epsilon0_ev * EV_TO_HARTREE

    @property
    def band_min(self) -> float:
        if self.band_min_au is not None:
            return float(self.band_min_au)
        return float(self.band_bottom_ev) * EV_TO_HARTREE

    @property
    def band_max(self) -> float:
        if self.band_max_au is not None:
            return float(self.band_max_au)
        return 0.0

    @property
    def gamma(self) -> Optional[float]:
        return None if self.gamma_au is None else float(self.gamma_au)

    @property
    def displacement_alpha(self) -> float:
        return math.sqrt(self.mass * self.omega / 2.0) * self.x0

    @property
    def reorganization_au(self) -> float:
        return 0.5 * self.mass * self.omega**2 * self.x0**2

    @property
    def reorganization_ev(self) -> float:
        return self.reorganization_au * HARTREE_TO_EV


@dataclass
class NewnsAndersonModel:
    nmetal: int
    nbasis: int
    params: NewnsAndersonParameters
    coupling_mode: str = "paper"
    fixed_hybridization_reference_n: int = 20
    metal_grid: str = "legacy"

    def __post_init__(self) -> None:
        if self.nmetal < 1:
            raise ValueError("nmetal must be positive")
        if self.nbasis < 8:
            raise ValueError("nbasis is too small")
        self.energies = self._build_metal_energies()
        self.couplings = self._build_couplings()
        self.hn_diag = self.params.omega * (np.arange(self.nbasis, dtype=float) + 0.5)
        self.hc = self._build_charged_oscillator()
        self.psi0 = self._build_initial_state()
        self.x_operator = self._build_x_operator()

    @property
    def nelectronic(self) -> int:
        return self.nmetal + 1

    def _build_metal_energies(self) -> np.ndarray:
        grid = str(self.metal_grid).lower()
        emin, emax = self.params.band_min, self.params.band_max
        if grid == "legacy":
            # Original package convention: epsilon_j = E_min * j / N, j=1,...,N.
            j = np.arange(1, self.nmetal + 1, dtype=float)
            return emin * j / self.nmetal
        if grid == "midpoint":
            de = (emax - emin) / self.nmetal
            return emin + (np.arange(self.nmetal, dtype=float) + 0.5) * de
        if grid == "endpoints":
            if self.nmetal == 1:
                return np.asarray([0.5 * (emin + emax)], dtype=float)
            return np.linspace(emin, emax, self.nmetal, dtype=float)
        raise ValueError(f"Unsupported metal_grid={self.metal_grid!r}")

    def metal_energy_spacing(self) -> float:
        span = self.params.band_max - self.params.band_min
        if str(self.metal_grid).lower() == "endpoints" and self.nmetal > 1:
            return span / (self.nmetal - 1)
        return span / self.nmetal

    def _build_couplings(self) -> np.ndarray:
        mode = str(self.coupling_mode).lower()
        if mode == "paper":
            f = self.params.coupling_au
        elif mode == "fixed_hybridization":
            f = self.params.coupling_au * math.sqrt(self.fixed_hybridization_reference_n / self.nmetal)
        elif mode == "wide_band_gamma":
            if self.params.gamma is None:
                raise ValueError("wide_band_gamma requires parameters.gamma_au")
            de = self.metal_energy_spacing()
            if de <= 0.0:
                raise ValueError("Metal bandwidth must be positive for wide_band_gamma")
            # Gamma(epsilon)=2*pi*|V_k|^2/delta_epsilon.
            f = math.sqrt(self.params.gamma * de / (2.0 * math.pi))
        else:
            raise ValueError(f"Unsupported coupling_mode={self.coupling_mode!r}")
        return np.full(self.nmetal, complex(f), dtype=np.complex128)

    def _build_charged_oscillator(self) -> sparse.csr_matrix:
        # Neutral-oscillator Fock basis:
        # H_c = omega(N+1/2) - omega*alpha(a+a^dagger) + omega*alpha^2 + epsilon0.
        k = self.nbasis
        alpha = self.params.displacement_alpha
        diag = self.hn_diag + self.params.omega * alpha * alpha + self.params.epsilon0
        off = -self.params.omega * alpha * np.sqrt(np.arange(1, k, dtype=float))
        return sparse.diags([off, diag, off], offsets=[-1, 0, 1], format="csr", dtype=np.complex128)

    def _build_initial_state(self) -> np.ndarray:
        # Coherent state in the neutral-oscillator Fock basis.  By default this
        # is the ground state of the displaced molecular/charged surface.
        # Literature wave-packet benchmarks may instead supply an arbitrary
        # phase-space centre (q,p), where p is the momentum in the standard
        # hbar=1 Schroedinger equation used by this implementation.
        if self.params.initial_q_bohr is None and self.params.initial_p_au is None:
            alpha = complex(self.params.displacement_alpha)
        else:
            q = float(self.params.initial_q_bohr or 0.0)
            p = float(self.params.initial_p_au or 0.0)
            alpha = (
                math.sqrt(self.params.mass * self.params.omega / 2.0) * q
                + 1j * p / math.sqrt(2.0 * self.params.mass * self.params.omega)
            )
        coeff = np.empty(self.nbasis, dtype=np.complex128)
        coeff[0] = np.exp(-0.5 * abs(alpha) ** 2)
        for n in range(1, self.nbasis):
            coeff[n] = coeff[n - 1] * alpha / math.sqrt(n)
        coeff /= _complex_norm(coeff)
        psi = np.zeros((self.nelectronic, self.nbasis), dtype=np.complex128)
        psi[0] = coeff
        return psi

    def _build_x_operator(self) -> sparse.csr_matrix:
        pref = 1.0 / math.sqrt(2.0 * self.params.mass * self.params.omega)
        off = pref * np.sqrt(np.arange(1, self.nbasis, dtype=float))
        return sparse.diags([off, off], offsets=[-1, 1], format="csr", dtype=np.complex128)

    def vertical_detunings(self, x_reference: Optional[float] = None) -> np.ndarray:
        """Metal-minus-molecular vertical energy gaps at a fixed nuclear coordinate."""
        x = self.params.x0 if x_reference is None else float(x_reference)
        vn = 0.5 * self.params.mass * self.params.omega**2 * x * x
        vc = 0.5 * self.params.mass * self.params.omega**2 * (x - self.params.x0) ** 2 + self.params.epsilon0
        return self.energies + vn - vc

    def choose_active_metal_states(self, L: int, x_reference: Optional[float] = None) -> np.ndarray:
        L = int(max(0, min(L, self.nmetal)))
        detuning = np.abs(self.vertical_detunings(x_reference=x_reference))
        return np.sort(np.argsort(detuning)[:L]).astype(np.int64)

    def build_active_hamiltonian(self, active_metal: Sequence[int]) -> sparse.csr_matrix:
        active = np.asarray(active_metal, dtype=np.int64)
        blocks: List[List[Optional[sparse.spmatrix]]] = []
        eye_k = sparse.identity(self.nbasis, dtype=np.complex128, format="csr")
        nact = active.size
        for a in range(nact + 1):
            row: List[Optional[sparse.spmatrix]] = []
            for b in range(nact + 1):
                if a == 0 and b == 0:
                    row.append(self.hc)
                elif a == b and a > 0:
                    j = active[a - 1]
                    row.append(sparse.diags(self.hn_diag + self.energies[j], format="csr", dtype=np.complex128))
                elif a == 0 and b > 0:
                    j = active[b - 1]
                    row.append(self.couplings[j] * eye_k)
                elif b == 0 and a > 0:
                    j = active[a - 1]
                    row.append(np.conjugate(self.couplings[j]) * eye_k)
                else:
                    row.append(None)
            blocks.append(row)
        return sparse.bmat(blocks, format="csr", dtype=np.complex128)

    def build_full_hamiltonian(self) -> sparse.csr_matrix:
        return self.build_active_hamiltonian(np.arange(self.nmetal, dtype=np.int64))


class ChebyshevPropagator:
    """Hermitian matrix exponential action using a scaled Chebyshev series."""

    def __init__(self, hamiltonian: sparse.csr_matrix, tol: float = 1.0e-11) -> None:
        self.h = hamiltonian.tocsr()
        self.tol = float(tol)
        dim = self.h.shape[0]
        if dim <= 3:
            evals = np.linalg.eigvalsh(self.h.toarray())
            emin, emax = float(evals[0]), float(evals[-1])
        else:
            # Gershgorin bounds are deterministic, O(nnz), and avoid occasional
            # slow convergence of extremal eigensolvers for large active spaces.
            diag = self.h.diagonal().real
            row_abs = np.asarray(np.abs(self.h).sum(axis=1)).ravel()
            off_radius = np.maximum(0.0, row_abs - np.abs(self.h.diagonal()))
            emin = float(np.min(diag - off_radius))
            emax = float(np.max(diag + off_radius))
        pad = 1.0e-10 + 1.0e-8 * max(1.0, abs(emin), abs(emax))
        self.emin = emin - pad
        self.emax = emax + pad
        self.center = 0.5 * (self.emax + self.emin)
        self.radius = max(1.0e-14, 0.5 * (self.emax - self.emin))
        self.scaled_h = (self.h - self.center * sparse.identity(dim, format="csr", dtype=np.complex128)) * (1.0 / self.radius)
        self.max_order_seen = 0

    def _order(self, z: float) -> int:
        az = abs(z)
        if az < 1.0e-14:
            return 0
        mmax = int(max(24, math.ceil(az + 12.0 * az ** (1.0 / 3.0) + 24.0)))
        consecutive = 0
        for m in range(0, mmax + 1):
            if m > az and abs(jv(m, z)) < self.tol * 0.05:
                consecutive += 1
                if consecutive >= 6:
                    return m
            else:
                consecutive = 0
        return mmax

    def apply(self, x: np.ndarray, dt: float) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex128)
        original_shape = arr.shape
        if arr.ndim == 1:
            mat = arr.reshape((-1, 1))
        else:
            mat = arr.reshape((self.h.shape[0], -1))
        if abs(dt) < 1.0e-15:
            return arr.copy()
        z = self.radius * dt
        order = self._order(z)
        self.max_order_seen = max(self.max_order_seen, order)
        t0 = mat.copy()
        result = jv(0, z) * t0
        if order >= 1:
            t1 = self.scaled_h @ t0
            result = result + 2.0 * ((-1j) ** 1) * jv(1, z) * t1
            for m in range(2, order + 1):
                t2 = 2.0 * (self.scaled_h @ t1) - t0
                result = result + 2.0 * ((-1j) ** m) * jv(m, z) * t2
                t0, t1 = t1, t2
        result *= np.exp(-1j * self.center * dt)
        return result.reshape(original_shape)


@dataclass
class ActiveSpace:
    model: NewnsAndersonModel
    active_metal: np.ndarray
    cheb_tol: float = 1.0e-11

    def __post_init__(self) -> None:
        self.active_metal = np.asarray(self.active_metal, dtype=np.int64)
        all_metal = np.arange(self.model.nmetal, dtype=np.int64)
        mask = np.ones(self.model.nmetal, dtype=bool)
        mask[self.active_metal] = False
        self.residual_metal = all_metal[mask]
        self.active_electronic = np.concatenate(([0], self.active_metal + 1)).astype(np.int64)
        self.active_h = self.model.build_active_hamiltonian(self.active_metal)
        self.propagator = ChebyshevPropagator(self.active_h, tol=self.cheb_tol)

    @property
    def L(self) -> int:
        return int(self.active_metal.size)

    def _flatten_active(self, active: np.ndarray) -> np.ndarray:
        # Input (..., L+1, K) -> ((L+1)K, batch)
        a = np.asarray(active, dtype=np.complex128)
        lead = a.shape[:-2]
        batch = int(np.prod(lead)) if lead else 1
        return np.moveaxis(a.reshape((batch, (self.L + 1) * self.model.nbasis)), 0, 1)

    def _restore_active(self, flat: np.ndarray, lead_shape: Tuple[int, ...]) -> np.ndarray:
        batch = int(np.prod(lead_shape)) if lead_shape else 1
        a = np.moveaxis(flat, 0, 1).reshape(lead_shape + (self.L + 1, self.model.nbasis))
        return a

    def propagate_active(self, active: np.ndarray, dt: float) -> np.ndarray:
        lead = tuple(active.shape[:-2])
        flat = self._flatten_active(active)
        out = self.propagator.apply(flat, dt)
        return self._restore_active(out, lead)

    def propagate_full(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Apply the active-space base propagator to one or many full wavefunctions."""
        p = np.asarray(psi, dtype=np.complex128)
        if p.ndim == 2:
            p = p[None, ...]
            squeeze = True
        elif p.ndim == 3:
            squeeze = False
        else:
            raise ValueError("psi must have shape (E,K) or (B,E,K)")
        out = np.zeros_like(p)
        active = p[:, self.active_electronic, :]
        out[:, self.active_electronic, :] = self.propagate_active(active, dt)
        if self.residual_metal.size:
            e = self.model.energies[self.residual_metal][:, None]
            phase = np.exp(-1j * (self.model.hn_diag[None, :] + e) * dt)
            out[:, self.residual_metal + 1, :] = p[:, self.residual_metal + 1, :] * phase[None, :, :]
        return out[0] if squeeze else out

    def propagate_compressed(
        self,
        active: np.ndarray,
        far: np.ndarray,
        far_metal: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        aout = self.propagate_active(active, dt)
        idx = np.asarray(far_metal, dtype=np.int64).reshape((-1,))
        f = np.asarray(far, dtype=np.complex128)
        lead = f.shape[:-1]
        ff = f.reshape((-1, self.model.nbasis))
        phase = np.exp(-1j * (self.model.hn_diag[None, :] + self.model.energies[idx, None]) * dt)
        ff = ff * phase
        return aout, ff.reshape(lead + (self.model.nbasis,))


@dataclass
class HybridSettings:
    tmax_fs: float = 200.0
    block_fs: float = 1.0
    output_fs: float = 1.0
    active_L: int = 8
    active_reference_q_bohr: Optional[float] = None
    propagation_mode: str = "unitary_sampled_hamiltonian"
    groups: int = 4
    replicas_per_group: int = 4
    first_order_samples: int = 16
    first_order_samples_per_residual: float = 0.0
    # ``all_channels`` means that every residual electronic channel is included
    # once in every time block.  The time integral is nevertheless evaluated
    # stochastically, so the cost is O(N-L), not an explicit channel-pair/time
    # quadrature.  ``sampled_channels`` retains the older Horvitz--Thompson
    # channel sampler, with all residual channels in its probability support.
    first_order_channel_mode: str = "sampled_channels"
    # Number of randomized time strata.  One random time is drawn in each
    # stratum and channels are randomly assigned to strata.  Every channel has
    # a uniform marginal time distribution, making the first-order time
    # integral unbiased while requiring only this many active propagations.
    time_quadrature_order: int = 2
    high_order_samples: int = 2
    high_order_mu: float = 0.7
    high_order_channel_mode: str = "conditional_star"
    probability_mode: str = "detuning"
    probability_uniform_mix: float = 0.02
    probability_broadening_ev: float = 0.15
    seed: int = 20260804
    cheb_tol: float = 1.0e-10


class HybridNewnsAndersonSimulator:
    def __init__(self, model: NewnsAndersonModel, settings: HybridSettings) -> None:
        self.model = model
        self.settings = settings
        active = model.choose_active_metal_states(settings.active_L, x_reference=settings.active_reference_q_bohr)
        self.space = ActiveSpace(model, active, cheb_tol=settings.cheb_tol)
        self.residual = self.space.residual_metal
        self.detuning = model.vertical_detunings()
        self.rngs = [np.random.default_rng(settings.seed + 104729 * g) for g in range(settings.groups)]
        self.profile: Dict[str, float] = {
            "channel_probability_s": 0.0,
            "zero_order_s": 0.0,
            "first_order_s": 0.0,
            "high_order_s": 0.0,
            "observables_s": 0.0,
            "first_order_calls": 0.0,
            "first_order_channel_time_samples": 0.0,
            "first_order_random_stratum_points": 0.0,
            "first_order_channel_stratum_assignments": 0.0,
            "high_order_paths": 0.0,
            "high_order_events": 0.0,
            "high_order_max_events": 0.0,
        }
        if settings.time_quadrature_order < 1:
            raise ValueError("time_quadrature_order must be positive")

    def first_order_sample_count(self) -> int:
        fixed = int(max(0, self.settings.first_order_samples))
        scaled = int(math.ceil(max(0.0, self.settings.first_order_samples_per_residual) * self.residual.size))
        return max(fixed, scaled)

    def channel_probabilities(self, psi: np.ndarray, dt: float) -> np.ndarray:
        if self.residual.size == 0:
            return np.zeros(0, dtype=float)
        mode = str(self.settings.probability_mode).lower()
        if mode == "uniform":
            # For constant-coupling wide-band Hamiltonians this minimizes the
            # Horvitz--Thompson operator variance. Combined with systematic
            # sampling it also stratifies the full energy interval.
            return np.full(self.residual.size, 1.0 / self.residual.size, dtype=float)
        if mode in {"coupling", "hamiltonian_norm"}:
            score = np.abs(self.model.couplings[self.residual])
            if not np.any(score > 0.0):
                score = np.ones_like(score)
            return score / np.sum(score)
        if mode != "detuning":
            raise ValueError(f"Unknown probability_mode={self.settings.probability_mode!r}")
        dnorm = _complex_norm(psi[0])
        jnorm = np.sqrt(np.sum(np.abs(psi[self.residual + 1]) ** 2, axis=1).real)
        delta = np.abs(self.detuning[self.residual])
        broad = self.settings.probability_broadening_ev * EV_TO_HARTREE
        # Oscillatory first-order integral envelope, regularized near resonance.
        eff_delta = np.sqrt(delta * delta + broad * broad)
        filter_amp = np.minimum(abs(dt), 2.0 / np.maximum(eff_delta, 1.0e-14))
        score = np.abs(self.model.couplings[self.residual]) * filter_amp * (dnorm + jnorm + 1.0e-14)
        if not np.any(score > 0.0):
            score = np.ones_like(score)
        p = score / np.sum(score)
        mix = float(np.clip(self.settings.probability_uniform_mix, 0.0, 1.0))
        p = (1.0 - mix) * p + mix / p.size
        p /= np.sum(p)
        return p

    def _first_order_batch(
        self,
        sources: np.ndarray,
        dt: float,
        p_channel: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _profile_start = time.perf_counter()
        B = sources.shape[0]
        out = np.zeros_like(sources)
        R = int(self.residual.size)
        if R == 0:
            self.profile["first_order_s"] += time.perf_counter() - _profile_start
            self.profile["first_order_calls"] += 1.0
            return out

        channel_mode = str(self.settings.first_order_channel_mode).lower()
        if channel_mode == "all_channels":
            # Every residual channel is represented exactly once.  Randomness
            # enters only through its time sample, so no channel is deleted and
            # the channel sum remains O(R).
            sampled_local = np.repeat(np.arange(R, dtype=np.int64)[None, :], B, axis=0)
            sample_weights = np.full((B, R), dt, dtype=float)
            M = R
        elif channel_mode == "sampled_channels":
            M = self.first_order_sample_count()
            if M <= 0:
                self.profile["first_order_s"] += time.perf_counter() - _profile_start
                self.profile["first_order_calls"] += 1.0
                return out
            sampled_local = np.stack([systematic_sample(p_channel, M, rng) for _ in range(B)], axis=0)
            q = p_channel[sampled_local]
            sample_weights = dt / (M * q)
        else:
            raise ValueError(f"Unknown first_order_channel_mode={self.settings.first_order_channel_mode!r}")

        sampled_metal = self.residual[sampled_local]
        nstrata = int(self.settings.time_quadrature_order)
        self.profile["first_order_calls"] += 1.0
        self.profile["first_order_channel_time_samples"] += float(B * M)
        self.profile["first_order_random_stratum_points"] += float(nstrata)
        self.profile["first_order_channel_stratum_assignments"] += float(B * M)
        # A random point in each equal-width stratum.  Assigning each channel
        # independently and uniformly to a stratum gives every channel a
        # uniform marginal time on [0,dt].
        taus = (np.arange(nstrata, dtype=float) + rng.random(nstrata)) * (dt / nstrata)
        stratum = rng.integers(0, nstrata, size=(B, M), endpoint=False)

        for sidx, tau in enumerate(taus):
            src_tau = self.space.propagate_full(sources, float(tau))
            remaining = dt - float(tau)
            for b in range(B):
                pos = np.nonzero(stratum[b] == sidx)[0]
                if pos.size == 0:
                    continue
                metals = sampled_metal[b, pos]
                weights = sample_weights[b, pos]
                couplings = self.model.couplings[metals]

                # Sum all residual->molecule kicks before active propagation.
                # This is the key linear-scaling aggregation: one active-space
                # propagation per time stratum, not one propagation per channel.
                kick_active = np.zeros((self.space.L + 1, self.model.nbasis), dtype=np.complex128)
                kick_active[0] = np.sum(
                    (-1j * weights * np.conjugate(couplings))[:, None]
                    * src_tau[b, metals + 1, :],
                    axis=0,
                )
                out[b, self.space.active_electronic, :] += self.space.propagate_active(kick_active, remaining)

                # Molecule->residual kicks remain diagonal under the base
                # Hamiltonian and can be propagated for all channels in O(RK).
                far = (-1j * weights * couplings)[:, None] * src_tau[b, 0, :][None, :]
                phase = np.exp(
                    -1j
                    * (self.model.hn_diag[None, :] + self.model.energies[metals, None])
                    * remaining
                )
                far *= phase
                np.add.at(out[b], metals + 1, far)
        self.profile["first_order_s"] += time.perf_counter() - _profile_start
        return out

    def _one_high_order_path_conditional_star(
        self,
        source: np.ndarray,
        dt: float,
        p_channel: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Conditional Poisson path for the star-coupled residual Hamiltonian.

        When the path is on a residual state j, the next residual action can
        only return it to the molecular state; this return is evaluated
        deterministically.  Only the molecular->residual branch is sampled.
        This preserves unbiasedness but removes the O(R) mismatch variance of
        independently resampling a channel for a forced return event.
        """
        mu = float(self.settings.high_order_mu)
        lam = mu / max(abs(dt), 1.0e-14)
        n = sample_poisson_conditioned_ge2(mu, rng)
        self.profile["high_order_paths"] += 1.0
        self.profile["high_order_events"] += float(n)
        self.profile["high_order_max_events"] = max(self.profile["high_order_max_events"], float(n))
        times = np.sort(rng.uniform(0.0, dt, size=n))

        full = self.space.propagate_full(source, float(times[0]))
        # Exact residual->molecule contraction is O(RK), while the outgoing
        # molecule->residual branch is sampled over the full residual support.
        d_return = np.sum(
            np.conjugate(self.model.couplings[self.residual])[:, None]
            * full[self.residual + 1, :],
            axis=0,
        )
        local = int(rng.choice(self.residual.size, p=p_channel))
        current_j = int(self.residual[local])
        qj = float(p_channel[local])
        fj = complex(self.model.couplings[current_j])
        active = np.zeros((1, self.space.L + 1, self.model.nbasis), dtype=np.complex128)
        active[0, 0] = (-1j / lam) * d_return
        far = ((-1j * fj / (lam * qj)) * full[0])[None, :]
        prev = float(times[0])

        for event_idx in range(1, n):
            tau = float(times[event_idx])
            active, far = self.space.propagate_compressed(active, far, np.asarray([current_j]), tau - prev)
            d_amp = active[0, 0].copy()
            far_amp = far[0].copy()

            # Forced return from the currently occupied residual channel.
            new_active = np.zeros_like(active)
            new_active[0, 0] = (-1j * np.conjugate(self.model.couplings[current_j]) / lam) * far_amp

            # Sample only the outgoing branch from the molecular amplitude.
            local = int(rng.choice(self.residual.size, p=p_channel))
            current_j = int(self.residual[local])
            qj = float(p_channel[local])
            fj = complex(self.model.couplings[current_j])
            far = ((-1j * fj / (lam * qj)) * d_amp)[None, :]
            active = new_active
            prev = tau

        active, far = self.space.propagate_compressed(active, far, np.asarray([current_j]), dt - prev)
        out = np.zeros_like(source)
        out[self.space.active_electronic] = active[0]
        out[current_j + 1] += far[0]
        return out

    def _one_high_order_path(
        self,
        source: np.ndarray,
        dt: float,
        p_channel: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mu = float(self.settings.high_order_mu)
        lam = mu / max(abs(dt), 1.0e-14)
        n = sample_poisson_conditioned_ge2(mu, rng)
        self.profile["high_order_paths"] += 1.0
        self.profile["high_order_events"] += float(n)
        self.profile["high_order_max_events"] = max(self.profile["high_order_max_events"], float(n))
        times = np.sort(rng.uniform(0.0, dt, size=n))
        local_channels = rng.choice(self.residual.size, size=n, p=p_channel)
        metals = self.residual[local_channels]
        prev = 0.0

        full = self.space.propagate_full(source, float(times[0]))
        j = int(metals[0])
        qj = float(p_channel[local_channels[0]])
        fj = self.model.couplings[j]
        active = np.zeros((1, self.space.L + 1, self.model.nbasis), dtype=np.complex128)
        active[0, 0] = (-1j * np.conjugate(fj) / (lam * qj)) * full[j + 1]
        far = ((-1j * fj / (lam * qj)) * full[0])[None, :]
        current_j = j
        prev = float(times[0])

        for event_idx in range(1, n):
            tau = float(times[event_idx])
            active, far = self.space.propagate_compressed(active, far, np.asarray([current_j]), tau - prev)
            j = int(metals[event_idx])
            qj = float(p_channel[local_channels[event_idx]])
            fj = self.model.couplings[j]
            d_amp = active[0, 0].copy()
            far_in = far[0] if j == current_j else np.zeros(self.model.nbasis, dtype=np.complex128)
            new_active = np.zeros_like(active)
            new_active[0, 0] = (-1j * np.conjugate(fj) / (lam * qj)) * far_in
            far = ((-1j * fj / (lam * qj)) * d_amp)[None, :]
            active = new_active
            current_j = j
            prev = tau

        active, far = self.space.propagate_compressed(active, far, np.asarray([current_j]), dt - prev)
        out = np.zeros_like(source)
        out[self.space.active_electronic] = active[0]
        out[current_j + 1] += far[0]
        return out

    def _high_order_batch(
        self,
        sources: np.ndarray,
        dt: float,
        p_channel: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _profile_start = time.perf_counter()
        B = sources.shape[0]
        M = int(self.settings.high_order_samples)
        out = np.zeros_like(sources)
        if M <= 0 or self.residual.size == 0 or self.settings.high_order_mu <= 0.0:
            self.profile["high_order_s"] += time.perf_counter() - _profile_start
            return out
        weight = _poisson_tail_ge2(self.settings.high_order_mu) / M
        for b in range(B):
            acc = np.zeros_like(sources[b])
            for _ in range(M):
                if str(self.settings.high_order_channel_mode).lower() == "conditional_star":
                    acc += self._one_high_order_path_conditional_star(sources[b], dt, p_channel, rng)
                else:
                    acc += self._one_high_order_path(sources[b], dt, p_channel, rng)
            out[b] = weight * acc
        self.profile["high_order_s"] += time.perf_counter() - _profile_start
        return out


    def _apply_sampled_residual_unitary_batch(
        self,
        psi_half: np.ndarray,
        p_channel: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply a Hermitian Horvitz-Thompson residual Hamiltonian.

        For each replica, H_hat = sum_s H_{J_s}/(M q_{J_s}). Therefore
        E[H_hat] = H_res and the complete first-order term is unbiased. The
        exponential of every sampled H_hat is unitary, preventing path-norm
        explosion. Higher orders have a controllable finite-M bias.
        """
        B = psi_half.shape[0]
        M = self.first_order_sample_count()
        if M <= 0 or self.residual.size == 0:
            return psi_half.copy()
        out = psi_half.copy()
        for b in range(B):
            local = systematic_sample(p_channel, M, rng)
            counts = np.bincount(local, minlength=self.residual.size)
            selected_local = np.nonzero(counts)[0]
            selected_metal = self.residual[selected_local]
            weights = counts[selected_local] / (M * p_channel[selected_local])
            couplings = self.model.couplings[selected_metal] * weights
            gnorm = float(np.sqrt(np.sum(np.abs(couplings) ** 2).real))
            if gnorm <= 1.0e-15:
                continue
            d_amp = out[b, 0].copy()
            metal_amp = out[b, selected_metal + 1].copy()
            bright = np.sum(couplings[:, None] * metal_amp, axis=0) / gnorm
            theta = gnorm * dt
            ctheta = math.cos(theta)
            stheta = math.sin(theta)
            d_new = ctheta * d_amp - 1j * stheta * bright
            bright_new = ctheta * bright - 1j * stheta * d_amp
            metal_amp += ((bright_new - bright)[None, :] * np.conjugate(couplings)[:, None] / gnorm)
            out[b, 0] = d_new
            out[b, selected_metal + 1] = metal_amp
        return out


    def _apply_sampled_residual_random_product_batch(
        self,
        psi_half: np.ndarray,
        p_channel: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Second-order randomized product formula for the residual star terms.

        Each sampled channel j represents H_j/(M p_j), so the first-order
        generator is unbiased.  Applying the selected two-level rotations in
        a palindromic random order avoids the large collective-coupling bias
        of exponentiating one Horvitz--Thompson bright Hamiltonian.  Every
        realization remains exactly unitary.
        """
        B = psi_half.shape[0]
        M = self.first_order_sample_count()
        if M <= 0 or self.residual.size == 0:
            return psi_half.copy()
        out = psi_half.copy()
        for b in range(B):
            local = systematic_sample(p_channel, M, rng)
            counts = np.bincount(local, minlength=self.residual.size)
            selected_local = np.nonzero(counts)[0]
            if selected_local.size == 0:
                continue
            rng.shuffle(selected_local)
            durations = dt * counts[selected_local] / (M * p_channel[selected_local])

            def rotate(local_idx: int, duration: float) -> None:
                j = int(self.residual[local_idx])
                f = complex(self.model.couplings[j])
                af = abs(f)
                if af <= 1.0e-16 or abs(duration) <= 1.0e-16:
                    return
                theta = af * duration
                ct = math.cos(theta)
                st = math.sin(theta)
                phase = f / af
                d0 = out[b, 0].copy()
                m0 = out[b, j + 1].copy()
                out[b, 0] = ct * d0 - 1j * st * phase * m0
                out[b, j + 1] = ct * m0 - 1j * st * np.conjugate(phase) * d0

            # Symmetric product: half sweep forward, half sweep backward.
            for loc, dur in zip(selected_local, durations):
                rotate(int(loc), 0.5 * float(dur))
            for loc, dur in zip(selected_local[::-1], durations[::-1]):
                rotate(int(loc), 0.5 * float(dur))
        return out

    def _propagate_group_block_unitary(self, psi: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        B = int(self.settings.replicas_per_group)
        sources = np.repeat(psi[None, :, :], B, axis=0)
        p = self.channel_probabilities(psi, dt)
        half = self.space.propagate_full(sources, 0.5 * dt)
        mode = str(self.settings.propagation_mode).lower()
        if mode == "unitary_random_product":
            sampled = self._apply_sampled_residual_random_product_batch(half, p, dt, rng)
        else:
            sampled = self._apply_sampled_residual_unitary_batch(half, p, dt, rng)
        completed = self.space.propagate_full(sampled, 0.5 * dt)
        return np.mean(completed, axis=0)

    def _propagate_particle_group_unitary(self, particles: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        density_proxy = np.sqrt(np.mean(np.abs(particles) ** 2, axis=0)).astype(np.complex128)
        p = self.channel_probabilities(density_proxy, dt)
        half = self.space.propagate_full(particles, 0.5 * dt)
        mode = str(self.settings.propagation_mode).lower()
        if mode == "unitary_random_product_ensemble":
            sampled = self._apply_sampled_residual_random_product_batch(half, p, dt, rng)
        else:
            sampled = self._apply_sampled_residual_unitary_batch(half, p, dt, rng)
        return self.space.propagate_full(sampled, 0.5 * dt)

    def propagate_group_block(self, psi: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        mode = str(self.settings.propagation_mode).lower()
        if mode in {"unitary_sampled_hamiltonian", "unitary_random_product"}:
            return self._propagate_group_block_unitary(psi, dt, rng)
        if mode != "strict_dyson_poisson":
            raise ValueError(f"Unknown propagation_mode={self.settings.propagation_mode!r}")
        B = int(self.settings.replicas_per_group)
        sources = np.repeat(psi[None, :, :], B, axis=0)
        _t = time.perf_counter()
        p = self.channel_probabilities(psi, dt)
        self.profile["channel_probability_s"] += time.perf_counter() - _t
        _t = time.perf_counter()
        psi0 = self.space.propagate_full(sources, dt)
        self.profile["zero_order_s"] += time.perf_counter() - _t
        psi1 = self._first_order_batch(sources, dt, p, rng)
        psih = self._high_order_batch(sources, dt, p, rng)
        # Linear common-basis annihilation and unbiased re-expansion.
        return np.mean(psi0 + psi1 + psih, axis=0)

    def run(self) -> Dict[str, Any]:
        s = self.settings
        tmax = s.tmax_fs * FS_TO_AU
        block = s.block_fs * FS_TO_AU
        output = s.output_fs * FS_TO_AU
        mode = str(s.propagation_mode).lower()
        trajectory_mode = mode in {"unitary_trajectory_ensemble", "unitary_random_product_ensemble"}
        if trajectory_mode:
            particles = np.repeat(
                self.model.psi0[None, None, :, :],
                s.groups * s.replicas_per_group,
                axis=0,
            ).reshape((s.groups, s.replicas_per_group, self.model.nelectronic, self.model.nbasis))
            times_au: List[float] = [0.0]
            _tobs = time.perf_counter()
            records: List[Dict[str, float]] = [trajectory_ensemble_observables(particles, self.model)]
            self.profile["observables_s"] += time.perf_counter() - _tobs
        else:
            groups = np.repeat(self.model.psi0[None, :, :], s.groups, axis=0)
            times_au = [0.0]
            _tobs = time.perf_counter()
            records = [group_observables(groups, self.model)]
            self.profile["observables_s"] += time.perf_counter() - _tobs
        elapsed_records: List[float] = [0.0]
        start_clock = time.perf_counter()
        t = 0.0
        next_output = output
        nblocks = 0
        while t < tmax - 1.0e-12:
            dt = min(block, tmax - t)
            if trajectory_mode:
                for g in range(s.groups):
                    particles[g] = self._propagate_particle_group_unitary(particles[g], dt, self.rngs[g])
            else:
                for g in range(s.groups):
                    groups[g] = self.propagate_group_block(groups[g], dt, self.rngs[g])
            t += dt
            if tmax - t < 1.0e-8:
                t = tmax
            nblocks += 1
            if os.environ.get("NA_PROGRESS", "0") == "1" and nblocks % 100 == 0:
                print(f"[progress] blocks={nblocks} time_fs={t * AU_TO_FS:.3f} elapsed_s={time.perf_counter()-start_clock:.3f}", flush=True)
            should_record = (t + 1.0e-8 >= next_output) or (t >= tmax - 1.0e-8)
            if should_record and abs(t - times_au[-1]) > 1.0e-8:
                _tobs = time.perf_counter()
                obs = trajectory_ensemble_observables(particles, self.model) if trajectory_mode else group_observables(groups, self.model)
                self.profile["observables_s"] += time.perf_counter() - _tobs
                times_au.append(t)
                records.append(obs)
                elapsed_records.append(time.perf_counter() - start_clock)
                while next_output <= t + 1.0e-8:
                    next_output += output
        result: Dict[str, Any] = {
            "times_fs": np.asarray(times_au) * AU_TO_FS,
            "elapsed_s": np.asarray(elapsed_records),
            "active_metal_indices": self.space.active_metal.copy(),
            "active_metal_energies_ev": self.model.energies[self.space.active_metal] * HARTREE_TO_EV,
            "residual_count": int(self.residual.size),
            "nblocks": nblocks,
            "chebyshev_max_order": int(self.space.propagator.max_order_seen),
            "profile": {**self.profile,
                "high_order_mean_events_per_path": (self.profile["high_order_events"] / self.profile["high_order_paths"]) if self.profile["high_order_paths"] > 0 else 0.0,
                "unprofiled_loop_s": max(0.0, (time.perf_counter() - start_clock) - sum(self.profile[k] for k in ["channel_probability_s", "zero_order_s", "first_order_s", "high_order_s", "observables_s"])),
            },
            "settings": vars(s).copy(),
            "model": {
                "nmetal": self.model.nmetal,
                "nbasis": self.model.nbasis,
                "coupling_mode": self.model.coupling_mode,
                "metal_grid": self.model.metal_grid,
                "coupling_au": float(abs(self.model.couplings[0])),
                "gamma_au": self.model.params.gamma,
                "band_min_au": self.model.params.band_min,
                "band_max_au": self.model.params.band_max,
                "metal_spacing_au": self.model.metal_energy_spacing(),
                "benchmark_name": self.model.params.benchmark_name,
                "reorganization_ev": self.model.params.reorganization_ev,
                "displacement_alpha": self.model.params.displacement_alpha,
            },
        }
        for key in records[0]:
            result[key] = np.asarray([r[key] for r in records], dtype=float)
        if trajectory_mode:
            result["particle_states_final"] = particles
        else:
            result["group_states_final"] = groups
        return result


def _cross_expectation(groups: np.ndarray, operator_apply) -> float:
    G = groups.shape[0]
    if G < 2:
        vals = [np.vdot(g.ravel(), operator_apply(g).ravel()).real for g in groups]
        return float(np.mean(vals))
    total = 0.0 + 0.0j
    count = 0
    operated = [operator_apply(groups[h]) for h in range(G)]
    for g in range(G):
        for h in range(G):
            if g == h:
                continue
            total += np.vdot(groups[g].ravel(), operated[h].ravel())
            count += 1
    return float((total / count).real)


def group_observables(groups: np.ndarray, model: NewnsAndersonModel) -> Dict[str, float]:
    mean_psi = np.mean(groups, axis=0)
    plugin_norm = float(np.vdot(mean_psi.ravel(), mean_psi.ravel()).real)
    plugin_pop_d = float(np.vdot(mean_psi[0], mean_psi[0]).real)

    def identity(x: np.ndarray) -> np.ndarray:
        return x

    def proj_d(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        y[0] = x[0]
        return y

    def x_apply(x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        for e in range(model.nelectronic):
            y[e] = model.x_operator @ x[e]
        return y

    def x2_apply(x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        for e in range(model.nelectronic):
            tmp = model.x_operator @ x[e]
            y[e] = model.x_operator @ tmp
        return y

    # Cross-group estimators are unbiased for the norm and observable
    # numerators separately.  Physical populations and moments are ratios of
    # these estimators.  Keep the raw numerators in the output for diagnostics,
    # but report normalized observables under the established public keys.
    norm_u = _cross_expectation(groups, identity)
    pop_d_num_u = _cross_expectation(groups, proj_d)
    x_num_u = _cross_expectation(groups, x_apply)
    x2_num_u = _cross_expectation(groups, x2_apply)
    safe_norm_u = norm_u if abs(norm_u) > 1.0e-14 else 1.0
    pop_d_u = pop_d_num_u / safe_norm_u
    x_u = x_num_u / safe_norm_u
    x2_u = x2_num_u / safe_norm_u
    width2 = x2_u - x_u * x_u
    plugin_safe_norm = plugin_norm if plugin_norm > 1.0e-14 else 1.0
    plugin_pop_d_normalized = plugin_pop_d / plugin_safe_norm
    individual_norms = np.sum(np.abs(groups) ** 2, axis=(1, 2)).real
    individual_pop = np.sum(np.abs(groups[:, 0, :]) ** 2, axis=1).real
    top = min(8, model.nbasis)
    boundary = float(np.max(np.sum(np.abs(groups[:, :, -top:]) ** 2, axis=(1, 2)).real))
    return {
        "norm_u": norm_u,
        "population_d_u": pop_d_u,
        "q_mean_bohr_u": x_u,
        "q_width_bohr_u": math.sqrt(max(0.0, width2)),
        "population_d_numerator_u": pop_d_num_u,
        "q_mean_numerator_bohr_u": x_num_u,
        "q_x2_numerator_bohr2_u": x2_num_u,
        "norm_plugin": plugin_norm,
        "population_d_plugin": plugin_pop_d_normalized,
        "population_d_numerator_plugin": plugin_pop_d,
        "group_norm_std": float(np.std(individual_norms, ddof=1)) if groups.shape[0] > 1 else 0.0,
        "group_population_std": float(np.std(individual_pop, ddof=1)) if groups.shape[0] > 1 else 0.0,
        "boundary_population_top8": boundary,
    }


def trajectory_ensemble_observables(particles: np.ndarray, model: NewnsAndersonModel) -> Dict[str, float]:
    # Vectorized harmonic-oscillator moments; avoids hundreds of thousands of
    # small sparse matvec calls for large electronic spaces.
    G, R, _, K = particles.shape
    abs2 = np.abs(particles) ** 2
    norms = np.sum(abs2, axis=(2, 3)).real
    pops = np.sum(abs2[:, :, 0, :], axis=2).real
    pref = 1.0 / math.sqrt(2.0 * model.params.mass * model.params.omega)
    n1 = np.sqrt(np.arange(1, K, dtype=float))
    cross1 = np.sum(np.conjugate(particles[..., :-1]) * particles[..., 1:] * n1, axis=(2, 3))
    xs = 2.0 * pref * cross1.real
    ndiag = 2.0 * np.arange(K, dtype=float) + 1.0
    x2_diag = np.sum(abs2 * ndiag, axis=(2, 3)).real
    if K >= 3:
        n2 = np.sqrt(np.arange(1, K - 1, dtype=float) * np.arange(2, K, dtype=float))
        cross2 = np.sum(np.conjugate(particles[..., :-2]) * particles[..., 2:] * n2, axis=(2, 3))
        x2s = pref * pref * (x2_diag + 2.0 * cross2.real)
    else:
        x2s = pref * pref * x2_diag
    group_norm = np.mean(norms, axis=1)
    group_pop = np.mean(pops, axis=1)
    group_x = np.mean(xs, axis=1)
    group_x2 = np.mean(x2s, axis=1)
    norm = float(np.mean(group_norm))
    pop = float(np.mean(group_pop))
    x = float(np.mean(group_x))
    x2 = float(np.mean(group_x2))
    top = min(8, model.nbasis)
    boundary = float(np.max(np.sum(abs2[..., -top:], axis=(2, 3)).real))
    return {
        "norm_u": norm,
        "population_d_u": pop,
        "q_mean_bohr_u": x,
        "q_width_bohr_u": math.sqrt(max(0.0, x2 - x * x)),
        "norm_plugin": norm,
        "population_d_plugin": pop,
        "group_norm_std": float(np.std(group_norm, ddof=1)) if G > 1 else 0.0,
        "group_population_std": float(np.std(group_pop, ddof=1)) if G > 1 else 0.0,
        "boundary_population_top8": boundary,
    }


def state_observables(psi: np.ndarray, model: NewnsAndersonModel) -> Dict[str, float]:
    K = model.nbasis
    abs2 = np.abs(psi) ** 2
    norm = float(np.sum(abs2).real)
    pop_d = float(np.sum(abs2[0]).real)
    pref = 1.0 / math.sqrt(2.0 * model.params.mass * model.params.omega)
    n1 = np.sqrt(np.arange(1, K, dtype=float))
    x_num = float(2.0 * pref * np.sum(np.conjugate(psi[:, :-1]) * psi[:, 1:] * n1).real)
    ndiag = 2.0 * np.arange(K, dtype=float) + 1.0
    x2_num = pref * pref * float(np.sum(abs2 * ndiag).real)
    if K >= 3:
        n2 = np.sqrt(np.arange(1, K - 1, dtype=float) * np.arange(2, K, dtype=float))
        x2_num += pref * pref * float(2.0 * np.sum(np.conjugate(psi[:, :-2]) * psi[:, 2:] * n2).real)
    top = min(8, model.nbasis)
    boundary = float(np.sum(abs2[:, -top:]).real)
    safe_norm = norm if norm > 1.0e-14 else 1.0
    pop_d_normalized = pop_d / safe_norm
    x = x_num / safe_norm
    x2 = x2_num / safe_norm
    return {
        "norm_exact": norm,
        "population_d_exact": pop_d_normalized,
        "population_d_numerator_exact": pop_d,
        "q_mean_bohr_exact": x,
        "q_width_bohr_exact": math.sqrt(max(0.0, x2 - x * x)),
        "q_mean_numerator_bohr_exact": x_num,
        "q_x2_numerator_bohr2_exact": x2_num,
        "boundary_population_top8_exact": boundary,
    }


def run_exact_reference(model: NewnsAndersonModel, tmax_fs: float, output_fs: float) -> Dict[str, Any]:
    """Deterministic sparse reference propagated in short Chebyshev intervals.

    A single long-time ``expm_multiply`` call can choose an unnecessarily large
    Krylov workload for this displaced-oscillator Hamiltonian.  Reusing one
    rigorously bounded Chebyshev propagator for each output interval is both
    deterministic and substantially faster while retaining machine-precision
    norm conservation.
    """
    h = model.build_full_hamiltonian()
    propagator = ChebyshevPropagator(h, tol=2.0e-12)
    nout = int(round(tmax_fs / output_fs)) + 1
    dt = output_fs * FS_TO_AU
    psi = model.psi0.reshape((-1,)).copy()
    records = [state_observables(psi.reshape((model.nelectronic, model.nbasis)), model)]
    start = time.perf_counter()
    for _ in range(1, nout):
        psi = propagator.apply(psi, dt)
        records.append(state_observables(psi.reshape((model.nelectronic, model.nbasis)), model))
    elapsed = time.perf_counter() - start
    out: Dict[str, Any] = {
        "times_fs": np.linspace(0.0, tmax_fs, nout),
        "runtime_s": elapsed,
        "nnz": int(h.nnz),
        "dimension": int(h.shape[0]),
        "chebyshev_max_order": int(propagator.max_order_seen),
    }
    for key in records[0]:
        out[key] = np.asarray([r[key] for r in records], dtype=float)
    return out


def save_timeseries_csv(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == data["times_fs"].size]
    if "times_fs" in keys:
        keys.remove("times_fs")
    keys = ["times_fs"] + sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(data["times_fs"].size):
            w.writerow([f"{float(data[k][i]):.12g}" for k in keys])


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 100:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items() if k not in {"group_states_final", "particle_states_final"}}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def save_metadata(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(data), f, indent=2, ensure_ascii=False)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_from_config(cfg: Dict[str, Any]) -> NewnsAndersonModel:
    mc = cfg["model"]
    params = NewnsAndersonParameters(**mc.get("parameters", {}))
    return NewnsAndersonModel(
        nmetal=int(mc["nmetal"]),
        nbasis=int(mc.get("nbasis", 224)),
        params=params,
        coupling_mode=str(mc.get("coupling_mode", "paper")),
        fixed_hybridization_reference_n=int(mc.get("fixed_hybridization_reference_n", 20)),
        metal_grid=str(mc.get("metal_grid", "legacy")),
    )


def settings_from_config(cfg: Dict[str, Any]) -> HybridSettings:
    return HybridSettings(**cfg.get("hybrid", {}))


def run_single_config(cfg: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    model = model_from_config(cfg)
    settings = settings_from_config(cfg)
    sim = HybridNewnsAndersonSimulator(model, settings)
    hybrid = sim.run()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_timeseries_csv(output_dir / "hybrid.csv", hybrid)
    save_metadata(output_dir / "hybrid_metadata.json", hybrid)
    # Final particle/group tensors are not needed for diagnostics and can retain
    # a large fragmented allocation before the deterministic reference starts.
    hybrid.pop("particle_states_final", None)
    hybrid.pop("group_states_final", None)
    gc.collect()
    result: Dict[str, Any] = {"hybrid": hybrid}
    if bool(cfg.get("reference", {}).get("enabled", False)):
        ref = run_exact_reference(model, settings.tmax_fs, settings.output_fs)
        save_timeseries_csv(output_dir / "exact.csv", ref)
        save_metadata(output_dir / "exact_metadata.json", ref)
        result["exact"] = ref
        # Common diagnostics.
        result["diagnostics"] = compare_hybrid_exact(hybrid, ref)
        save_metadata(output_dir / "diagnostics.json", result["diagnostics"])
    return result


def compare_hybrid_exact(hybrid: Dict[str, Any], exact: Dict[str, Any]) -> Dict[str, float]:
    th = np.asarray(hybrid["times_fs"], dtype=float)
    te = np.asarray(exact["times_fs"], dtype=float)

    def exact_on_hybrid(key: str) -> np.ndarray:
        return np.interp(th, te, np.asarray(exact[key], dtype=float))

    def rms(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    pop_exact = exact_on_hybrid("population_d_exact")
    norm_exact = exact_on_hybrid("norm_exact")
    q_exact = exact_on_hybrid("q_mean_bohr_exact")
    qw_exact = exact_on_hybrid("q_width_bohr_exact")
    return {
        "population_rms_u": rms(hybrid["population_d_u"], pop_exact),
        "population_rms_plugin": rms(hybrid["population_d_plugin"], pop_exact),
        "q_mean_rms_bohr_u": rms(hybrid["q_mean_bohr_u"], q_exact),
        "q_width_rms_bohr_u": rms(hybrid["q_width_bohr_u"], qw_exact),
        "max_abs_norm_error_u": float(np.max(np.abs(hybrid["norm_u"] - norm_exact))),
        "max_abs_population_error_u": float(np.max(np.abs(hybrid["population_d_u"] - pop_exact))),
        "final_abs_population_error_u": float(abs(hybrid["population_d_u"][-1] - pop_exact[-1])),
        "hybrid_runtime_s": float(hybrid["elapsed_s"][-1]),
        "exact_runtime_s": float(exact["runtime_s"]),
    }


def run_scaling(cfg: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Run fixed-budget state-count scaling with restartable checkpoints."""
    sc = cfg["scaling"]
    n_values = [int(x) for x in sc["n_values"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def fit_runtime(minimum_n: Optional[int] = None) -> Tuple[float, float]:
        selected = [r for r in rows if minimum_n is None or int(r["nmetal"]) >= minimum_n]
        if len(selected) < 2:
            return float("nan"), float("nan")
        x = np.log(np.asarray([r["nmetal"] for r in selected], dtype=float))
        y = np.log(np.asarray([r["runtime_reported_s"] for r in selected], dtype=float))
        exponent, intercept = np.polyfit(x, y, 1)
        return float(exponent), float(math.exp(intercept))

    def write_checkpoint() -> Dict[str, Any]:
        if not rows:
            return {"rows": []}
        with (output_dir / "scaling.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        exponent, prefactor = fit_runtime()
        tail_min = 400 if sum(int(r["nmetal"]) >= 400 for r in rows) >= 2 else None
        tail_exp, tail_pref = fit_runtime(tail_min)
        summary = {
            "rows": rows,
            "runtime_scaling_exponent": exponent,
            "runtime_prefactor": prefactor,
            "large_n_minimum": tail_min,
            "large_n_runtime_scaling_exponent": tail_exp,
            "large_n_runtime_prefactor": tail_pref,
            "completed_n_values": [int(r["nmetal"]) for r in rows],
        }
        save_metadata(output_dir / "scaling_summary.json", summary)
        return summary

    for n in n_values:
        local = json.loads(json.dumps(cfg))
        local["model"]["nmetal"] = n
        local["hybrid"].update(sc.get("hybrid_overrides", {}))
        model = model_from_config(local)
        settings = settings_from_config(local)
        simulator = HybridNewnsAndersonSimulator(model, settings)
        wall_start = time.perf_counter()
        result = simulator.run()
        process_wall = time.perf_counter() - wall_start
        row = {
            "nmetal": n,
            "dimension": (n + 1) * model.nbasis,
            "runtime_s": process_wall,
            "runtime_reported_s": float(result["elapsed_s"][-1]),
            "final_norm_u": float(result["norm_u"][-1]),
            "max_abs_norm_error": float(np.max(np.abs(result["norm_u"] - 1.0))),
            "final_population_d_u": float(result["population_d_u"][-1]),
            "max_group_population_std": float(np.max(result["group_population_std"])),
            "boundary_population_top8_max": float(np.max(result["boundary_population_top8"])),
            "chebyshev_max_order": int(result["chebyshev_max_order"]),
            "coupling_au": float(abs(model.couplings[0])),
        }
        rows.append(row)
        n_dir = output_dir / f"N{n}"
        save_timeseries_csv(n_dir / "hybrid.csv", result)
        save_metadata(n_dir / "metadata.json", result)
        write_checkpoint()
        result.pop("particle_states_final", None)
        result.pop("group_states_final", None)
        del result, simulator, model, local
        gc.collect()
    return write_checkpoint()


def collect_scaling_results(cfg: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Collect independently executed N cases into one scaling summary."""
    n_values = [int(x) for x in cfg["scaling"]["n_values"]]
    rows: List[Dict[str, Any]] = []
    for n in n_values:
        n_dir = output_dir / f"N{n}"
        csv_path = n_dir / "hybrid.csv"
        meta_path = n_dir / "hybrid_metadata.json"
        if not meta_path.exists():
            meta_path = n_dir / "metadata.json"
        if not csv_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing completed scaling case N={n}: {n_dir}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            data_rows = list(csv.DictReader(f))
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        final = data_rows[-1]
        norm_values = np.asarray([float(r["norm_u"]) for r in data_rows])
        group_std_values = np.asarray([float(r["group_population_std"]) for r in data_rows])
        boundary_values = np.asarray([float(r["boundary_population_top8"]) for r in data_rows])
        runtime = float(final["elapsed_s"])
        rows.append({
            "nmetal": n,
            "dimension": (n + 1) * int(metadata["model"]["nbasis"]),
            "runtime_s": runtime,
            "runtime_reported_s": runtime,
            "final_norm_u": float(final["norm_u"]),
            "max_abs_norm_error": float(np.max(np.abs(norm_values - 1.0))),
            "final_population_d_u": float(final["population_d_u"]),
            "max_group_population_std": float(np.max(group_std_values)),
            "boundary_population_top8_max": float(np.max(boundary_values)),
            "chebyshev_max_order": int(metadata["chebyshev_max_order"]),
            "coupling_au": float(metadata["model"]["coupling_au"]),
        })
    x = np.log(np.asarray([r["nmetal"] for r in rows], dtype=float))
    y = np.log(np.asarray([r["runtime_reported_s"] for r in rows], dtype=float))
    exponent, intercept = np.polyfit(x, y, 1)
    tail_rows = [r for r in rows if int(r["nmetal"]) >= 400]
    xt = np.log(np.asarray([r["nmetal"] for r in tail_rows], dtype=float))
    yt = np.log(np.asarray([r["runtime_reported_s"] for r in tail_rows], dtype=float))
    tail_exp, tail_intercept = np.polyfit(xt, yt, 1)
    summary = {
        "rows": rows,
        "runtime_scaling_exponent": float(exponent),
        "runtime_prefactor": float(math.exp(intercept)),
        "large_n_minimum": 400,
        "large_n_runtime_scaling_exponent": float(tail_exp),
        "large_n_runtime_prefactor": float(math.exp(tail_intercept)),
        "completed_n_values": n_values,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scaling.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_metadata(output_dir / "scaling_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Active-L exact + residual stochastic Newns-Anderson dynamics")
    parser.add_argument("config", help="YAML configuration")
    parser.add_argument("--mode", choices=["single", "scaling", "collect-scaling"], default="single")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--nmetal", type=int, default=None, help="Override model.nmetal for one isolated case")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.nmetal is not None:
        cfg["model"]["nmetal"] = int(args.nmetal)
    out = Path(args.output or cfg.get("output", {}).get("directory", "results/run")).resolve()
    if args.mode == "single":
        result = run_single_config(cfg, out)
        print(json.dumps(_jsonify({"output": str(out), "diagnostics": result.get("diagnostics", {})}), indent=2, ensure_ascii=False))
    elif args.mode == "scaling":
        summary = run_scaling(cfg, out)
        print(json.dumps(_jsonify({"output": str(out), **summary}), indent=2, ensure_ascii=False))
    else:
        summary = collect_scaling_results(cfg, out)
        print(json.dumps(_jsonify({"output": str(out), **summary}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
