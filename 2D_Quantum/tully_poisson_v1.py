import csv
import math
import os
import socket
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.fft as _scipy_fft
except Exception:  # pragma: no cover - optional backend
    _scipy_fft = None


@dataclass
class Grid:
    ndim: int
    x: np.ndarray
    dx: float
    kx: np.ndarray
    mx: float
    y: Optional[np.ndarray] = None
    dy: Optional[float] = None
    ky: Optional[np.ndarray] = None
    my: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.ndim == 1:
            return (self.x.size,)
        assert self.y is not None
        return (self.x.size, self.y.size)

    @property
    def measure(self) -> float:
        if self.ndim == 1:
            return self.dx
        assert self.dy is not None
        return self.dx * self.dy

    @property
    def num_points(self) -> int:
        out = 1
        for n in self.shape:
            out *= int(n)
        return out

    def mesh(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.ndim == 1:
            return self.x, None
        assert self.y is not None
        xx, yy = np.meshgrid(self.x, self.y, indexing='ij')
        return xx, yy


@dataclass
class EdgeCoupling:
    i: int
    j: int
    values: np.ndarray
    name: str


@dataclass
class RadialRegion:
    name: str
    center: Tuple[float, float]
    r_min: float
    r_max: float


@dataclass
class MultiStatePotentials:
    nstates: int
    diag: np.ndarray  # (M, *shape)
    edges: List[EdgeCoupling]
    full_matrix: np.ndarray  # (*shape, M, M)
    min_gap: np.ndarray  # shape


DEFAULT_TULLY_PARAMS: Dict[str, Dict[str, float]] = {
    'tully1': {'A': 0.01, 'B': 1.6, 'C': 0.005, 'D': 1.0},
    'tully2': {'A': 0.10, 'B': 0.28, 'C': 0.015, 'D': 0.06, 'E0': 0.05},
    'tully3': {'A': 0.0006, 'B': 0.10, 'C': 0.90},
}

_FFT_BACKEND = 'scipy' if _scipy_fft is not None else 'numpy'
_PARALLEL_STATE: Dict[str, Any] = {}

_STATUS_ONCE = {'main': False, 'hi_pool': False, 'replica_pool': False, 'cap': False}
_WORKER_SEEN: set = set()


def _configure_stdio_inplace() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


def _ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _print_runtime_status(tag: str, extra: str = '') -> None:
    _configure_stdio_inplace()
    host = socket.gethostname()
    pid = os.getpid()
    ppid = os.getppid()
    proc = mp.current_process()
    try:
        start_method = mp.get_start_method(allow_none=True)
    except Exception:
        start_method = 'unknown'
    omp = os.environ.get('OMP_NUM_THREADS', 'unset')
    mkl = os.environ.get('MKL_NUM_THREADS', 'unset')
    openblas = os.environ.get('OPENBLAS_NUM_THREADS', 'unset')
    msg = (
        f'[{_ts()}] [{tag}] host={host} pid={pid} ppid={ppid} '
        f'proc_name={proc.name} daemon={proc.daemon} start_method={start_method} '
        f'python={sys.executable} OMP_NUM_THREADS={omp} MKL_NUM_THREADS={mkl} OPENBLAS_NUM_THREADS={openblas}'
    )
    if extra:
        msg += f' | {extra}'
    print(msg, flush=True)
try:
    _FORK_CTX = mp.get_context('fork')
except ValueError:  # pragma: no cover - non-POSIX fallback
    _FORK_CTX = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _initial_fft_workers() -> int:
    return max(1, _env_int('POISSON_V1_FFT_WORKERS', _cpu_count()))


_FFT_WORKERS = _initial_fft_workers()


def _set_fft_workers(nworkers: int) -> None:
    global _FFT_WORKERS
    _FFT_WORKERS = max(1, int(nworkers))


def _parallel_child_init() -> None:
    _set_fft_workers(1)
    _print_runtime_status('CHILD_INIT', extra='FFT_WORKERS forced to 1 in child process')


def _log_cap_status(cap: Optional[np.ndarray]) -> None:
    if _STATUS_ONCE.get('cap', False):
        return
    if cap is None:
        _print_runtime_status('CAP', extra='disabled (norm-preserving boundary mode)')
    else:
        carr = np.asarray(cap, dtype=float)
        _print_runtime_status('CAP', extra=f'enabled shape={carr.shape} max={float(np.max(carr)):.6e} min={float(np.min(carr)):.6e}')
    _STATUS_ONCE['cap'] = True


def _requested_parallel_workers(kind: str, task_count: int) -> int:
    specific = f'POISSON_V1_{kind.upper()}_WORKERS'
    requested = _env_int(specific, _env_int('POISSON_V1_WORKERS', _cpu_count()))
    if requested <= 0:
        requested = _cpu_count()
    return max(1, min(int(requested), _cpu_count(), int(task_count)))


def _make_process_pool(kind: str, task_count: int) -> Tuple[Optional[ProcessPoolExecutor], int]:
    if task_count <= 1 or _FORK_CTX is None:
        return None, 1
    workers = _requested_parallel_workers(kind, task_count)
    if workers <= 1:
        return None, 1
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=_FORK_CTX, initializer=_parallel_child_init)
    return ex, workers


def _fft_1d(arr: np.ndarray, axis: int) -> np.ndarray:
    if _FFT_BACKEND == 'scipy':
        return _scipy_fft.fft(arr, axis=axis, workers=_FFT_WORKERS)
    return np.fft.fft(arr, axis=axis)


def _ifft_1d(arr: np.ndarray, axis: int) -> np.ndarray:
    if _FFT_BACKEND == 'scipy':
        return _scipy_fft.ifft(arr, axis=axis, workers=_FFT_WORKERS)
    return np.fft.ifft(arr, axis=axis)


def _fftn_2d(arr: np.ndarray, axes: Tuple[int, int]) -> np.ndarray:
    if _FFT_BACKEND == 'scipy':
        return _scipy_fft.fftn(arr, axes=axes, workers=_FFT_WORKERS)
    return np.fft.fftn(arr, axes=axes)


def _ifftn_2d(arr: np.ndarray, axes: Tuple[int, int]) -> np.ndarray:
    if _FFT_BACKEND == 'scipy':
        return _scipy_fft.ifftn(arr, axes=axes, workers=_FFT_WORKERS)
    return np.fft.ifftn(arr, axes=axes)


# ------------------------------
# Utilities
# ------------------------------

def _resolve_path(path: str, config: Optional[Dict[str, Any]] = None) -> str:
    if os.path.isabs(path):
        return path
    cfg_dir = None
    if config is not None:
        cfg_dir = config.get('_config_dir')
    if cfg_dir:
        return os.path.abspath(os.path.join(cfg_dir, path))
    return os.path.abspath(path)


def build_grid(grid_cfg: Dict[str, Any]) -> Grid:
    xmin = float(grid_cfg['xmin'])
    xmax = float(grid_cfg['xmax'])
    nx = int(grid_cfg['nx'])
    x = np.linspace(xmin, xmax, nx, endpoint=False)
    dx = float(x[1] - x[0])
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    if 'ymin' not in grid_cfg and 'ymax' not in grid_cfg and 'ny' not in grid_cfg:
        mass = float(grid_cfg.get('mass', grid_cfg.get('mx', 1.0)))
        return Grid(ndim=1, x=x, dx=dx, kx=kx, mx=mass)

    ymin = float(grid_cfg['ymin'])
    ymax = float(grid_cfg['ymax'])
    ny = int(grid_cfg['ny'])
    y = np.linspace(ymin, ymax, ny, endpoint=False)
    dy = float(y[1] - y[0])
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    mx = float(grid_cfg.get('mx', grid_cfg.get('mass', 1.0)))
    my = float(grid_cfg.get('my', grid_cfg.get('mass', mx)))
    return Grid(ndim=2, x=x, dx=dx, kx=kx, mx=mx, y=y, dy=dy, ky=ky, my=my)


def _dense_from_diag_and_edges(diag: np.ndarray, edges: Sequence[EdgeCoupling]) -> np.ndarray:
    nstates = diag.shape[0]
    spatial_shape = tuple(diag.shape[1:])
    mats = np.zeros(spatial_shape + (nstates, nstates), dtype=np.complex128)
    for s in range(nstates):
        mats[..., s, s] = diag[s]
    for e in edges:
        mats[..., e.i, e.j] = e.values
        mats[..., e.j, e.i] = np.conjugate(e.values)
    return mats


def _compute_min_gap(full_matrix: np.ndarray) -> np.ndarray:
    evals = np.linalg.eigvalsh(full_matrix)
    if evals.shape[-1] < 2:
        return np.zeros(evals.shape[:-1], dtype=float)
    gaps = np.diff(evals, axis=-1)
    return np.min(np.abs(gaps), axis=-1)


def _parse_edge_name(i: int, j: int) -> str:
    return f'V{i+1}{j+1}'


def _parse_pair(item: Any) -> Tuple[int, int]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        a, b = int(item[0]), int(item[1])
        return min(a, b), max(a, b)
    raise ValueError(f'Base-edge pair must be length-2 list/tuple, got: {item}')


def _base_edge_set(config: Dict[str, Any]) -> set[Tuple[int, int]]:
    model_cfg = config.get('model', {})
    algo_cfg = config.get('algorithm', {})
    raw = algo_cfg.get('base_edges', model_cfg.get('base_edges', []))
    return {_parse_pair(it) for it in raw}


def _spatial_shape_from_grid(grid: Grid) -> Tuple[int, ...]:
    return grid.shape


def _reshape_psi_to_points(psi: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    nstates = psi.shape[0]
    spatial_shape = tuple(psi.shape[1:])
    npts = int(np.prod(spatial_shape))
    flat = np.moveaxis(psi, 0, -1).reshape(npts, nstates)
    return flat, spatial_shape, nstates


def _restore_psi_from_points(flat: np.ndarray, spatial_shape: Tuple[int, ...]) -> np.ndarray:
    npts, nstates = flat.shape
    _ = npts
    arr = flat.reshape(spatial_shape + (nstates,))
    return np.moveaxis(arr, -1, 0)


def _channel_config(config: Dict[str, Any]) -> Dict[str, Any]:
    channels = dict(config.get('channels', {}))
    analysis = config.get('analysis', {})
    if 'x_divider' not in channels and 'x_divider' in analysis:
        channels['x_divider'] = analysis['x_divider']
    if 'radial_regions' not in channels and 'radial_regions' in analysis:
        channels['radial_regions'] = analysis['radial_regions']
    return channels


def _parse_radial_regions(config: Dict[str, Any]) -> List[RadialRegion]:
    channels = _channel_config(config)
    raw = channels.get('radial_regions', [])
    out: List[RadialRegion] = []
    for item in raw:
        center = item.get('center', [0.0, 0.0])
        if not (isinstance(center, (list, tuple)) and len(center) == 2):
            raise ValueError(f'Radial region center must be length-2 list, got {center}')
        out.append(
            RadialRegion(
                name=str(item['name']),
                center=(float(center[0]), float(center[1])),
                r_min=float(item.get('r_min', 0.0)),
                r_max=float(item['r_max']),
            )
        )
    return out


# ------------------------------
# Model builders
# ------------------------------

def tully_potentials(kind: str, x: np.ndarray, params: Optional[Dict[str, float]] = None) -> MultiStatePotentials:
    pars = dict(DEFAULT_TULLY_PARAMS[kind])
    if params:
        pars.update(params)

    if kind == 'tully1':
        A, B, C, D = pars['A'], pars['B'], pars['C'], pars['D']
        V11 = np.where(x >= 0.0, A * (1.0 - np.exp(-B * x)), -A * (1.0 - np.exp(B * x)))
        V22 = -V11
        V12 = C * np.exp(-D * x * x)
    elif kind == 'tully2':
        A, B, C, D, E0 = pars['A'], pars['B'], pars['C'], pars['D'], pars['E0']
        V11 = np.zeros_like(x)
        V22 = -A * np.exp(-B * x * x) + E0
        V12 = C * np.exp(-D * x * x)
    elif kind == 'tully3':
        A, B, C = pars['A'], pars['B'], pars['C']
        V11 = A * np.ones_like(x)
        V22 = -A * np.ones_like(x)
        V12 = np.where(x < 0.0, B * np.exp(C * x), B * (2.0 - np.exp(-C * x)))
    else:
        raise ValueError(f'Unsupported Tully model kind: {kind}')

    diag = np.vstack([V11, V22]).astype(np.complex128)
    edges = [EdgeCoupling(0, 1, np.asarray(V12, dtype=np.complex128), 'V12')]
    full = _dense_from_diag_and_edges(diag, edges)
    min_gap = _compute_min_gap(full)
    return MultiStatePotentials(nstates=2, diag=diag, edges=edges, full_matrix=full, min_gap=min_gap)


def _read_external_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'CSV file has no header: {path}')
        rows = list(reader)
    return reader.fieldnames, rows


def _load_external_multistate_csv(path: str, model_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[EdgeCoupling]]:
    fieldnames, rows = _read_external_csv(path)
    columns = model_cfg.get('columns', {})
    x_col = columns.get('x', 'x')
    nstates = int(model_cfg['nstates'])
    diag_cols = columns.get('diagonal')
    if diag_cols is None:
        diag_cols = [f'V{i+1}{i+1}' for i in range(nstates)]
    if len(diag_cols) != nstates:
        raise ValueError('columns.diagonal length must equal nstates')
    missing = [c for c in [x_col] + list(diag_cols) if c not in fieldnames]
    if missing:
        raise ValueError(f'External CSV is missing required columns {missing}. Available columns: {fieldnames}')

    x = np.asarray([float(r[x_col]) for r in rows], dtype=float)
    diag = np.vstack([np.asarray([float(r[c]) for r in rows], dtype=float) for c in diag_cols]).astype(np.complex128)

    edge_specs = columns.get('couplings')
    if edge_specs is None:
        edge_specs = []
        for i in range(nstates):
            for j in range(i + 1, nstates):
                nm = f'V{i+1}{j+1}'
                if nm in fieldnames:
                    edge_specs.append([i, j, nm])
    edges: List[EdgeCoupling] = []
    for spec in edge_specs:
        if not (isinstance(spec, (list, tuple)) and len(spec) == 3):
            raise ValueError(f'Each coupling spec must be [i, j, column_name], got: {spec}')
        i, j, col = int(spec[0]), int(spec[1]), str(spec[2])
        if col not in fieldnames:
            raise ValueError(f'Coupling column {col} not present in CSV. Available columns: {fieldnames}')
        vals = np.asarray([complex(r[col]) for r in rows], dtype=np.complex128)
        edges.append(EdgeCoupling(min(i, j), max(i, j), vals, _parse_edge_name(min(i, j), max(i, j))))
    return x, diag, edges


def _load_external_multistate_npz(path: str, model_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[EdgeCoupling]]:
    data = np.load(path)
    columns = model_cfg.get('columns', {})
    x_key = columns.get('x', 'x')
    nstates = int(model_cfg['nstates'])
    diag_keys = columns.get('diagonal')
    if diag_keys is None:
        diag_keys = [f'V{i+1}{i+1}' for i in range(nstates)]
    missing = [k for k in [x_key] + list(diag_keys) if k not in data]
    if missing:
        raise ValueError(f'External NPZ missing required arrays {missing}. Available arrays: {list(data.keys())}')
    x = np.asarray(data[x_key], dtype=float)
    diag = np.vstack([np.asarray(data[k], dtype=np.complex128) for k in diag_keys])
    edge_specs = columns.get('couplings')
    if edge_specs is None:
        edge_specs = []
        for i in range(nstates):
            for j in range(i + 1, nstates):
                nm = f'V{i+1}{j+1}'
                if nm in data:
                    edge_specs.append([i, j, nm])
    edges: List[EdgeCoupling] = []
    for spec in edge_specs:
        i, j, key = int(spec[0]), int(spec[1]), str(spec[2])
        if key not in data:
            raise ValueError(f'Coupling array {key} not present in NPZ. Available arrays: {list(data.keys())}')
        vals = np.asarray(data[key], dtype=np.complex128)
        edges.append(EdgeCoupling(min(i, j), max(i, j), vals, _parse_edge_name(min(i, j), max(i, j))))
    return x, diag, edges


def _validate_external_arrays(x_src: np.ndarray, diag_src: np.ndarray, edges: List[EdgeCoupling], path: str) -> Tuple[np.ndarray, np.ndarray, List[EdgeCoupling]]:
    if x_src.ndim != 1:
        raise ValueError(f'External x-grid must be 1D: {path}')
    if diag_src.ndim != 2:
        raise ValueError(f'External diagonal potentials must have shape (nstates, nx): {path}')
    _, nx = diag_src.shape
    if x_src.size != nx:
        raise ValueError(f'Length mismatch in {path}: len(x)={x_src.size}, diag nx={nx}')
    order = np.argsort(x_src)
    x = x_src[order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f'External x-grid must be strictly increasing after sorting: {path}')
    diag = np.asarray(diag_src[:, order], dtype=np.complex128)
    new_edges: List[EdgeCoupling] = []
    for e in edges:
        vals = np.asarray(e.values, dtype=np.complex128)
        if vals.ndim != 1 or vals.size != nx:
            raise ValueError(f'Edge {e.name} has wrong shape in {path}')
        new_edges.append(EdgeCoupling(e.i, e.j, vals[order], e.name))
    return x, diag, new_edges


def external_multistate_potentials(grid: Grid, model_cfg: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> MultiStatePotentials:
    path = _resolve_path(str(model_cfg['path']), config=config)
    if not os.path.exists(path):
        raise FileNotFoundError(f'External model file not found: {path}')
    fmt = str(model_cfg.get('format', 'auto')).lower()
    if fmt == 'auto':
        ext = os.path.splitext(path)[1].lower()
        fmt = 'csv' if ext in {'.csv', '.txt', '.dat'} else 'npz'
    if fmt == 'csv':
        x_src, diag_src, edges = _load_external_multistate_csv(path, model_cfg)
    elif fmt == 'npz':
        x_src, diag_src, edges = _load_external_multistate_npz(path, model_cfg)
    else:
        raise ValueError(f'Unsupported external format: {fmt}')
    x_src, diag_src, edges = _validate_external_arrays(x_src, diag_src, edges, path)

    diag = np.vstack([
        np.interp(grid.x, x_src, diag_src[s].real).astype(np.complex128)
        + 1j * np.interp(grid.x, x_src, diag_src[s].imag).astype(np.complex128)
        for s in range(diag_src.shape[0])
    ])
    interp_edges: List[EdgeCoupling] = []
    for e in edges:
        vals = np.interp(grid.x, x_src, e.values.real) + 1j * np.interp(grid.x, x_src, e.values.imag)
        interp_edges.append(EdgeCoupling(e.i, e.j, vals.astype(np.complex128), e.name))

    full = _dense_from_diag_and_edges(diag, interp_edges)
    min_gap = _compute_min_gap(full)
    return MultiStatePotentials(nstates=diag.shape[0], diag=diag, edges=interp_edges, full_matrix=full, min_gap=min_gap)


def _load_external_2d_multistate_npz(path: str, model_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[EdgeCoupling]]:
    data = np.load(path)
    columns = model_cfg.get('columns', {})
    x_key = columns.get('x', 'x')
    y_key = columns.get('y', 'y')
    nstates = int(model_cfg['nstates'])
    diag_keys = columns.get('diagonal')
    if diag_keys is None:
        diag_keys = [f'V{i+1}{i+1}' for i in range(nstates)]
    required = [x_key, y_key] + list(diag_keys)
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f'External 2D NPZ missing required arrays {missing}. Available arrays: {list(data.keys())}')

    x = np.asarray(data[x_key], dtype=float)
    y = np.asarray(data[y_key], dtype=float)
    diag_list = [np.asarray(data[k], dtype=np.complex128) for k in diag_keys]
    edge_specs = columns.get('couplings')
    if edge_specs is None:
        edge_specs = []
        for i in range(nstates):
            for j in range(i + 1, nstates):
                nm = f'V{i+1}{j+1}'
                if nm in data:
                    edge_specs.append([i, j, nm])
    edges: List[EdgeCoupling] = []
    for spec in edge_specs:
        if not (isinstance(spec, (list, tuple)) and len(spec) == 3):
            raise ValueError(f'Each 2D coupling spec must be [i, j, key], got: {spec}')
        i, j, key = int(spec[0]), int(spec[1]), str(spec[2])
        if key not in data:
            raise ValueError(f'Coupling array {key} not present in NPZ. Available arrays: {list(data.keys())}')
        vals = np.asarray(data[key], dtype=np.complex128)
        edges.append(EdgeCoupling(min(i, j), max(i, j), vals, _parse_edge_name(min(i, j), max(i, j))))
    diag = np.stack(diag_list, axis=0)
    return x, y, diag, edges


def _validate_external_2d_arrays(x_src: np.ndarray, y_src: np.ndarray, diag_src: np.ndarray, edges: List[EdgeCoupling], path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[EdgeCoupling]]:
    if x_src.ndim != 1 or y_src.ndim != 1:
        raise ValueError(f'External 2D x/y grids must be 1D vectors: {path}')
    if np.any(np.diff(x_src) <= 0.0) or np.any(np.diff(y_src) <= 0.0):
        raise ValueError(f'External 2D x/y grids must be strictly increasing: {path}')
    if diag_src.ndim != 3:
        raise ValueError(f'External 2D diagonal potentials must have shape (nstates, nx, ny): {path}')
    nstates, nx, ny = diag_src.shape
    if x_src.size != nx or y_src.size != ny:
        raise ValueError(f'External 2D grid length mismatch in {path}: diag shape={diag_src.shape}, len(x)={x_src.size}, len(y)={y_src.size}')
    checked_edges: List[EdgeCoupling] = []
    for e in edges:
        vals = np.asarray(e.values, dtype=np.complex128)
        if vals.shape != (nx, ny):
            raise ValueError(f'2D edge {e.name} shape {vals.shape} does not match (nx, ny)=({nx}, {ny}) in {path}')
        checked_edges.append(EdgeCoupling(e.i, e.j, vals, e.name))
    return x_src, y_src, np.asarray(diag_src, dtype=np.complex128), checked_edges


def external_2d_multistate_potentials(grid: Grid, model_cfg: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> MultiStatePotentials:
    if grid.ndim != 2:
        raise ValueError('external_2d_multistate requires a 2D grid')
    path = _resolve_path(str(model_cfg['path']), config=config)
    if not os.path.exists(path):
        raise FileNotFoundError(f'External 2D model file not found: {path}')
    fmt = str(model_cfg.get('format', 'npz')).lower()
    if fmt != 'npz':
        raise ValueError('2D multistate external models currently support NPZ input only')
    x_src, y_src, diag_src, edges = _load_external_2d_multistate_npz(path, model_cfg)
    x_src, y_src, diag_src, edges = _validate_external_2d_arrays(x_src, y_src, diag_src, edges, path)
    if grid.x.size != x_src.size or grid.y is None or grid.y.size != y_src.size:
        raise ValueError('2D external NPZ grid must match simulation grid exactly; no interpolation is performed in v1')
    if not np.allclose(grid.x, x_src, rtol=0.0, atol=1e-12):
        raise ValueError('2D external NPZ x grid does not match simulation x grid')
    if not np.allclose(grid.y, y_src, rtol=0.0, atol=1e-12):
        raise ValueError('2D external NPZ y grid does not match simulation y grid')
    full = _dense_from_diag_and_edges(diag_src, edges)
    min_gap = _compute_min_gap(full)
    return MultiStatePotentials(nstates=diag_src.shape[0], diag=diag_src, edges=edges, full_matrix=full, min_gap=min_gap)


def build_model_potentials(grid: Grid, config: Dict[str, Any]) -> MultiStatePotentials:
    model_cfg = config['model']
    kind = model_cfg['kind']
    if kind in DEFAULT_TULLY_PARAMS:
        if grid.ndim != 1:
            raise ValueError(f'Built-in Tully model {kind} is 1D only')
        return tully_potentials(kind, grid.x, model_cfg.get('parameters'))
    if kind in {'external_1d_two_state', 'external_1d_multistate'}:
        if kind == 'external_1d_two_state' and 'nstates' not in model_cfg:
            model_cfg = dict(model_cfg)
            model_cfg['nstates'] = 2
        return external_multistate_potentials(grid, model_cfg, config=config)
    if kind == 'external_2d_multistate':
        return external_2d_multistate_potentials(grid, model_cfg, config=config)
    raise ValueError(f'Unsupported model kind: {kind}')


# ------------------------------
# Wavepacket, channels, CAP, observables
# ------------------------------

def initial_wavepacket(grid: Grid, initial_cfg: Dict[str, Any], nstates: int) -> np.ndarray:
    state_index = int(initial_cfg['state_index'])
    if grid.ndim == 1:
        x0 = float(initial_cfg['x0'])
        p0 = float(initial_cfg['p0'])
        sigma = float(initial_cfg.get('sigma', initial_cfg.get('a', 0.5)))
        g = np.exp(-((grid.x - x0) ** 2) / (2.0 * sigma * sigma) + 1j * p0 * (grid.x - x0))
        g /= math.sqrt(np.sum(np.abs(g) ** 2) * grid.dx)
        psi = np.zeros((nstates, grid.x.size), dtype=np.complex128)
        psi[state_index] = g
        return psi

    assert grid.y is not None and grid.dy is not None
    x0 = float(initial_cfg['x0'])
    y0 = float(initial_cfg.get('y0', 0.0))
    px0 = float(initial_cfg.get('px0', initial_cfg.get('p0', 0.0)))
    py0 = float(initial_cfg.get('py0', 0.0))
    sigma_x = float(initial_cfg.get('sigma_x', initial_cfg.get('sigma', initial_cfg.get('a', 0.5))))
    sigma_y = float(initial_cfg.get('sigma_y', initial_cfg.get('sigma', initial_cfg.get('a', 0.5))))
    xx, yy = np.meshgrid(grid.x, grid.y, indexing='ij')
    g = np.exp(
        -((xx - x0) ** 2) / (2.0 * sigma_x * sigma_x)
        -((yy - y0) ** 2) / (2.0 * sigma_y * sigma_y)
        + 1j * px0 * (xx - x0)
        + 1j * py0 * (yy - y0)
    )
    g /= math.sqrt(np.sum(np.abs(g) ** 2) * grid.measure)
    psi = np.zeros((nstates, grid.x.size, grid.y.size), dtype=np.complex128)
    psi[state_index] = g
    return psi


def wavefunction_l2_norm(psi: np.ndarray, grid: Grid) -> float:
    """Return the grid-measure-weighted L2 norm of a multistate wavefunction."""
    return math.sqrt(max(0.0, float(np.sum(np.abs(psi) ** 2) * grid.measure)))


def normalize_wavefunction(psi: np.ndarray, grid: Grid, eps: float = 1e-300) -> Tuple[np.ndarray, float]:
    """Normalize a wavefunction by its own grid-measure-weighted L2 norm.

    The returned norm is the value before normalization. If the norm is
    numerically zero, the input is returned unchanged to avoid introducing
    artificial infinities or NaNs.
    """
    norm = wavefunction_l2_norm(psi, grid)
    if norm <= eps:
        return psi, norm
    return psi / norm, norm


def state_populations(psi: np.ndarray, grid: Grid) -> np.ndarray:
    spatial_axes = tuple(range(1, psi.ndim))
    return np.sum(np.abs(psi) ** 2, axis=spatial_axes).real * grid.measure


def _build_radial_masks(grid: Grid, regions: Sequence[RadialRegion]) -> List[np.ndarray]:
    if grid.ndim != 2 or len(regions) == 0:
        return []
    xx, yy = grid.mesh()
    assert yy is not None
    masks = []
    for reg in regions:
        rr = np.sqrt((xx - reg.center[0]) ** 2 + (yy - reg.center[1]) ** 2)
        masks.append((rr >= reg.r_min) & (rr < reg.r_max))
    return masks


def build_cap(grid: Grid, config: Dict[str, Any]) -> Optional[np.ndarray]:
    cap_cfg = config.get('cap', {})
    if not cap_cfg:
        return None
    if not bool(cap_cfg.get('enabled', False)):
        return None
    mode = str(cap_cfg.get('mode', 'separable_poly')).lower()
    if mode in {'off', 'none', 'disabled'}:
        return None
    if mode != 'separable_poly':
        raise ValueError(f'Unsupported CAP mode: {mode}')

    def axis_cap(coord: np.ndarray, lower_on: Optional[float], upper_on: Optional[float], eta: float, power: float) -> np.ndarray:
        vals = np.zeros_like(coord, dtype=float)
        if eta <= 0.0:
            return vals
        xmin = float(coord.min())
        xmax = float(coord.max())
        if lower_on is not None and lower_on > xmin:
            scale = max(1e-12, lower_on - xmin)
            mask = coord < lower_on
            vals[mask] += eta * ((lower_on - coord[mask]) / scale) ** power
        if upper_on is not None and upper_on < xmax:
            scale = max(1e-12, xmax - upper_on)
            mask = coord > upper_on
            vals[mask] += eta * ((coord[mask] - upper_on) / scale) ** power
        return vals

    if grid.ndim == 1:
        xcfg = cap_cfg.get('x', cap_cfg)
        cap = axis_cap(
            grid.x,
            lower_on=xcfg.get('x_min_on', xcfg.get('min_on')),
            upper_on=xcfg.get('x_max_on', xcfg.get('max_on')),
            eta=float(xcfg.get('eta', cap_cfg.get('eta', 0.0))),
            power=float(xcfg.get('power', cap_cfg.get('power', 2))),
        )
        return None if not np.any(np.abs(cap) > 0.0) else cap

    assert grid.y is not None
    xcfg = cap_cfg.get('x', {})
    ycfg = cap_cfg.get('y', {})
    cap_x = axis_cap(
        grid.x,
        lower_on=xcfg.get('x_min_on', xcfg.get('min_on')),
        upper_on=xcfg.get('x_max_on', xcfg.get('max_on')),
        eta=float(xcfg.get('eta', cap_cfg.get('eta', 0.0))),
        power=float(xcfg.get('power', cap_cfg.get('power', 2))),
    )[:, None]
    cap_y = axis_cap(
        grid.y,
        lower_on=ycfg.get('y_min_on', ycfg.get('min_on')),
        upper_on=ycfg.get('y_max_on', ycfg.get('max_on')),
        eta=float(ycfg.get('eta', cap_cfg.get('eta', 0.0))),
        power=float(ycfg.get('power', cap_cfg.get('power', 2))),
    )[None, :]
    cap = cap_x + cap_y
    return None if not np.any(np.abs(cap) > 0.0) else cap


def compute_observables(psi: np.ndarray, grid: Grid, config: Dict[str, Any], radial_regions: Optional[Sequence[RadialRegion]] = None, radial_masks: Optional[Sequence[np.ndarray]] = None) -> Dict[str, Any]:
    rho_state = np.abs(psi) ** 2
    rho = np.sum(rho_state, axis=0)
    spatial_axes = tuple(range(1, psi.ndim))
    pops = np.sum(rho_state, axis=spatial_axes).real * grid.measure
    norm = float(np.sum(rho) * grid.measure)

    channels = _channel_config(config)
    x_div = float(channels.get('x_divider', 0.0))
    if grid.ndim == 1:
        left = grid.x < x_div
        right = ~left
        R_state = np.sum(rho_state[:, left], axis=1).real * grid.measure
        T_state = np.sum(rho_state[:, right], axis=1).real * grid.measure
        if norm > 1e-300:
            x_mean = float(np.sum(grid.x * rho) * grid.measure / norm)
            x2_mean = float(np.sum((grid.x ** 2) * rho) * grid.measure / norm)
            x_width = math.sqrt(max(0.0, x2_mean - x_mean * x_mean))
        else:
            x_mean, x2_mean, x_width = 0.0, 0.0, 0.0
        region_vals = np.zeros(len(radial_regions or []), dtype=float)
        return {
            'pops': pops,
            'norm': norm,
            'R_state': R_state,
            'T_state': T_state,
            'R_total': float(np.sum(R_state)),
            'T_total': float(np.sum(T_state)),
            'x_mean': x_mean,
            'x2_mean': x2_mean,
            'x_width': x_width,
            'y_mean': 0.0,
            'y2_mean': 0.0,
            'y_width': 0.0,
            'region_probs': region_vals,
        }

    assert grid.y is not None
    xx, yy = grid.mesh()
    assert yy is not None
    left = xx < x_div
    right = ~left
    R_state = np.sum(rho_state * left[None, :, :], axis=(1, 2)).real * grid.measure
    T_state = np.sum(rho_state * right[None, :, :], axis=(1, 2)).real * grid.measure
    if norm > 1e-300:
        x_mean = float(np.sum(xx * rho) * grid.measure / norm)
        y_mean = float(np.sum(yy * rho) * grid.measure / norm)
        x2_mean = float(np.sum((xx ** 2) * rho) * grid.measure / norm)
        y2_mean = float(np.sum((yy ** 2) * rho) * grid.measure / norm)
        x_width = math.sqrt(max(0.0, x2_mean - x_mean * x_mean))
        y_width = math.sqrt(max(0.0, y2_mean - y_mean * y_mean))
    else:
        x_mean = y_mean = x2_mean = y2_mean = x_width = y_width = 0.0
    if radial_regions and radial_masks:
        region_vals = np.asarray([float(np.sum(rho[mask]) * grid.measure) for mask in radial_masks], dtype=float)
    else:
        region_vals = np.zeros(len(radial_regions or []), dtype=float)
    return {
        'pops': pops,
        'norm': norm,
        'R_state': R_state,
        'T_state': T_state,
        'R_total': float(np.sum(R_state)),
        'T_total': float(np.sum(T_state)),
        'x_mean': x_mean,
        'x2_mean': x2_mean,
        'x_width': x_width,
        'y_mean': y_mean,
        'y2_mean': y2_mean,
        'y_width': y_width,
        'region_probs': region_vals,
    }


# ------------------------------
# Propagators
# ------------------------------

def _precompute_eigh(mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(mats)
    return evals, evecs


def _apply_batched_potential_from_eigh(psi: np.ndarray, evals: np.ndarray, evecs: np.ndarray, dt: float, cap: Optional[np.ndarray] = None) -> np.ndarray:
    if abs(dt) < 1e-18:
        return psi.copy()
    phase = np.exp(-1j * evals * dt)
    psi_flat, spatial_shape, _ = _reshape_psi_to_points(psi)
    evals_flat = evals.reshape((-1, evals.shape[-1]))
    evecs_flat = evecs.reshape((-1, evecs.shape[-2], evecs.shape[-1]))
    coeff = np.einsum('xji,xj->xi', np.conjugate(evecs_flat), psi_flat, optimize=True)
    coeff *= phase.reshape(coeff.shape)
    out_flat = np.einsum('xij,xj->xi', evecs_flat, coeff, optimize=True)
    out = _restore_psi_from_points(out_flat, spatial_shape)
    if cap is not None:
        out *= np.exp(-np.asarray(cap, dtype=float) * dt)[None, ...]
    return out


def apply_kinetic_half(psi: np.ndarray, grid: Grid, dt: float) -> np.ndarray:
    if abs(dt) < 1e-18:
        return psi.copy()
    if grid.ndim == 1:
        phase = np.exp(-1j * (grid.kx ** 2) * dt / (4.0 * grid.mx))
        psi_k = _fft_1d(psi, axis=1)
        psi_k *= phase[None, :]
        return _ifft_1d(psi_k, axis=1)

    assert grid.ky is not None and grid.my is not None
    k2 = (grid.kx[:, None] ** 2) / (2.0 * grid.mx) + (grid.ky[None, :] ** 2) / (2.0 * grid.my)
    phase = np.exp(-1j * k2 * dt / 2.0)
    psi_k = _fftn_2d(psi, axes=(1, 2))
    psi_k *= phase[None, :, :]
    return _ifftn_2d(psi_k, axes=(1, 2))


def propagate_exact_split(psi: np.ndarray, grid: Grid, full_evals: np.ndarray, full_evecs: np.ndarray, dt: float, cap: Optional[np.ndarray]) -> np.ndarray:
    out = apply_kinetic_half(psi, grid, dt)
    out = _apply_batched_potential_from_eigh(out, full_evals, full_evecs, dt, cap=cap)
    out = apply_kinetic_half(out, grid, dt)
    return out


def propagate_base(psi: np.ndarray, grid: Grid, base_evals: np.ndarray, base_evecs: np.ndarray, dt: float, cap: Optional[np.ndarray]) -> np.ndarray:
    out = apply_kinetic_half(psi, grid, dt)
    out = _apply_batched_potential_from_eigh(out, base_evals, base_evecs, dt, cap=cap)
    out = apply_kinetic_half(out, grid, dt)
    return out


def build_base_and_residual(pots: MultiStatePotentials, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[EdgeCoupling]]:
    base_set = _base_edge_set(config)
    base_edges: List[EdgeCoupling] = []
    residual_edges: List[EdgeCoupling] = []
    for e in pots.edges:
        key = (e.i, e.j)
        if key in base_set:
            base_edges.append(e)
        else:
            residual_edges.append(e)
    base_mat = _dense_from_diag_and_edges(pots.diag, base_edges)
    base_evals, base_evecs = _precompute_eigh(base_mat)
    return base_evals, base_evecs, residual_edges


def edge_expectations(psi: np.ndarray, residual_edges: Sequence[EdgeCoupling], grid: Grid) -> np.ndarray:
    rho = np.sum(np.abs(psi) ** 2, axis=0)
    norm = float(np.sum(rho) * grid.measure)
    if norm < 1e-300 or len(residual_edges) == 0:
        return np.zeros(len(residual_edges), dtype=float)
    vals = [float(np.sum(rho * np.abs(e.values)) * grid.measure / norm) for e in residual_edges]
    return np.asarray(vals, dtype=float)


def gap_expectation(psi: np.ndarray, pots: MultiStatePotentials, grid: Grid) -> float:
    rho = np.sum(np.abs(psi) ** 2, axis=0)
    norm = float(np.sum(rho) * grid.measure)
    if norm < 1e-300:
        return 0.0
    return float(np.sum(rho * pots.min_gap) * grid.measure / norm)


def apply_residual_edge_event(psi: np.ndarray, edge: EdgeCoupling, lam_edge: float) -> np.ndarray:
    if lam_edge <= 0.0:
        raise ValueError('lam_edge must be positive')
    out = np.zeros_like(psi)
    coeff = -1j / lam_edge
    i, j = edge.i, edge.j
    vij = edge.values
    out[i] = coeff * vij * psi[j]
    out[j] = coeff * np.conjugate(vij) * psi[i]
    return out


def clip(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def choose_block_dt(
    psi: np.ndarray,
    pots: MultiStatePotentials,
    residual_edges: Sequence[EdgeCoupling],
    grid: Grid,
    algo: Dict[str, Any],
    t: float,
    tmax: float,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    edge_exp = edge_expectations(psi, residual_edges, grid)
    edge_raw = algo['scale'] * edge_exp
    lam_raw = float(np.sum(edge_raw)) + float(algo.get('floor', 0.0))
    lam = clip(lam_raw, algo['clip_min'], algo['clip_max']) if lam_raw > 0.0 else 0.0
    if lam_raw > 1e-300:
        lam_edges = lam * edge_raw / max(np.sum(edge_raw), 1e-300)
    else:
        lam_edges = np.zeros(len(residual_edges), dtype=float)
    coup_exp = float(np.sum(edge_exp))
    gap_exp_val = gap_expectation(psi, pots, grid)
    if lam <= 1e-14:
        dt = algo['dt_max']
    else:
        dt = clip(algo['mu_target'] / lam, algo['dt_min'], algo['dt_max'])
    if algo.get('gap_refine', True):
        ratio = lam_raw / max(gap_exp_val, 1e-12)
        if gap_exp_val < algo.get('gap_threshold', 0.02) or ratio > algo.get('eta_threshold', 0.25):
            dt = max(algo['dt_min'], algo.get('refine_factor', 0.5) * dt)
    dt = min(dt, tmax - t)
    return float(dt), float(lam), float(lam_raw), float(coup_exp), float(gap_exp_val), lam_edges


def gauss_legendre_nodes_2(t0: float, t1: float) -> List[Tuple[float, float]]:
    center = 0.5 * (t0 + t1)
    half = 0.5 * (t1 - t0)
    xi = 1.0 / math.sqrt(3.0)
    return [(center - half * xi, 0.5), (center + half * xi, 0.5)]


def sample_truncated_poisson_ge2(mu: float, rng: np.random.Generator) -> int:
    if mu <= 0.0:
        return 2
    p0 = math.exp(-mu)
    p1 = mu * p0
    tail = max(1e-300, 1.0 - p0 - p1)
    u = rng.random()
    n = 2
    pn = 0.5 * mu * mu * p0
    cdf = pn / tail
    while u > cdf:
        n += 1
        pn *= mu / n
        cdf += pn / tail
        if n > 10000:
            return n
    return n


def systematic_resample_probs(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = probs.size
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    start = rng.random() / n
    pts = start + np.arange(n) / n
    return np.searchsorted(cdf, pts)


def _sample_batch_masses(sample_batch: np.ndarray) -> np.ndarray:
    flat = sample_batch.reshape((sample_batch.shape[0], -1))
    norms = np.einsum('bi,bi->b', np.conjugate(flat), flat, optimize=True).real
    return np.sqrt(np.maximum(1e-300, norms))


def average_with_optional_resampling(sample_batch: np.ndarray, ess_threshold: float, rng: np.random.Generator, enable_resampling: bool) -> Tuple[np.ndarray, float]:
    ntraj = int(sample_batch.shape[0])
    if ntraj == 0:
        return np.zeros(sample_batch.shape[1:], dtype=sample_batch.dtype), 0.0
    masses = _sample_batch_masses(sample_batch)
    s1 = float(np.sum(masses))
    s2 = float(np.sum(masses * masses))
    ess = (s1 * s1) / max(1e-300, s2)
    if (not enable_resampling) or s1 <= 0.0 or (ess / ntraj) >= ess_threshold:
        return np.mean(sample_batch, axis=0), ess
    scale = masses.reshape((ntraj,) + (1,) * (sample_batch.ndim - 1))
    hats = sample_batch / np.maximum(scale, 1e-300)
    probs = masses / s1
    idx = systematic_resample_probs(probs, rng)
    avg = (s1 / ntraj) * np.mean(hats[idx], axis=0)
    return avg, ess


def estimate_mc_observable_variances(
    sample_batch: np.ndarray,
    psi_det: np.ndarray,
    w_hi: float,
    grid: Grid,
    config: Dict[str, Any],
    variance_batches: int,
    radial_regions: Sequence[RadialRegion],
    radial_masks: Sequence[np.ndarray],
) -> Dict[str, float]:
    keys = {
        'poisson_mc_state_var': 0.0,
        'poisson_mc_R_var': 0.0,
        'poisson_mc_T_var': 0.0,
        'poisson_mc_x_mean_var': 0.0,
        'poisson_mc_width_var': 0.0,
        'poisson_mc_y_mean_var': 0.0,
        'poisson_mc_y_width_var': 0.0,
    }
    nsamples = int(sample_batch.shape[0])
    if nsamples < 2 or variance_batches < 2 or abs(w_hi) < 1e-18:
        return keys
    nb = max(2, min(variance_batches, nsamples))
    chunks = np.array_split(np.arange(nsamples), nb)
    batch_states = []
    batch_R, batch_T, batch_x, batch_w, batch_y, batch_yw = [], [], [], [], [], []
    for idx in chunks:
        hi_avg = np.mean(sample_batch[idx], axis=0)
        psi_batch = psi_det + w_hi * hi_avg
        batch_states.append(psi_batch)
        obs = compute_observables(psi_batch, grid, config, radial_regions=radial_regions, radial_masks=radial_masks)
        batch_R.append(obs['R_total'])
        batch_T.append(obs['T_total'])
        batch_x.append(obs['x_mean'])
        batch_w.append(obs['x_width'])
        batch_y.append(obs['y_mean'])
        batch_yw.append(obs['y_width'])
    batch_states_arr = np.stack(batch_states, axis=0)
    mean_state = np.mean(batch_states_arr, axis=0)
    keys['poisson_mc_state_var'] = float(np.mean(np.sum(np.abs(batch_states_arr - mean_state[None, ...]) ** 2, axis=tuple(range(1, batch_states_arr.ndim)))))
    if len(batch_R) > 1:
        keys['poisson_mc_R_var'] = float(np.var(np.asarray(batch_R), ddof=1))
        keys['poisson_mc_T_var'] = float(np.var(np.asarray(batch_T), ddof=1))
        keys['poisson_mc_x_mean_var'] = float(np.var(np.asarray(batch_x), ddof=1))
        keys['poisson_mc_width_var'] = float(np.var(np.asarray(batch_w), ddof=1))
        keys['poisson_mc_y_mean_var'] = float(np.var(np.asarray(batch_y), ddof=1))
        keys['poisson_mc_y_width_var'] = float(np.var(np.asarray(batch_yw), ddof=1))
    return keys


def _sample_edge_index(lam_edges: np.ndarray, rng: np.random.Generator) -> int:
    probs = lam_edges / max(np.sum(lam_edges), 1e-300)
    return int(rng.choice(len(lam_edges), p=probs))


def _generate_hi_sample_batch_serial(
    psi: np.ndarray,
    grid: Grid,
    base_evals: np.ndarray,
    base_evecs: np.ndarray,
    residual_edges: Sequence[EdgeCoupling],
    dt_block: float,
    mu: float,
    lam_edges: np.ndarray,
    nsamples: int,
    rng: np.random.Generator,
    cap: Optional[np.ndarray],
) -> np.ndarray:
    batch = np.empty((nsamples,) + psi.shape, dtype=np.complex128)
    for idx in range(nsamples):
        n = sample_truncated_poisson_ge2(mu, rng)
        taus = np.sort(rng.uniform(0.0, dt_block, size=n))
        tmp = psi.copy()
        prev = 0.0
        for tau in taus:
            tmp = propagate_base(tmp, grid, base_evals, base_evecs, float(tau - prev), cap)
            edge_idx = _sample_edge_index(lam_edges, rng)
            tmp = apply_residual_edge_event(tmp, residual_edges[edge_idx], lam_edges[edge_idx])
            prev = float(tau)
        tmp = propagate_base(tmp, grid, base_evals, base_evecs, dt_block - prev, cap)
        batch[idx] = tmp
    return batch


def _parallel_hi_worker(nsamples: int, seed: int, psi: np.ndarray, dt_block: float, mu: float, lam_edges: np.ndarray) -> np.ndarray:
    key = ('hi', os.getpid())
    if key not in _WORKER_SEEN:
        _WORKER_SEEN.add(key)
        _print_runtime_status('HI_WORKER', extra=f'assigned_samples={int(nsamples)} first_seed={int(seed)}')
    ctx = _PARALLEL_STATE['hi']
    rng = np.random.default_rng(int(seed))
    return _generate_hi_sample_batch_serial(
        psi=psi,
        grid=ctx['grid'],
        base_evals=ctx['base_evals'],
        base_evecs=ctx['base_evecs'],
        residual_edges=ctx['residual_edges'],
        dt_block=dt_block,
        mu=mu,
        lam_edges=lam_edges,
        nsamples=nsamples,
        rng=rng,
        cap=ctx['cap'],
    )


def _parallel_hi_sample_batch(
    psi: np.ndarray,
    dt_block: float,
    mu: float,
    lam_edges: np.ndarray,
    total_samples: int,
    rng: np.random.Generator,
    executor: Optional[ProcessPoolExecutor],
    workers: int,
    grid: Grid,
    base_evals: np.ndarray,
    base_evecs: np.ndarray,
    residual_edges: Sequence[EdgeCoupling],
    cap: Optional[np.ndarray],
) -> np.ndarray:
    if total_samples <= 0:
        return np.zeros((0,) + psi.shape, dtype=np.complex128)
    if executor is None or workers <= 1:
        _print_runtime_status('HI_SERIAL_FALLBACK', extra=f'workers={int(workers)} total_samples={int(total_samples)}')
        return _generate_hi_sample_batch_serial(psi, grid, base_evals, base_evecs, residual_edges, dt_block, mu, lam_edges, total_samples, rng, cap)
    if not _STATUS_ONCE['hi_pool']:
        _print_runtime_status('HI_POOL', extra=f'workers={int(workers)} total_samples={int(total_samples)}')
        _STATUS_ONCE['hi_pool'] = True
    chunks = [c for c in np.array_split(np.arange(total_samples), workers) if len(c) > 0]
    counts = [len(c) for c in chunks]
    seeds = rng.integers(0, 2**63 - 1, size=len(counts), dtype=np.int64)
    futs = [executor.submit(_parallel_hi_worker, cnt, int(seed), psi, dt_block, mu, lam_edges) for cnt, seed in zip(counts, seeds)]
    batches = [f.result() for f in futs]
    return np.concatenate(batches, axis=0)


def propagate_block_v1(
    psi: np.ndarray,
    grid: Grid,
    base_evals: np.ndarray,
    base_evecs: np.ndarray,
    residual_edges: Sequence[EdgeCoupling],
    dt_block: float,
    lam: float,
    lam_edges: np.ndarray,
    algo: Dict[str, Any],
    rng: np.random.Generator,
    config: Dict[str, Any],
    cap: Optional[np.ndarray],
    variance_batches: int,
    radial_regions: Sequence[RadialRegion],
    radial_masks: Sequence[np.ndarray],
    hi_executor: Optional[ProcessPoolExecutor] = None,
    hi_parallel_workers: int = 1,
) -> Tuple[np.ndarray, float, float, Dict[str, float], np.ndarray]:
    if dt_block <= 0.0:
        psi_out = psi.copy()
        raw_norm = wavefunction_l2_norm(psi_out, grid)
        if bool(algo.get('normalize_block_output', True)):
            psi_out, raw_norm = normalize_wavefunction(psi_out, grid)
        return psi_out, 0.0, 0.0, {
            'poisson_mc_state_var': 0.0, 'poisson_mc_R_var': 0.0, 'poisson_mc_T_var': 0.0,
            'poisson_mc_x_mean_var': 0.0, 'poisson_mc_width_var': 0.0,
            'poisson_mc_y_mean_var': 0.0, 'poisson_mc_y_width_var': 0.0,
            'block_norm_raw': raw_norm,
        }, np.zeros(psi.shape[0], dtype=float)

    psi0 = propagate_base(psi, grid, base_evals, base_evecs, dt_block, cap)
    mu = lam * dt_block
    psi1 = np.zeros_like(psi)
    qnodes = gauss_legendre_nodes_2(0.0, dt_block)
    if lam > 1e-14 and len(residual_edges) > 0:
        propagated_to_tau = {float(tau): propagate_base(psi, grid, base_evals, base_evecs, float(tau), cap) for tau, _ in qnodes}
        for edge, lam_e in zip(residual_edges, lam_edges):
            if lam_e <= 1e-16:
                continue
            mu_e = lam_e * dt_block
            tmp_sum = np.zeros_like(psi)
            for tau, w in qnodes:
                tau = float(tau)
                tmp = apply_residual_edge_event(propagated_to_tau[tau], edge, lam_e)
                tmp = propagate_base(tmp, grid, base_evals, base_evecs, dt_block - tau, cap)
                tmp_sum += w * tmp
            psi1 += mu_e * tmp_sum
    psi_det = psi0 + psi1

    w_hi = math.exp(mu) - 1.0 - mu
    psi_hi = np.zeros_like(psi)
    ess = float(max(1, algo.get('ntraj_hi', 0)))
    var_diag = {
        'poisson_mc_state_var': 0.0, 'poisson_mc_R_var': 0.0, 'poisson_mc_T_var': 0.0,
        'poisson_mc_x_mean_var': 0.0, 'poisson_mc_width_var': 0.0,
        'poisson_mc_y_mean_var': 0.0, 'poisson_mc_y_width_var': 0.0,
        'block_norm_raw': 1.0,
    }
    per_state_var = np.zeros(psi.shape[0], dtype=float)

    if w_hi > 0.0 and algo.get('ntraj_hi', 0) > 0 and lam > 1e-14 and len(residual_edges) > 0:
        sample_batch = _parallel_hi_sample_batch(
            psi=psi,
            dt_block=dt_block,
            mu=mu,
            lam_edges=lam_edges,
            total_samples=int(algo['ntraj_hi']),
            rng=rng,
            executor=hi_executor,
            workers=hi_parallel_workers,
            grid=grid,
            base_evals=base_evals,
            base_evecs=base_evecs,
            residual_edges=residual_edges,
            cap=cap,
        )
        var_diag = estimate_mc_observable_variances(sample_batch, psi_det, w_hi, grid, config, variance_batches, radial_regions, radial_masks)
        avg_hi, ess = average_with_optional_resampling(
            sample_batch,
            ess_threshold=algo.get('ess_threshold', 0.5),
            rng=rng,
            enable_resampling=algo.get('resampling_enabled', False),
        )
        pop_batch = np.sum(np.abs(sample_batch) ** 2, axis=tuple(range(2, sample_batch.ndim))).real * grid.measure
        if pop_batch.shape[0] > 1:
            per_state_var = np.var(pop_batch, axis=0, ddof=1)
        psi_hi = w_hi * avg_hi

    psi_next = psi_det + psi_hi
    if bool(algo.get('normalize_block_output', True)):
        psi_next, raw_norm = normalize_wavefunction(psi_next, grid)
    else:
        raw_norm = wavefunction_l2_norm(psi_next, grid)
    var_diag['block_norm_raw'] = raw_norm
    return psi_next, ess, w_hi, var_diag, per_state_var


# ------------------------------
# Output helpers
# ------------------------------

def _timeseries_columns(nstates: int, ndim: int, region_names: Sequence[str]) -> List[str]:
    cols = [
        'time', 'pop1', 'pop2', 'norm', 'block_norm_raw', 'R1', 'R2', 'T1', 'T2', 'R_total', 'T_total',
        'x_mean', 'x_width', 'x2_mean', 'lambda', 'lambda_raw', 'block_dt', 'ess', 'event_weight_hi',
        'coupling_abs_exp', 'gap_exp', 'poisson_mc_state_var', 'poisson_mc_pop1_var', 'poisson_mc_pop2_var',
        'poisson_mc_R_var', 'poisson_mc_T_var', 'poisson_mc_x_mean_var', 'poisson_mc_width_var',
        'paper_var_phi1_sq', 'paper_var_phi2_sq', 'paper_var_norm_sq',
    ]
    if ndim == 2:
        cols += ['y_mean', 'y_width', 'y2_mean', 'poisson_mc_y_mean_var', 'poisson_mc_y_width_var']
        cols += [f'region_{nm}' for nm in region_names]
    for s in range(2, nstates):
        idx = s + 1
        cols += [f'pop{idx}', f'R{idx}', f'T{idx}', f'poisson_mc_pop{idx}_var', f'paper_var_phi{idx}_sq']
    return cols


def _progress_columns(ndim: int) -> List[str]:
    cols = [
        'percent', 'sim_time', 'elapsed_seconds', 'blocks_completed', 'pop1', 'pop2', 'norm', 'block_norm_raw', 'R_total', 'T_total',
        'x_mean', 'x_width', 'lambda', 'lambda_raw', 'block_dt', 'ess', 'event_weight_hi', 'coupling_abs_exp',
        'gap_exp', 'poisson_mc_state_var'
    ]
    if ndim == 2:
        cols += ['y_mean', 'y_width']
    return cols


def write_timeseries_csv(path: str, results: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    nstates = int(results['nstates'])
    ndim = int(results.get('ndim', 1))
    region_names = list(results.get('region_names', []))
    cols = _timeseries_columns(nstates, ndim, region_names)
    times = results['times']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(times.size):
            rowmap = {
                'time': results['times'][i],
                'pop1': results['populations'][0, i],
                'pop2': results['populations'][1, i] if nstates > 1 else 0.0,
                'norm': results['norm'][i],
                'block_norm_raw': results.get('block_norm_raw', results['norm'])[i],
                'R1': results['R_state'][0, i],
                'R2': results['R_state'][1, i] if nstates > 1 else 0.0,
                'T1': results['T_state'][0, i],
                'T2': results['T_state'][1, i] if nstates > 1 else 0.0,
                'R_total': results['R_total'][i],
                'T_total': results['T_total'][i],
                'x_mean': results['x_mean'][i],
                'x_width': results['x_width'][i],
                'x2_mean': results['x2_mean'][i],
                'lambda': results['lambdas'][i],
                'lambda_raw': results['lambda_raw'][i],
                'block_dt': results['block_dt'][i],
                'ess': results['ess'][i],
                'event_weight_hi': results['event_weight_hi'][i],
                'coupling_abs_exp': results['coupling_abs_exp'][i],
                'gap_exp': results['gap_exp'][i],
                'poisson_mc_state_var': results['poisson_mc_state_var'][i],
                'poisson_mc_pop1_var': results['poisson_mc_pop_var'][0, i],
                'poisson_mc_pop2_var': results['poisson_mc_pop_var'][1, i] if nstates > 1 else 0.0,
                'poisson_mc_R_var': results['poisson_mc_R_var'][i],
                'poisson_mc_T_var': results['poisson_mc_T_var'][i],
                'poisson_mc_x_mean_var': results['poisson_mc_x_mean_var'][i],
                'poisson_mc_width_var': results['poisson_mc_width_var'][i],
                'paper_var_phi1_sq': results['paper_var_pop'][0, i],
                'paper_var_phi2_sq': results['paper_var_pop'][1, i] if nstates > 1 else 0.0,
                'paper_var_norm_sq': results['paper_var_norm'][i],
            }
            if ndim == 2:
                rowmap['y_mean'] = results['y_mean'][i]
                rowmap['y_width'] = results['y_width'][i]
                rowmap['y2_mean'] = results['y2_mean'][i]
                rowmap['poisson_mc_y_mean_var'] = results['poisson_mc_y_mean_var'][i]
                rowmap['poisson_mc_y_width_var'] = results['poisson_mc_y_width_var'][i]
                for ir, nm in enumerate(region_names):
                    rowmap[f'region_{nm}'] = results['region_probs'][ir, i]
            for s in range(2, nstates):
                idx = s + 1
                rowmap[f'pop{idx}'] = results['populations'][s, i]
                rowmap[f'R{idx}'] = results['R_state'][s, i]
                rowmap[f'T{idx}'] = results['T_state'][s, i]
                rowmap[f'poisson_mc_pop{idx}_var'] = results['poisson_mc_pop_var'][s, i]
                rowmap[f'paper_var_phi{idx}_sq'] = results['paper_var_pop'][s, i]
            writer.writerow([f"{rowmap[c]:.12g}" for c in cols])
    return path


def write_model_profile_csv(path: str, grid: Grid, pots: MultiStatePotentials) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if grid.ndim != 1:
        raise ValueError('write_model_profile_csv is 1D only')
    header = ['x'] + [f'V{i+1}{i+1}' for i in range(pots.nstates)]
    for e in pots.edges:
        header.append(e.name)
    for e in pots.edges:
        header.append(f'abs_{e.name}')
    header.append('min_gap')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ix, x in enumerate(grid.x):
            row: List[str] = [f'{x:.12g}']
            row += [f'{pots.diag[s, ix].real:.12g}' for s in range(pots.nstates)]
            row += [f'{e.values[ix].real:.12g}' if abs(e.values[ix].imag) < 1e-14 else f'{e.values[ix]}' for e in pots.edges]
            row += [f'{abs(e.values[ix]):.12g}' for e in pots.edges]
            row += [f'{pots.min_gap[ix]:.12g}']
            writer.writerow(row)
    return path


def _nearest_index(vec: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(vec - value)))


def write_model_preview_flat_csv(path: str, grid: Grid, pots: MultiStatePotentials) -> str:
    if grid.ndim != 2 or grid.y is None:
        raise ValueError('write_model_preview_flat_csv requires a 2D grid')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    header = ['ix', 'iy', 'x', 'y'] + [f'V{i+1}{i+1}' for i in range(pots.nstates)]
    for e in pots.edges:
        header.append(e.name)
    header.append('min_gap')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ix, x in enumerate(grid.x):
            for iy, y in enumerate(grid.y):
                row: List[str] = [str(ix), str(iy), f'{x:.12g}', f'{y:.12g}']
                row += [f'{pots.diag[s, ix, iy].real:.12g}' for s in range(pots.nstates)]
                row += [f'{e.values[ix, iy].real:.12g}' if abs(e.values[ix, iy].imag) < 1e-14 else f'{e.values[ix, iy]}' for e in pots.edges]
                row += [f'{pots.min_gap[ix, iy]:.12g}']
                writer.writerow(row)
    return path


def write_model_cut_csv(path: str, grid: Grid, pots: MultiStatePotentials, preview_cfg: Dict[str, Any]) -> str:
    if grid.ndim != 2 or grid.y is None:
        raise ValueError('write_model_cut_csv requires a 2D grid')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    header = ['cut_type', 'cut_value', 'index', 'coord'] + [f'V{i+1}{i+1}' for i in range(pots.nstates)]
    for e in pots.edges:
        header.append(e.name)
    header.append('min_gap')
    yvals = [float(v) for v in preview_cfg.get('preview_y_values', [0.0])]
    xvals = [float(v) for v in preview_cfg.get('preview_x_values', [])]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for y0 in yvals:
            iy = _nearest_index(grid.y, y0)
            for ix, x in enumerate(grid.x):
                row: List[str] = ['y_fixed', f'{grid.y[iy]:.12g}', str(ix), f'{x:.12g}']
                row += [f'{pots.diag[s, ix, iy].real:.12g}' for s in range(pots.nstates)]
                row += [f'{e.values[ix, iy].real:.12g}' if abs(e.values[ix, iy].imag) < 1e-14 else f'{e.values[ix, iy]}' for e in pots.edges]
                row += [f'{pots.min_gap[ix, iy]:.12g}']
                writer.writerow(row)
        for x0 in xvals:
            ix = _nearest_index(grid.x, x0)
            for iy, y in enumerate(grid.y):
                row = ['x_fixed', f'{grid.x[ix]:.12g}', str(iy), f'{y:.12g}']
                row += [f'{pots.diag[s, ix, iy].real:.12g}' for s in range(pots.nstates)]
                row += [f'{e.values[ix, iy].real:.12g}' if abs(e.values[ix, iy].imag) < 1e-14 else f'{e.values[ix, iy]}' for e in pots.edges]
                row += [f'{pots.min_gap[ix, iy]:.12g}']
                writer.writerow(row)
    return path


def append_progress_row(path: str, row: Dict[str, float], ndim: int, header_if_new: bool = True) -> None:
    cols = _progress_columns(ndim)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header_if_new and not exists:
            writer.writerow(cols)
        writer.writerow([f'{row[c]:.12g}' for c in cols])
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def log_progress(tag: str, percent: float, t: float, tmax: float, elapsed: float, block_count: int, obs: Dict[str, Any], lam: float, lam_raw: float, dt_blk: float, ess: float, w_hi: float, coup_exp: float, gap_exp_val: float, state_var: float, ndim: int, wf_error_inst: Optional[float] = None, wf_error_cum_rms: Optional[float] = None, wf_fidelity: Optional[float] = None) -> None:
    pops = obs['pops']
    p1 = pops[0] if pops.size > 0 else 0.0
    p2 = pops[1] if pops.size > 1 else 0.0
    extra = '' if ndim == 1 else f' | <y>={obs["y_mean"]:.8f} | y_width={obs["y_width"]:.8f}'
    wf_extra = ''
    if wf_error_inst is not None:
        wf_extra += f' | wf_err={wf_error_inst:.6e}'
    if wf_error_cum_rms is not None:
        wf_extra += f' | wf_rms={wf_error_cum_rms:.6e}'
    if wf_fidelity is not None:
        wf_extra += f' | fidelity={wf_fidelity:.8f}'
    print(
        f'[progress][{tag}] {percent:5.1f}% | t={t:.6f}/{tmax:.6f} | blocks={block_count} | '
        f'elapsed={elapsed:.2f}s | pop=({p1:.8f}, {p2:.8f}) | norm={obs["norm"]:.8f} | '
        f'R={obs["R_total"]:.8f} | T={obs["T_total"]:.8f} | <x>={obs["x_mean"]:.8f} | width={obs["x_width"]:.8f}{extra} | '
        f'lambda={lam:.6e} | lambda_raw={lam_raw:.6e} | dt_blk={dt_blk:.6e} | '
        f'ESS={ess:.3f} | w_hi={w_hi:.6e} | <|Voff|>={coup_exp:.6e} | <gap>={gap_exp_val:.6e} | '
        f'mc_state_var={state_var:.6e}{wf_extra}',
        flush=True,
    )


# ------------------------------
# Paper-style variance (optional)
# ------------------------------

def propagate_block_replica(
    psi: np.ndarray,
    grid: Grid,
    base_evals: np.ndarray,
    base_evecs: np.ndarray,
    residual_edges: Sequence[EdgeCoupling],
    dt_block: float,
    lam: float,
    lam_edges: np.ndarray,
    ntraj_hi: int,
    rng: np.random.Generator,
    cap: Optional[np.ndarray],
    normalize_block_output: bool = True,
) -> np.ndarray:
    psi0 = propagate_base(psi, grid, base_evals, base_evecs, dt_block, cap)
    mu = lam * dt_block
    psi1 = np.zeros_like(psi)
    if lam > 1e-14 and len(residual_edges) > 0:
        qnodes = gauss_legendre_nodes_2(0.0, dt_block)
        propagated_to_tau = {float(tau): propagate_base(psi, grid, base_evals, base_evecs, float(tau), cap) for tau, _ in qnodes}
        for edge, lam_e in zip(residual_edges, lam_edges):
            if lam_e <= 1e-16:
                continue
            mu_e = lam_e * dt_block
            tmp_sum = np.zeros_like(psi)
            for tau, w in qnodes:
                tau = float(tau)
                tmp = apply_residual_edge_event(propagated_to_tau[tau], edge, lam_e)
                tmp = propagate_base(tmp, grid, base_evals, base_evecs, dt_block - tau, cap)
                tmp_sum += w * tmp
            psi1 += mu_e * tmp_sum
    psi_det = psi0 + psi1
    w_hi = math.exp(mu) - 1.0 - mu
    if w_hi <= 0.0 or ntraj_hi <= 0 or lam <= 1e-14 or len(residual_edges) == 0:
        psi_next = psi_det
    else:
        sample_batch = _generate_hi_sample_batch_serial(psi, grid, base_evals, base_evecs, residual_edges, dt_block, mu, lam_edges, int(ntraj_hi), rng, cap)
        psi_next = psi_det + w_hi * np.mean(sample_batch, axis=0)
    if normalize_block_output:
        psi_next, _ = normalize_wavefunction(psi_next, grid)
    return psi_next


def _paper_variance_replica_from_state(replica_index: int) -> Tuple[np.ndarray, np.ndarray]:
    key = ('replica', os.getpid())
    if key not in _WORKER_SEEN:
        _WORKER_SEEN.add(key)
        _print_runtime_status('REPLICA_WORKER', extra=f'first_replica_index={int(replica_index)}')
    ctx = _PARALLEL_STATE['paper_variance']
    config = ctx['config']
    grid = ctx['grid']
    base_evals = ctx['base_evals']
    base_evecs = ctx['base_evecs']
    residual_edges = ctx['residual_edges']
    dt_hist = ctx['dt_hist']
    lam_hist = ctx['lam_hist']
    lam_edge_hist = ctx['lam_edge_hist']
    sample_times = ctx['sample_times']
    cap = ctx['cap']
    ntraj_hi = ctx['ntraj_hi']
    radial_regions = ctx['radial_regions']
    radial_masks = ctx['radial_masks']
    nstates = ctx['nstates']
    base_seed = ctx['base_seed']

    rng = np.random.default_rng(base_seed + 100000 + int(replica_index))
    psi = initial_wavepacket(grid, config['initial_state'], nstates)
    t = 0.0
    idx = 0
    pop = np.zeros((nstates, sample_times.size), dtype=float)
    norm = np.zeros(sample_times.size, dtype=float)
    for b, dt in enumerate(dt_hist):
        if dt <= 0.0:
            break
        psi = propagate_block_replica(
            psi, grid, base_evals, base_evecs, residual_edges, float(dt), float(lam_hist[b]),
            lam_edge_hist[b], ntraj_hi, rng, cap,
            normalize_block_output=bool(config.get('algorithm', {}).get('normalize_block_output', True)),
        )
        t += float(dt)
        while idx < sample_times.size and t >= sample_times[idx] - 1e-12:
            obs = compute_observables(psi, grid, config, radial_regions=radial_regions, radial_masks=radial_masks)
            pop[:, idx] = obs['pops']
            norm[idx] = obs['norm']
            idx += 1
    while idx < sample_times.size:
        obs = compute_observables(psi, grid, config, radial_regions=radial_regions, radial_masks=radial_masks)
        pop[:, idx] = obs['pops']
        norm[idx] = obs['norm']
        idx += 1
    return pop, norm


def estimate_paper_style_population_variance(
    config: Dict[str, Any],
    grid: Grid,
    pots: MultiStatePotentials,
    base_evals: np.ndarray,
    base_evecs: np.ndarray,
    residual_edges: Sequence[EdgeCoupling],
    dt_hist: np.ndarray,
    lam_hist: np.ndarray,
    lam_edge_hist: np.ndarray,
    sample_times: np.ndarray,
    cap: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    analysis = config.get('analysis', {})
    nrep = int(analysis.get('paper_variance_replicas', 0))
    if nrep < 2:
        return np.zeros((pots.nstates, sample_times.size), dtype=float), np.zeros(sample_times.size, dtype=float)
    ntraj_hi = int(analysis.get('paper_variance_ntraj_hi', config['algorithm'].get('ntraj_hi', 0)))
    base_seed = int(config['simulation'].get('seed', 12345))
    radial_regions = _parse_radial_regions(config)
    radial_masks = _build_radial_masks(grid, radial_regions)

    pop_rep = np.zeros((nrep, pots.nstates, sample_times.size), dtype=float)
    norm_rep = np.zeros((nrep, sample_times.size), dtype=float)
    state = {
        'config': config,
        'grid': grid,
        'base_evals': base_evals,
        'base_evecs': base_evecs,
        'residual_edges': residual_edges,
        'dt_hist': dt_hist,
        'lam_hist': lam_hist,
        'lam_edge_hist': lam_edge_hist,
        'sample_times': sample_times,
        'cap': cap,
        'ntraj_hi': ntraj_hi,
        'radial_regions': radial_regions,
        'radial_masks': radial_masks,
        'nstates': pots.nstates,
        'base_seed': base_seed,
    }
    _PARALLEL_STATE['paper_variance'] = state
    executor, workers = _make_process_pool('replica', nrep)
    if executor is None or workers <= 1:
        _print_runtime_status('REPLICA_SERIAL_FALLBACK', extra=f'workers={int(workers)} nreplicas={int(nrep)}')
        try:
            for r in range(nrep):
                pop_rep[r], norm_rep[r] = _paper_variance_replica_from_state(r)
        finally:
            _PARALLEL_STATE.pop('paper_variance', None)
        return np.var(pop_rep, axis=0, ddof=1), np.var(norm_rep, axis=0, ddof=1)

    try:
        if not _STATUS_ONCE['replica_pool']:
            _print_runtime_status('REPLICA_POOL', extra=f'workers={int(workers)} nreplicas={int(nrep)}')
            _STATUS_ONCE['replica_pool'] = True
        futs = [executor.submit(_paper_variance_replica_from_state, r) for r in range(nrep)]
        for r, fut in enumerate(futs):
            pop_rep[r], norm_rep[r] = fut.result()
    finally:
        executor.shutdown(wait=True)
        _PARALLEL_STATE.pop('paper_variance', None)
    return np.var(pop_rep, axis=0, ddof=1), np.var(norm_rep, axis=0, ddof=1)


# ------------------------------
# Core simulations
# ------------------------------

def _parse_wf_error_tracking_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    analysis = config.get('analysis', {})
    raw = analysis.get('wavefunction_error_tracking', False)
    if isinstance(raw, dict):
        return {
            'enabled': bool(raw.get('enabled', False)),
            'progress_log': bool(raw.get('progress_log', True)),
        }
    return {'enabled': bool(raw), 'progress_log': True}


def _build_result_storage(nstates: int, nsteps: int, ndim: int, region_names: Sequence[str]) -> Dict[str, Any]:
    out = {
        'nstates': nstates,
        'ndim': ndim,
        'region_names': list(region_names),
        'times': np.zeros(nsteps, dtype=float),
        'populations': np.zeros((nstates, nsteps), dtype=float),
        'norm': np.zeros(nsteps, dtype=float),
        'lambdas': np.zeros(nsteps, dtype=float),
        'lambda_raw': np.zeros(nsteps, dtype=float),
        'block_dt': np.zeros(nsteps, dtype=float),
        'ess': np.zeros(nsteps, dtype=float),
        'event_weight_hi': np.zeros(nsteps, dtype=float),
        'coupling_abs_exp': np.zeros(nsteps, dtype=float),
        'gap_exp': np.zeros(nsteps, dtype=float),
        'R_state': np.zeros((nstates, nsteps), dtype=float),
        'T_state': np.zeros((nstates, nsteps), dtype=float),
        'R_total': np.zeros(nsteps, dtype=float),
        'T_total': np.zeros(nsteps, dtype=float),
        'x_mean': np.zeros(nsteps, dtype=float),
        'x_width': np.zeros(nsteps, dtype=float),
        'x2_mean': np.zeros(nsteps, dtype=float),
        'y_mean': np.zeros(nsteps, dtype=float),
        'y_width': np.zeros(nsteps, dtype=float),
        'y2_mean': np.zeros(nsteps, dtype=float),
        'region_probs': np.zeros((len(region_names), nsteps), dtype=float),
        'poisson_mc_state_var': np.zeros(nsteps, dtype=float),
        'poisson_mc_pop_var': np.zeros((nstates, nsteps), dtype=float),
        'poisson_mc_R_var': np.zeros(nsteps, dtype=float),
        'poisson_mc_T_var': np.zeros(nsteps, dtype=float),
        'poisson_mc_x_mean_var': np.zeros(nsteps, dtype=float),
        'poisson_mc_width_var': np.zeros(nsteps, dtype=float),
        'poisson_mc_y_mean_var': np.zeros(nsteps, dtype=float),
        'poisson_mc_y_width_var': np.zeros(nsteps, dtype=float),
        'paper_var_pop': np.zeros((nstates, nsteps), dtype=float),
        'paper_var_norm': np.zeros(nsteps, dtype=float),
        'block_norm_raw': np.ones(nsteps, dtype=float),
    }
    return out


def _build_wf_diag_storage(nsteps: int) -> Dict[str, Any]:
    return {
        'times': np.zeros(nsteps, dtype=float),
        'block_index': np.zeros(nsteps, dtype=int),
        'ess': np.zeros(nsteps, dtype=float),
        'wf_error_inst': np.zeros(nsteps, dtype=float),
        'wf_error_cum_rms': np.zeros(nsteps, dtype=float),
        'wf_fidelity': np.zeros(nsteps, dtype=float),
    }


def _snapshot_wf_diag(store: Dict[str, Any], idx: int, t: float, block_index: int, ess: float, wf_error_inst: float, wf_error_cum_rms: float, wf_fidelity: float) -> None:
    store['times'][idx] = t
    store['block_index'][idx] = block_index
    store['ess'][idx] = ess
    store['wf_error_inst'][idx] = wf_error_inst
    store['wf_error_cum_rms'][idx] = wf_error_cum_rms
    store['wf_fidelity'][idx] = wf_fidelity


def _slice_wf_diag(store: Dict[str, Any], nfill: int) -> Dict[str, Any]:
    return {k: (v[:nfill].copy() if isinstance(v, np.ndarray) else v) for k, v in store.items()}


def _propagate_exact_to_time(psi_exact: np.ndarray, current_t: float, target_t: float, exact_dt: float, grid: Grid, full_evals: np.ndarray, full_evecs: np.ndarray, cap: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    t = float(current_t)
    psi = psi_exact
    while t < target_t - 1e-12:
        cur_dt = min(exact_dt, target_t - t)
        if cur_dt <= 0.0:
            break
        psi = propagate_exact_split(psi, grid, full_evals, full_evecs, cur_dt, cap)
        t += cur_dt
    return psi, t


def _wavefunction_distance_and_fidelity(psi: np.ndarray, psi_exact: np.ndarray) -> Tuple[float, float]:
    flat = np.ravel(psi)
    flat_ex = np.ravel(psi_exact)
    n_ex_sq = float(np.vdot(flat_ex, flat_ex).real)
    n_sq = float(np.vdot(flat, flat).real)
    if n_ex_sq <= 1e-300 or n_sq <= 1e-300:
        return 0.0, 1.0
    overlap = np.vdot(flat_ex, flat)
    abs_overlap = abs(overlap)
    phase = overlap / abs_overlap if abs_overlap > 1e-300 else 1.0 + 0.0j
    diff = flat - phase * flat_ex
    dist = math.sqrt(max(0.0, float(np.vdot(diff, diff).real) / n_ex_sq))
    fidelity = (abs_overlap * abs_overlap) / max(1e-300, n_ex_sq * n_sq)
    fidelity = min(1.0, max(0.0, float(fidelity)))
    return dist, fidelity


def _snapshot_results(store: Dict[str, Any], idx: int, t: float, obs: Dict[str, Any], lam: float, lam_raw: float, dt_blk: float, ess: float, w_hi: float, coup_exp: float, gap_exp_val: float, mc_diag: Dict[str, float], per_state_var: np.ndarray) -> None:
    store['times'][idx] = t
    store['populations'][:, idx] = obs['pops']
    store['norm'][idx] = obs['norm']
    store['lambdas'][idx] = lam
    store['lambda_raw'][idx] = lam_raw
    store['block_dt'][idx] = dt_blk
    store['ess'][idx] = ess
    store['event_weight_hi'][idx] = w_hi
    store['coupling_abs_exp'][idx] = coup_exp
    store['gap_exp'][idx] = gap_exp_val
    store['R_state'][:, idx] = obs['R_state']
    store['T_state'][:, idx] = obs['T_state']
    store['R_total'][idx] = obs['R_total']
    store['T_total'][idx] = obs['T_total']
    store['x_mean'][idx] = obs['x_mean']
    store['x_width'][idx] = obs['x_width']
    store['x2_mean'][idx] = obs['x2_mean']
    store['y_mean'][idx] = obs.get('y_mean', 0.0)
    store['y_width'][idx] = obs.get('y_width', 0.0)
    store['y2_mean'][idx] = obs.get('y2_mean', 0.0)
    if store['region_probs'].shape[0] > 0:
        store['region_probs'][:, idx] = obs.get('region_probs', 0.0)
    store['poisson_mc_state_var'][idx] = mc_diag.get('poisson_mc_state_var', 0.0)
    store['poisson_mc_R_var'][idx] = mc_diag.get('poisson_mc_R_var', 0.0)
    store['poisson_mc_T_var'][idx] = mc_diag.get('poisson_mc_T_var', 0.0)
    store['poisson_mc_x_mean_var'][idx] = mc_diag.get('poisson_mc_x_mean_var', 0.0)
    store['poisson_mc_width_var'][idx] = mc_diag.get('poisson_mc_width_var', 0.0)
    store['poisson_mc_y_mean_var'][idx] = mc_diag.get('poisson_mc_y_mean_var', 0.0)
    store['poisson_mc_y_width_var'][idx] = mc_diag.get('poisson_mc_y_width_var', 0.0)
    store['block_norm_raw'][idx] = mc_diag.get('block_norm_raw', obs['norm'])
    if per_state_var.size:
        store['poisson_mc_pop_var'][:, idx] = per_state_var


def _copy_results_prefix(store: Dict[str, Any], nfill: int) -> Dict[str, Any]:
    out = {
        'nstates': store['nstates'],
        'ndim': store.get('ndim', 1),
        'region_names': list(store.get('region_names', [])),
    }
    for k, v in store.items():
        if k in {'nstates', 'ndim', 'region_names'}:
            continue
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                out[k] = v[:nfill].copy()
            elif v.ndim == 2:
                out[k] = v[:, :nfill].copy()
            else:
                out[k] = v.copy()
        else:
            out[k] = v
    return out


def _progress_row(percent: float, t: float, elapsed: float, block_count: int, obs: Dict[str, Any], lam: float, lam_raw: float, dt_blk: float, ess: float, w_hi: float, coup_exp: float, gap_exp_val: float, state_var: float, ndim: int, block_norm_raw: Optional[float] = None) -> Dict[str, float]:
    pops = obs['pops']
    row = {
        'percent': percent,
        'sim_time': t,
        'elapsed_seconds': elapsed,
        'blocks_completed': block_count,
        'pop1': float(pops[0]) if pops.size > 0 else 0.0,
        'pop2': float(pops[1]) if pops.size > 1 else 0.0,
        'norm': obs['norm'],
        'block_norm_raw': obs['norm'] if block_norm_raw is None else float(block_norm_raw),
        'R_total': obs['R_total'],
        'T_total': obs['T_total'],
        'x_mean': obs['x_mean'],
        'x_width': obs['x_width'],
        'lambda': lam,
        'lambda_raw': lam_raw,
        'block_dt': dt_blk,
        'ess': ess,
        'event_weight_hi': w_hi,
        'coupling_abs_exp': coup_exp,
        'gap_exp': gap_exp_val,
        'poisson_mc_state_var': state_var,
    }
    if ndim == 2:
        row['y_mean'] = obs.get('y_mean', 0.0)
        row['y_width'] = obs.get('y_width', 0.0)
    return row


def _write_model_preview_outputs(outdir: str, prefix: str, grid: Grid, pots: MultiStatePotentials, output_cfg: Dict[str, Any]) -> None:
    if grid.ndim == 1:
        write_model_profile_csv(os.path.join(outdir, f'{prefix}_model_profile.csv'), grid, pots)
        return
    if not bool(output_cfg.get('export_preview_csv', True)):
        return
    mode = str(output_cfg.get('preview_mode', 'cuts')).lower()
    if mode == 'flat':
        write_model_preview_flat_csv(os.path.join(outdir, f'{prefix}_model_flat.csv'), grid, pots)
    elif mode == 'cuts':
        write_model_cut_csv(os.path.join(outdir, f'{prefix}_model_cuts.csv'), grid, pots, output_cfg)
    elif mode == 'both':
        write_model_preview_flat_csv(os.path.join(outdir, f'{prefix}_model_flat.csv'), grid, pots)
        write_model_cut_csv(os.path.join(outdir, f'{prefix}_model_cuts.csv'), grid, pots, output_cfg)
    else:
        raise ValueError(f'Unsupported preview_mode: {mode}')


def run_v1_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATUS_ONCE['main']:
        hi_workers = _env_int('POISSON_V1_HI_WORKERS', _env_int('POISSON_V1_WORKERS', 1))
        replica_workers = _env_int('POISSON_V1_REPLICA_WORKERS', _env_int('POISSON_V1_WORKERS', 1))
        fft_workers = _env_int('POISSON_V1_FFT_WORKERS', 1)
        _print_runtime_status('MAIN', extra=f'HI_WORKERS={hi_workers} REPLICA_WORKERS={replica_workers} FFT_WORKERS={fft_workers}')
        _STATUS_ONCE['main'] = True
    grid = build_grid(config['grid'])
    pots = build_model_potentials(grid, config)
    base_evals, base_evecs, residual_edges = build_base_and_residual(pots, config)
    cap = build_cap(grid, config)
    _log_cap_status(cap)
    psi = initial_wavepacket(grid, config['initial_state'], pots.nstates)

    sim = config['simulation']
    algo = config['algorithm']
    analysis = config.get('analysis', {})
    wf_track_cfg = _parse_wf_error_tracking_cfg(config)
    if wf_track_cfg['enabled'] and not bool(sim.get('compute_exact', False)):
        raise ValueError('analysis.wavefunction_error_tracking requires simulation.compute_exact: true')
    tmax = float(sim['tmax'])
    rng = np.random.default_rng(int(sim.get('seed', 12345)))
    variance_batches = int(analysis.get('variance_batches', 8))
    radial_regions = _parse_radial_regions(config)
    radial_masks = _build_radial_masks(grid, radial_regions)

    nalloc = max(2, int(math.ceil(tmax / max(1e-12, float(algo.get('dt_min', 0.01))))) + 4)
    res = _build_result_storage(pots.nstates, nalloc, grid.ndim, [r.name for r in radial_regions])
    wf_diag = _build_wf_diag_storage(nalloc) if wf_track_cfg['enabled'] else None

    outdir = _resolve_path(config['output']['directory'], config)
    prefix = config['output'].get('prefix', 'run')
    partial_csv = os.path.join(outdir, f'{prefix}_poisson_v1_partial.csv')
    progress_csv = os.path.join(outdir, f'{prefix}_poisson_v1_progress.csv')
    _write_model_preview_outputs(outdir, prefix, grid, pots, config['output'])
    for p in [partial_csv, progress_csv]:
        if os.path.exists(p):
            os.remove(p)

    t = 0.0
    block_count = 0
    save_idx = 0
    progress_step = 0.05
    checkpoints = [min(1.0, i * progress_step) for i in range(1, int(round(1.0 / progress_step)) + 1)]
    next_cp_idx = 0
    dt_hist, lam_hist = [], []
    lam_edge_hist = []
    start = time.time()

    exact_psi = None
    exact_t = 0.0
    full_evals_exact = None
    full_evecs_exact = None
    exact_dt_track = None
    wf_sum_sq = 0.0
    if wf_track_cfg['enabled']:
        exact_psi = initial_wavepacket(grid, config['initial_state'], pots.nstates)
        full_evals_exact, full_evecs_exact = _precompute_eigh(pots.full_matrix)
        exact_dt_track = float(sim.get('exact_dt', algo.get('dt_min', 0.01)))
        print('[info] wavefunction error tracking enabled; exact companion propagation will run at Poisson block boundaries', flush=True)

    _PARALLEL_STATE['hi'] = {
        'grid': grid,
        'base_evals': base_evals,
        'base_evecs': base_evecs,
        'residual_edges': residual_edges,
        'cap': cap,
    }
    hi_executor, hi_parallel_workers = _make_process_pool('hi', int(algo.get('ntraj_hi', 0)))

    try:
        while t < tmax - 1e-12:
            if save_idx >= res['times'].size:
                grown = _build_result_storage(pots.nstates, res['times'].size * 2, grid.ndim, res['region_names'])
                for k, v in res.items():
                    if k in {'nstates', 'ndim', 'region_names'}:
                        continue
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            grown[k][:v.shape[0]] = v
                        elif v.ndim == 2:
                            grown[k][:, :v.shape[1]] = v
                res = grown
                if wf_diag is not None:
                    grown_wf = _build_wf_diag_storage(res['times'].size)
                    for k, v in wf_diag.items():
                        if isinstance(v, np.ndarray):
                            grown_wf[k][:v.shape[0]] = v
                    wf_diag = grown_wf

            dt_blk, lam, lam_raw, coup_exp, gap_exp_val, lam_edges = choose_block_dt(psi, pots, residual_edges, grid, algo, t, tmax)
            psi, ess, w_hi, mc_diag, per_state_var = propagate_block_v1(
                psi, grid, base_evals, base_evecs, residual_edges, dt_blk, lam, lam_edges, algo, rng,
                config, cap, variance_batches, radial_regions, radial_masks,
                hi_executor=hi_executor, hi_parallel_workers=hi_parallel_workers,
            )
            t += dt_blk
            block_count += 1
            obs = compute_observables(psi, grid, config, radial_regions=radial_regions, radial_masks=radial_masks)
            _snapshot_results(res, save_idx, t, obs, lam, lam_raw, dt_blk, ess, w_hi, coup_exp, gap_exp_val, mc_diag, per_state_var)
            wf_error_inst = None
            wf_error_cum_rms = None
            wf_fidelity = None
            if wf_diag is not None and exact_psi is not None and full_evals_exact is not None and full_evecs_exact is not None and exact_dt_track is not None:
                exact_psi, exact_t = _propagate_exact_to_time(exact_psi, exact_t, t, exact_dt_track, grid, full_evals_exact, full_evecs_exact, cap)
                wf_error_inst, wf_fidelity = _wavefunction_distance_and_fidelity(psi, exact_psi)
                wf_sum_sq += wf_error_inst * wf_error_inst
                wf_error_cum_rms = math.sqrt(max(0.0, wf_sum_sq / float(save_idx + 1)))
                _snapshot_wf_diag(wf_diag, save_idx, t, block_count, ess, wf_error_inst, wf_error_cum_rms, wf_fidelity)
            dt_hist.append(dt_blk)
            lam_hist.append(lam)
            lam_edge_hist.append(lam_edges.copy())
            save_idx += 1

            frac = t / tmax if tmax > 0 else 1.0
            while next_cp_idx < len(checkpoints) and frac >= checkpoints[next_cp_idx] - 1e-12:
                percent = checkpoints[next_cp_idx] * 100.0
                elapsed = time.time() - start
                state_var = mc_diag.get('poisson_mc_state_var', 0.0)
                log_progress('poisson_v1', percent, t, tmax, elapsed, block_count, obs, lam, lam_raw, dt_blk, ess, w_hi, coup_exp, gap_exp_val, state_var, grid.ndim, wf_error_inst=wf_error_inst if wf_track_cfg['progress_log'] else None, wf_error_cum_rms=wf_error_cum_rms if wf_track_cfg['progress_log'] else None, wf_fidelity=wf_fidelity if wf_track_cfg['progress_log'] else None)
                append_progress_row(progress_csv, _progress_row(
                    percent, t, elapsed, block_count, obs, lam, lam_raw, dt_blk, ess, w_hi,
                    coup_exp, gap_exp_val, state_var, grid.ndim,
                    block_norm_raw=mc_diag.get('block_norm_raw', obs['norm']),
                ), grid.ndim)
                write_timeseries_csv(partial_csv, _copy_results_prefix(res, save_idx))
                next_cp_idx += 1
    finally:
        if hi_executor is not None:
            hi_executor.shutdown(wait=True)
        _PARALLEL_STATE.pop('hi', None)

    out = _copy_results_prefix(res, save_idx)
    if wf_diag is not None:
        out['wf_diag'] = _slice_wf_diag(wf_diag, save_idx)

    if int(analysis.get('paper_variance_replicas', 0)) >= 2:
        print('[info] estimating paper-style variance replicas ...', flush=True)
        paper_var_pop, paper_var_norm = estimate_paper_style_population_variance(
            config=config,
            grid=grid,
            pots=pots,
            base_evals=base_evals,
            base_evecs=base_evecs,
            residual_edges=residual_edges,
            dt_hist=np.asarray(dt_hist, dtype=float),
            lam_hist=np.asarray(lam_hist, dtype=float),
            lam_edge_hist=np.asarray(lam_edge_hist, dtype=float),
            sample_times=out['times'],
            cap=cap,
        )
        out['paper_var_pop'] = paper_var_pop
        out['paper_var_norm'] = paper_var_norm

    return out


def run_exact_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    grid = build_grid(config['grid'])
    pots = build_model_potentials(grid, config)
    cap = build_cap(grid, config)
    _log_cap_status(cap)
    psi = initial_wavepacket(grid, config['initial_state'], pots.nstates)

    sim = config['simulation']
    tmax = float(sim['tmax'])
    dt = float(sim.get('exact_dt', config['algorithm']['dt_min']))
    radial_regions = _parse_radial_regions(config)
    radial_masks = _build_radial_masks(grid, radial_regions)

    full_evals, full_evecs = _precompute_eigh(pots.full_matrix)
    nsteps = max(1, int(math.ceil(tmax / max(1e-12, dt))))
    res = _build_result_storage(pots.nstates, nsteps, grid.ndim, [r.name for r in radial_regions])

    outdir = _resolve_path(config['output']['directory'], config)
    prefix = config['output'].get('prefix', 'run')
    partial_csv = os.path.join(outdir, f'{prefix}_exact_partial.csv')
    progress_csv = os.path.join(outdir, f'{prefix}_exact_progress.csv')
    _write_model_preview_outputs(outdir, prefix, grid, pots, config['output'])
    for p in [partial_csv, progress_csv]:
        if os.path.exists(p):
            os.remove(p)

    t = 0.0
    start = time.time()
    progress_step = 0.05
    checkpoints = [min(1.0, i * progress_step) for i in range(1, int(round(1.0 / progress_step)) + 1)]
    next_cp_idx = 0
    for istep in range(nsteps):
        cur_dt = min(dt, tmax - t)
        if cur_dt <= 0.0:
            break
        psi = propagate_exact_split(psi, grid, full_evals, full_evecs, cur_dt, cap)
        t += cur_dt
        obs = compute_observables(psi, grid, config, radial_regions=radial_regions, radial_masks=radial_masks)
        _snapshot_results(
            res, istep, t, obs,
            lam=0.0, lam_raw=0.0, dt_blk=cur_dt, ess=0.0, w_hi=0.0,
            coup_exp=float(np.mean([np.mean(np.abs(e.values)) for e in pots.edges])) if pots.edges else 0.0,
            gap_exp_val=float(np.mean(pots.min_gap)),
            mc_diag={
                'poisson_mc_state_var': 0.0, 'poisson_mc_R_var': 0.0, 'poisson_mc_T_var': 0.0,
                'poisson_mc_x_mean_var': 0.0, 'poisson_mc_width_var': 0.0,
                'poisson_mc_y_mean_var': 0.0, 'poisson_mc_y_width_var': 0.0,
            },
            per_state_var=np.zeros(pots.nstates, dtype=float),
        )
        frac = t / tmax if tmax > 0 else 1.0
        while next_cp_idx < len(checkpoints) and frac >= checkpoints[next_cp_idx] - 1e-12:
            percent = checkpoints[next_cp_idx] * 100.0
            elapsed = time.time() - start
            log_progress('exact', percent, t, tmax, elapsed, istep + 1, obs, 0.0, 0.0, cur_dt, 0.0, 0.0, 0.0, float(np.mean(pots.min_gap)), 0.0, grid.ndim)
            append_progress_row(progress_csv, _progress_row(percent, t, elapsed, istep + 1, obs, 0.0, 0.0, cur_dt, 0.0, 0.0, 0.0, float(np.mean(pots.min_gap)), 0.0, grid.ndim), grid.ndim)
            write_timeseries_csv(partial_csv, _copy_results_prefix(res, istep + 1))
            next_cp_idx += 1
    return _copy_results_prefix(res, min(nsteps, int(np.sum(res['times'] > 0))))


def write_wavefunction_error_csv(path: str, wf_diag: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cols = ['time', 'block_index', 'ess', 'wf_error_inst', 'wf_error_cum_rms', 'wf_fidelity']
    times = np.asarray(wf_diag['times'])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(times.size):
            writer.writerow([
                f"{float(wf_diag['times'][i]):.12g}",
                str(int(wf_diag['block_index'][i])),
                f"{float(wf_diag['ess'][i]):.12g}",
                f"{float(wf_diag['wf_error_inst'][i]):.12g}",
                f"{float(wf_diag['wf_error_cum_rms'][i]):.12g}",
                f"{float(wf_diag['wf_fidelity'][i]):.12g}",
            ])
    return path


def save_results(outdir: str, prefix: str, results: Dict[str, Any]) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'{prefix}.csv')
    write_timeseries_csv(path, results)
    wf_diag = results.get('wf_diag')
    if isinstance(wf_diag, dict):
        wf_path = os.path.join(outdir, f'{prefix}_wf_error.csv')
        write_wavefunction_error_csv(wf_path, wf_diag)
    return path
