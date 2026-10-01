#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml

from tully_poisson_v1 import run_exact_benchmark, run_v1_simulation, save_results




def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_dir'] = os.path.dirname(os.path.abspath(path))
    return cfg


def resolve_output_dir(cfg: Dict[str, Any]) -> str:
    outdir = cfg['output']['directory']
    if os.path.isabs(outdir):
        return outdir
    cfg_dir = cfg.get('_config_dir')
    if cfg_dir:
        return os.path.abspath(os.path.join(cfg_dir, outdir))
    return os.path.abspath(outdir)


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description='Blockwise stratified Poisson v1 solver for 1D/2D multistate diabatic models.')
    parser.add_argument('config', help='Path to YAML config file.')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    outdir = resolve_output_dir(cfg)
    prefix = cfg['output'].get('prefix', os.path.splitext(os.path.basename(args.config))[0])
    run_poisson = bool(cfg.get('simulation', {}).get('run_poisson', True))
    run_exact = bool(cfg.get('simulation', {}).get('compute_exact', False))

    print('[info] running multistate v1 Poisson solver', flush=True)
    print(json.dumps({
        'model': cfg['model'],
        'simulation': cfg['simulation'],
        'grid': cfg['grid'],
        'algorithm': cfg['algorithm'],
        'output_directory': outdir,
    }, indent=2), flush=True)

    if run_poisson:
        res_v1 = run_v1_simulation(cfg)
        path_v1 = save_results(outdir, prefix + '_poisson_v1', res_v1)
        print(f'[done] saved v1 results -> {path_v1}', flush=True)

    if run_exact:
        print('[info] running exact split-operator benchmark', flush=True)
        res_exact = run_exact_benchmark(cfg)
        path_exact = save_results(outdir, prefix + '_exact', res_exact)
        print(f'[done] saved exact results -> {path_exact}', flush=True)


if __name__ == '__main__':
    main()
