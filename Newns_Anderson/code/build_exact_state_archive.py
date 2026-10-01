
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, yaml
from newns_anderson_linear import model_from_config, ChebyshevPropagator, FS_TO_AU, state_observables


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--outdir',type=Path,required=True); ap.add_argument('--output-fs',type=float,default=1.0); args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8')); model=model_from_config(cfg); tmax=float(cfg['hybrid']['tmax_fs']); dt=args.output_fs*FS_TO_AU; nt=int(round(tmax/args.output_fs))+1
    args.outdir.mkdir(parents=True,exist_ok=True)
    path=args.outdir/'exact_states.npy'; arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.complex128,shape=(nt,model.nelectronic,model.nbasis)); psi=model.psi0.copy(); arr[0]=psi
    prop=ChebyshevPropagator(model.build_full_hamiltonian(),tol=2e-12)
    rows=[]; o=state_observables(psi,model); rows.append({'time_fs':0.0,**o})
    t0=time.perf_counter()
    for i in range(1,nt):
        psi=prop.apply(psi,dt); arr[i]=psi; o=state_observables(psi,model); rows.append({'time_fs':i*args.output_fs,**o})
    arr.flush(); elapsed=time.perf_counter()-t0
    import csv
    with (args.outdir/'exact_observables.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    meta={'N':model.nmetal,'K':model.nbasis,'tmax_fs':tmax,'output_fs':args.output_fs,'elapsed_s':elapsed,'shape':[nt,model.nelectronic,model.nbasis],'state_file':str(path)}
    (args.outdir/'exact_archive_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
