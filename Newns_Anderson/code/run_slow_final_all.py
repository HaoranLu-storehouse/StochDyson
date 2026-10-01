from __future__ import annotations
import argparse, subprocess, sys, shutil, json, csv
from pathlib import Path

def call(args):
    print(' '.join(map(str,args)), flush=True)
    subprocess.check_call(list(map(str,args)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--segment-fs',type=float,default=50.0)
    args=ap.parse_args()
    root=args.root.resolve(); py=sys.executable
    cfg=root/'configs/N800_slow_group1.yaml'
    exact_work=root/'work/exact'; dyn_work=root/'work/figure5_200fs'
    exact_work.mkdir(parents=True,exist_ok=True); dyn_work.mkdir(parents=True,exist_ok=True)
    states=exact_work/'exact_states.npy'
    if not states.exists():
        call([py,root/'code/build_exact_state_archive.py','--config',cfg,'--outdir',exact_work,'--output-fs','1.0'])
    for L in [0,16,32,48,64]:
        case=dyn_work/f'L{L:02d}'
        while not (case/'summary.json').exists():
            call([py,root/'code/run_activeL_poisson_segment_mmap.py','--config',cfg,'--L',L,'--workdir',dyn_work,'--segment-fs',args.segment_fs,'--reference-states',states])
    # collect 200-fs results
    res=root/'results/figure5_200fs'; res.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for L in [0,16,32,48,64]:
        src=dyn_work/f'L{L:02d}'; dst=res/f'L{L:02d}'; dst.mkdir(parents=True,exist_ok=True)
        for fn in ['timeseries.csv','summary.json']:
            shutil.copy2(src/fn,dst/fn)
        summaries.append(json.loads((src/'summary.json').read_text()))
    with (res/'L_sweep_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0].keys()));w.writeheader();w.writerows(summaries)
    # copy exact observable archive metadata (not the large state file)
    ev=root/'results/exact_validation'; ev.mkdir(parents=True,exist_ok=True)
    for fn in ['exact_observables.csv','exact_archive_meta.json']:
        if (exact_work/fn).exists(): shutil.copy2(exact_work/fn,ev/fn)
    # N scaling
    sc=root/'results/scaling'; sc.mkdir(parents=True,exist_ok=True)
    call([py,root/'code/run_activeL_scaling_sweep.py','--config',root/'configs/N_slow_scaling5.yaml','--outdir',sc,'--L',0,16,32,48,64,'--N',200,400,800,1200,1600,2000,'--tmax-fs',5])
    call([py,root/'code/make_slow_final_figures.py'])
    print('Completed. Final figures are in figures_slow_final/.', flush=True)
if __name__=='__main__': main()
