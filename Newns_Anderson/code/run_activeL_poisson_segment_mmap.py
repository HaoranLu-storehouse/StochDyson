
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np,yaml
from run_activeL_poisson_diagnostics import PoissonDiagnosticSimulator, phase_aligned_error
from newns_anderson_linear import FS_TO_AU, AU_TO_FS, HybridSettings, model_from_config, group_observables, state_observables


def _save_rng(path,rngs): path.write_text(json.dumps([r.bit_generator.state for r in rngs]),encoding='utf-8')
def _load_rng(path,rngs):
    states=json.loads(path.read_text(encoding='utf-8'))
    for r,s in zip(rngs,states): r.bit_generator.state=s

def run_segment(cfg_path:Path,L:int,workdir:Path,segment_fs:float,ref_states_path:Path):
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8')); cfg=json.loads(json.dumps(cfg)); cfg['hybrid']['active_L']=L
    model=model_from_config(cfg); settings=HybridSettings(**cfg['hybrid']); d=cfg.get('poisson_diagnostics',{})
    sim=PoissonDiagnosticSimulator(model,settings,float(d.get('lambda_scale',8.0)),float(d.get('mu_min',0.05)),float(d.get('mu_max',0.8)))
    case=workdir/f'L{L:02d}'; case.mkdir(parents=True,exist_ok=True)
    statep=case/'checkpoint.npz'; rngp=case/'rng.json'; csvp=case/'timeseries.csv'
    dt=settings.block_fs*FS_TO_AU; out_every=max(1,int(round(settings.output_fs/settings.block_fs)))
    ref_states=np.load(ref_states_path,mmap_mode='r')
    expected=(int(round(settings.tmax_fs/settings.output_fs))+1, model.nelectronic, model.nbasis)
    if tuple(ref_states.shape)!=expected: raise ValueError(f'reference shape {ref_states.shape} != {expected}')
    if statep.exists():
        z=np.load(statep); groups=z['groups']; block0=int(z['block']); stoch=float(z['stoch']); sq=float(z['sq']); nerr=int(z['nerr']); _load_rng(rngp,sim.rngs)
    else:
        groups=np.repeat(model.psi0[None,:,:],settings.groups,axis=0); block0=0; stoch=sq=0.0; nerr=0
        obs=group_observables(groups,model); ro=state_observables(np.asarray(ref_states[0]),model)
        fields=['time_fs','lambda_au_inv','lambda_fs_inv','mu','residual_strength_au','ess_normalized','mean_poisson_order','max_poisson_order','population_d','population_exact','norm_u','group_population_std','phase_aligned_error','cumulative_rms_wavefunction_error','cpu_elapsed_s']
        with csvp.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader();w.writerow(dict(time_fs=0.0,lambda_au_inv=0.0,lambda_fs_inv=0.0,mu=0.0,residual_strength_au=sim.residual_transition_strength(groups.mean(0)),ess_normalized=1.0,mean_poisson_order=0.0,max_poisson_order=0,population_d=obs['population_d_u'],population_exact=ro['population_d_exact'],norm_u=obs['norm_u'],group_population_std=obs['group_population_std'],phase_aligned_error=0.0,cumulative_rms_wavefunction_error=0.0,cpu_elapsed_s=0.0))
    total_blocks=int(round(settings.tmax_fs/settings.block_fs)); seg_blocks=int(round(segment_fs/settings.block_fs)); end=min(total_blocks,block0+seg_blocks)
    for b in range(block0+1,end+1):
        rate=sim.set_block_rate(groups.mean(0),dt); t=time.perf_counter()
        for g in range(settings.groups): groups[g]=sim.propagate_group_block(groups[g],dt,sim.rngs[g])
        stoch+=time.perf_counter()-t; ess=sim.block_ess(); mo,xo=sim.block_order_stats()
        if b%out_every==0 or b==total_blocks:
            oi=int(round((b*settings.block_fs)/settings.output_fs)); ref=np.asarray(ref_states[oi]); obs=group_observables(groups,model); ro=state_observables(ref,model); err=phase_aligned_error(groups.mean(0),ref); sq+=err*err;nerr+=1
            rec=dict(time_fs=b*settings.block_fs,lambda_au_inv=rate['lambda_au_inv'],lambda_fs_inv=rate['lambda_au_inv']/AU_TO_FS,mu=rate['mu'],residual_strength_au=rate['residual_strength_au'],ess_normalized=ess,mean_poisson_order=mo,max_poisson_order=xo,population_d=obs['population_d_u'],population_exact=ro['population_d_exact'],norm_u=obs['norm_u'],group_population_std=obs['group_population_std'],phase_aligned_error=err,cumulative_rms_wavefunction_error=math.sqrt(sq/nerr),cpu_elapsed_s=stoch)
            with csvp.open('a',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=list(rec.keys())).writerow(rec)
    np.savez_compressed(statep,groups=groups,block=end,stoch=stoch,sq=sq,nerr=nerr); _save_rng(rngp,sim.rngs)
    print(json.dumps({'L':L,'block':end,'time_fs':end*settings.block_fs,'stochastic_cpu_s':stoch,'complete':end==total_blocks},indent=2),flush=True)
    if end==total_blocks:
        rows=list(csv.DictReader(csvp.open(encoding='utf-8'))); arr=lambda k:np.array([float(r[k]) for r in rows]); lam=arr('lambda_au_inv')[1:]; ess=arr('ess_normalized')[1:]; pop=arr('population_d'); pref=arr('population_exact'); cr=arr('cumulative_rms_wavefunction_error'); norm=arr('norm_u')
        summary={'L':L,'N':model.nmetal,'K':model.nbasis,'residual_states':int(sim.residual.size),'tmax_fs':settings.tmax_fs,'block_fs':settings.block_fs,'groups':settings.groups,'replicas_per_group':settings.replicas_per_group,'first_order_samples_per_residual':settings.first_order_samples_per_residual,'first_order_samples_per_replica':sim.first_order_sample_count(),'time_quadrature_order':settings.time_quadrature_order,'high_order_samples_per_replica':settings.high_order_samples,'high_order_paths_per_block':settings.groups*settings.replicas_per_group*settings.high_order_samples,'lambda_mean_au_inv':float(lam.mean()),'lambda_min_au_inv':float(lam.min()),'lambda_max_au_inv':float(lam.max()),'mu_mean':float(arr('mu')[1:].mean()),'ess_mean':float(ess.mean()),'ess_min':float(ess.min()),'fraction_ess_gt_075':float(np.mean(ess>=.75)),'population_rms':float(np.sqrt(np.mean((pop-pref)**2))),'population_max_abs':float(np.max(np.abs(pop-pref))),'final_cumulative_rms_wavefunction_error':float(cr[-1]),'max_cumulative_rms_wavefunction_error':float(cr.max()),'max_abs_norm_error':float(np.max(np.abs(norm-1))),'stochastic_cpu_s':stoch,'csv':str(csvp)}
        (case/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--L',type=int,required=True);a.add_argument('--workdir',type=Path,required=True);a.add_argument('--segment-fs',type=float,default=25.0);a.add_argument('--reference-states',type=Path,required=True);x=a.parse_args();run_segment(x.config,x.L,x.workdir,x.segment_fs,x.reference_states)
