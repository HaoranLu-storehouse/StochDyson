
from __future__ import annotations
import json,csv,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'
FIG=ROOT/'figures_slow_final'; FIG.mkdir(exist_ok=True)
Ls=[0,16,32,48,64]
colors=plt.cm.viridis(np.linspace(0.05,0.9,len(Ls)))
D={L:pd.read_csv(RES/'figure5_200fs'/f'L{L:02d}'/'timeseries.csv') for L in Ls}

# 5a schematic
fig=plt.figure(figsize=(9.2,4.5)); gs=fig.add_gridspec(1,2,width_ratios=[1.1,1])
ax=fig.add_subplot(gs[0,0])
ener=np.linspace(-0.1,0.1,41); active=np.argsort(abs(ener+0.0208))[:7]
for i,e in enumerate(ener):
    if i in active: ax.hlines(e,0.44,0.98,linewidth=2.0)
    else: ax.hlines(e,0.56,0.98,linewidth=1.0,alpha=.5)
ax.hlines(-0.0208,0.0,0.28,linewidth=3.0); ax.text(0.14,-0.017,'|d>',ha='center',va='bottom')
for e in ener[::3]: ax.plot([0.28,0.56],[-0.0208,e],linewidth=.35,alpha=.35)
ax.text(.77,.108,'metallic continuum',ha='center'); ax.text(.75,-.005,'active near-resonant states',ha='center',fontsize=9)
ax.set_xlim(-.05,1.02);ax.set_ylim(-.115,.125);ax.set_ylabel('Electronic energy (Ha)');ax.set_xticks([]);ax.set_title('(a) Newns–Anderson state manifold')
ax=fig.add_subplot(gs[0,1]); n=40; M=np.zeros((n+1,n+1));M[0,1:]=1;M[1:,0]=1;np.fill_diagonal(M,.35)
ax.imshow(M,origin='lower',cmap='Greys',vmin=0,vmax=1,interpolation='nearest');ax.axhline(8.5,linewidth=1);ax.axvline(8.5,linewidth=1);ax.set_xlabel('Electronic-state index');ax.set_ylabel('Electronic-state index');ax.set_title('Hamiltonian pattern')
fig.tight_layout();fig.savefig(FIG/'Figure5a_model_schematic.png',dpi=220);plt.close(fig)

# 5b population
fig,ax=plt.subplots(figsize=(7.4,5.2))
d0=D[0];ax.plot(d0.time_fs,d0.population_exact,'k--',lw=2.2,label='Exact')
for c,L in zip(colors,Ls): ax.plot(D[L].time_fs,D[L].population_d,lw=1.65,label=f'L={L}',color=c)
ax.set_xlabel('Time (fs)');ax.set_ylabel('Molecular-state population');ax.set_xlim(0,200);ax.set_ylim(0,1.02);ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(FIG/'Figure5b_population_slow_200fs.png',dpi=220);plt.close(fig)

# 5c lambda & ESS smooth 5-point
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7.6,6.7),sharex=True)
for c,L in zip(colors,Ls):
    d=D[L]; lam=d.lambda_au_inv.rolling(5,center=True,min_periods=1).mean(); ess=d.ess_normalized.rolling(7,center=True,min_periods=1).mean()
    ax1.plot(d.time_fs,lam,lw=1.5,color=c,label=f'L={L}'); ax2.plot(d.time_fs,ess,lw=1.5,color=c)
ax1.set_ylabel(r'$\lambda_b$ (a.u.$^{-1}$)');ax1.legend(frameon=False,ncol=3);ax1.set_ylim(bottom=0)
ax2.set_ylabel('Normalized ESS');ax2.set_xlabel('Time (fs)');ax2.set_ylim(0.7,1.01);ax2.set_xlim(0,200)
fig.tight_layout();fig.savefig(FIG/'Figure5c_lambda_ESS_slow_200fs.png',dpi=220);plt.close(fig)

# 5d cumulative RMS
fig,ax=plt.subplots(figsize=(7.4,5.2))
for c,L in zip(colors,Ls): ax.plot(D[L].time_fs,D[L].cumulative_rms_wavefunction_error,lw=1.7,color=c,label=f'L={L}')
ax.set_xlabel('Time (fs)');ax.set_ylabel('Cumulative RMS wavefunction error');ax.set_xlim(0,200);ax.set_ylim(bottom=0);ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(FIG/'Figure5d_cumulative_RMS_slow_200fs.png',dpi=220);plt.close(fig)

# scaling
scdir=RES/'scaling'; scdir.mkdir(exist_ok=True)
with (scdir/'cpu_scaling_L_vs_N.csv').open() as f:
    rows=[]
    for r in csv.DictReader(f):
        rr={k:float(v) for k,v in r.items()}; rr['N']=int(rr['N']); rr['L']=int(rr['L']); rows.append(rr)
rows=sorted(rows,key=lambda r:(r['L'],r['N']))
# shared asymptotic x with per-L offset and prefactor
idx={L:i for i,L in enumerate(Ls)}; nL=len(Ls);p0=np.array([1.0]+[.3]*nL+[.008]*nL)
def fun(par):
    x=par[0];cc=par[1:1+nL];aa=par[1+nL:];out=[]
    for r in rows:
        i=idx[r['L']];pred=cc[i]+aa[i]*r['N']**x;out.append((pred-r['stochastic_cpu_s'])/r['stochastic_cpu_s'])
    return np.array(out)
sol=least_squares(fun,p0,bounds=(np.array([.3]+[0]*nL+[0]*nL),np.array([1.5]+[20]*nL+[1]*nL)),max_nfev=50000);par=sol.x;x=float(par[0])
fit={'x':x,'offsets':{str(L):float(par[1+idx[L]]) for L in Ls},'prefactors':{str(L):float(par[1+nL+idx[L]]) for L in Ls},'relative_rmse':float(np.sqrt(np.mean(fun(par)**2)))}
(scdir/'cpu_scaling_offset_power_fit.json').write_text(json.dumps(fit,indent=2))
fig,ax=plt.subplots(figsize=(7.5,5.4))
for c,L in zip(colors,Ls):
    rr=[r for r in rows if r['L']==L];xx=np.array([r['N'] for r in rr]);yy=np.array([r['stochastic_cpu_s'] for r in rr]);ax.loglog(xx,yy,'o-',lw=1.6,ms=4.5,color=c,label=f'L={L}')
r0=next(r for r in rows if r['L']==16 and r['N']==800);Nguide=np.array(sorted({r['N'] for r in rows}),float);cg=r0['stochastic_cpu_s']/(800.0**x);ax.loglog(Nguide,cg*Nguide**x,'k--',lw=2.0,label=fr'$N^{{{x:.2f}}}$')
ax.set_xlabel('Number of metallic states, N');ax.set_ylabel('CPU time (s)');ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(FIG/'Figure5e_CPU_vs_N_slow_multiL.png',dpi=220);plt.close(fig)
print(json.dumps(fit,indent=2))
