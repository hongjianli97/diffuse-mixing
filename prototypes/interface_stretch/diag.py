import numpy as np, pickle
import cases_mix as CM, stretch_est as SE, lhdx as X
def detF_comp(c, frames_list):
    FF=SE.chained_F(c['frames'],c['h'],bad=c['bad'],block=5,keep=frames_list,idx=c['band'])
    return {s: float(np.mean(-np.log(np.abs(np.linalg.det(FF[s][0]))))) for s in frames_list}
for nm,fn,kw,fl in (('sod',CM.setup_sod,[{}, {'noise':0.1}],[40]),('blast',CM.setup_blast,[{},{'noise':0.1}],[15,60]),('sb',lambda **k: CM.setup_dns('sb',**k),[{},{'noise':0.05},{'noise':0.1}],[30,70])):
    for k in kw:
        c=fn(**k); dF=detF_comp(c,fl)
        print(nm,k,{s:('ledger %.3f detF %.3f ref %.3f'%(np.mean(-c['lnJ'][s][c['band']]),dF[s], (np.mean(c['comp_per'][s]) if 'comp_per' in c else c['comp_exact'](s)))) for s in fl})
# SB: ledger vs DNS compression as a function of distance from the interface
c=CM.setup_dns('sb'); rho=c['D']['rho'].astype(float)
for lo_,hi_ in ((-0.06,-0.03),(-0.03,-0.015),(-0.015,-0.005),(-0.003,0.003),(0.005,0.015),(0.015,0.03),(0.03,0.06)):
    idx=np.where((c['phi0']>lo_)&(c['phi0']<hi_))[0]
    print('phi0 in [%.3f,%.3f] n=%d'%(lo_,hi_,len(idx)),' '.join('t=%.1f ledger %.3f dns %.3f'%(c['t'][s],np.mean(-c['lnJ'][s][idx]),np.mean(np.log(rho[s][idx]/rho[0][idx]))) for s in (30,70)))
