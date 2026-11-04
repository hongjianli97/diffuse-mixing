"""One Becker case: geometric validity and matched FV scalar comparisons."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
run=Path(a.run);r=json.loads((run/'results.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
with np.load(run/'geometry-audit.npz') as z:
    t=z['time'];true=z['material_faces'];dual=z['dual_faces'];m=z['mass']
    vt=np.diff(true,axis=1);vd=np.diff(dual,axis=1);actual=z['true_fluid_mass_in_dual_cells']
    rho=z['material_average_density'];proxy=z['dual_density_proxy'];labels=z['point_labels']
axes[0,0].plot(t,100*np.max(abs(vd-vt)/vt,axis=1),label='Volume proxy versus material volume')
axes[0,0].plot(t,100*np.max(abs(actual-m)/m,axis=1),label='Actual dual-cell mass versus stored m')
axes[0,0].set(xlabel='Time',ylabel='Maximum relative mismatch (%)',title='129 saved geometry audit frames')
axes[0,0].legend(fontsize=7)
axes[0,1].plot(labels,rho[-1],'o-',label='m / true material volume')
axes[0,1].plot(labels,proxy[-1],'s--',label='m / midpoint-dual volume')
axes[0,1].set(xlabel='Initial cell label',ylabel='Cell density / proxy',title='T=0.1, same fixed mass identities')
axes[0,1].legend(fontsize=7)
names=['Oracle','True faces\nmean rho','True faces\ndual rho*','Dual faces\ndual rho*']
rows=r['final_comparison'];x=np.arange(len(rows))
axes[1,0].bar(x,[q['normalized_error'] for q in rows],color=['black','tab:blue','tab:orange','tab:red'])
axes[1,0].set(xticks=x,xticklabels=names,ylabel='Error / initial mass fluctuation norm',
              title='* diagnostic proxies, not accepted material cells')
axes[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
with np.load(run/'change-chain.npz') as z:
    e=z['mass_edges'];mu=(e[:-1]+e[1:])/2;ref=z['reference']
    for key,label in zip([q['method'] for q in rows],['Oracle','True-face mean density','Dual density only*','Dual geometry*']):
        axes[1,1].plot(mu,z[key]-ref,'o-',markersize=2,label=label)
axes[1,1].set(xlabel='Fixed material mass coordinate',ylabel='Signed concentration error',title='Same m, c0 and independent reference')
axes[1,1].legend(fontsize=7)
for ax in axes.flat:ax.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
