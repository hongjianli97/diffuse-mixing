"""Generate a standalone scientific figure from saved reconstructed arrays."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);data=json.loads((root/'results.json').read_text())
names={'ordinary_volume_center':'Ordinary FV: geometric center','ordinary_mass_center':'Ordinary FV: mass midpoint','mass_average_resistance':'Mass-averaged resistance'}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for method,label in names.items():
 rows=[r for r in data['candidate_runs'] if r['method']==method]
 axes[0].plot([r['steps'] for r in rows],[r['normalized_error'] for r in rows],'o-',label=label)
 with np.load(root/f'{method}-k512.npz') as z:
  mu=(z['mass_edges'][:-1]+z['mass_edges'][1:])/2/z['mass_edges'][-1]
  axes[1].plot(mu,z['c']-z['reference'],'o-',markersize=3,label=label)
axes[0].set(xlabel='Backward Euler steps',ylabel='Mass-weighted error / initial fluctuation norm',title='One fixed 32-cell Becker case')
axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[0].legend(fontsize=8)
axes[1].set(xlabel='Material mass coordinate / total mass',ylabel='Cell-average concentration error',title='Signed errors at 512 steps')
axes[1].axhline(0,color='gray',lw=.7)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
