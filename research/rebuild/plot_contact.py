"""Plot exact-reference contact studies, including the fixed two-mode control."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);d=json.loads((root/'results.json').read_text());control=json.loads((root/'two-mode-control/results.json').read_text())
names={'ordinary_volume_center':'Geometric midpoint','ordinary_mass_center':'Mass midpoint','mass_average_resistance':'Average resistance'}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for method,name in names.items():
 row=d['cut_cell_spatial_convergence'][method]
 axes[0].loglog(row['cells'],row['errors'],'o-',label=name)
axes[0].set_xticks([32,64,128],labels=['32','64','128']);axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set(xlabel='Material cells',ylabel='Semidiscrete error / initial fluctuation norm',title='Contact inside a material cell: single mode')
axes[0].legend(fontsize=8)
x=np.arange(3)
first=[next(r['spatial_error'] for r in d['runs'] if r['placement']=='cut' and r['cells']==64 and r['method']==m) for m in names]
second=[next(r['spatial_error'] for r in control['runs'] if r['method']==m) for m in names]
axes[1].bar(x-.18,first,.36,label='Single mode');axes[1].bar(x+.18,second,.36,label='Two compatible modes')
axes[1].set(xticks=x,xticklabels=['Geometric\nmidpoint','Mass\nmidpoint','Average\nresistance'],yscale='log',ylabel='Error / own initial fluctuation norm',title='Same 64-cell grid; scalar initial field changes')
axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.15)
fig.savefig(a.output,dpi=180)
