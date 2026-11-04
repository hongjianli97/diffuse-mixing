"""Standalone spatial-convergence and uncertainty diagnostic figure."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);d=json.loads((root/'results.json').read_text())
names={'ordinary_volume_center':'Geometric midpoint','ordinary_mass_center':'Mass midpoint','mass_average_resistance':'Average resistance'}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for method,name in names.items():
 c=d['observed_spatial_convergence'][method]
 axes[0].loglog(c['columns'],c['errors'],'o-',label=name)
 rows=[r for r in d['candidate_runs'] if r['method']==method and r['steps']==8192]
 axes[1].plot([r['material_columns'] for r in rows],[100*r['combined_indicator_over_error'] for r in rows],'o-',label=name)
axes[0].set(xlabel='Material columns (same physical case)',ylabel='Error / initial fluctuation norm',title='Three spatial resolutions, 8192 time steps')
axes[0].set_xticks([32,64,128],labels=['32','64','128']);axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());axes[0].legend(fontsize=8)
axes[1].axhline(10,color='gray',linestyle='--',label='10% empirical separation diagnostic')
axes[1].set(xlabel='Material columns',ylabel='Combined uncertainty indicator / error (%)',title='Time and reference separation')
axes[1].set_xticks([32,64,128]);axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
