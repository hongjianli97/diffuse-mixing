"""Early layer and inventory observability, using saved results only."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);r=json.loads((root/'results.json').read_text())
names={'ordinary_volume_center':'Geometric midpoint','ordinary_mass_center':'Mass midpoint','mass_average_resistance':'Average resistance'}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for method,name in names.items():
 rows=[x for x in r['runs'] if x['placement']=='cut' and x['cells']==64 and x['method']==method]
 axes[0].loglog([x['time'] for x in rows],[x['spatial_error'] for x in rows],'o-',label=name)
axes[0].set(xlabel='Time',ylabel='Error / initial cell-average fluctuation norm',title='Initial step: 64 cut cells')
axes[0].legend(fontsize=8)
rows=[x for x in r['runs'] if x['placement']=='cut' and x['cells']==64 and x['method']=='mass_average_resistance']
t=np.array([x['time'] for x in rows]);bounds=np.array([x['whole_right_phase_state_inventory_bounds'] for x in rows])
axes[1].fill_between(t,bounds[:,0],bounds[:,1],alpha=.2,label='State identifiability range (not error bars)')
axes[1].semilogx(t,[x['whole_right_phase_exact_mass'] for x in rows],'k-o',label='True whole right-phase inventory')
axes[1].semilogx(t,[x['whole_right_phase_uniform_reconstruction'] for x in rows],'o-',label='Uniform subcell reconstruction')
initial=next(x['uniform_subcell_reconstruction_initial_right_mass'] for x in r['initial_projection'] if x['placement']=='cut' and x['cells']==64)
axes[1].axhline(initial,color='gray',linestyle='--',label='False initial inventory from reconstruction')
axes[1].set(xlabel='Time',ylabel='Scalar mass per unit transverse area',title='One straddling-cell average loses subcell inventory')
axes[1].legend(fontsize=7)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
