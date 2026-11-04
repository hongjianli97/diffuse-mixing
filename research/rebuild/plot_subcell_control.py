"""Fixed subcells: parent accuracy and directly observable right inventory."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);r=json.loads((root/'results.json').read_text());rows=[x for x in r['runs'] if x['parent_cells']==64];t=np.array([x['time'] for x in rows])
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
names={'ordinary_volume_center':'Original geometric midpoint','ordinary_mass_center':'Original mass midpoint','mass_average_resistance':'Original average resistance'}
for method,name in names.items():axes[0].loglog(t,[x['comparisons_to_original'][method]['old_parent_error'] for x in rows],'o-',label=name)
axes[0].loglog(t,[x['aggregated_error'] for x in rows],'ks-',label='65 fixed children, aggregated')
axes[0].set(xlabel='Time',ylabel='Error / parent initial fluctuation norm',title='Adding subcell information is not uniformly better')
with np.load(root/'partition-parent64.npz') as z:
 m=z['child_mass'];right=z['child_faces'][:-1]>=0
homogeneous=[]
for time in t:
 with np.load(root/f'result-parent64-t{time:g}.npz') as z:homogeneous.append(float(m[right]@z['ordinary_volume_center_homogenized_child_solution'][right]))
axes[1].semilogx(t,[x['right_phase_inventory_exact'] for x in rows],'k-o',label='Exact scalar-step reference')
axes[1].semilogx(t,[x['right_phase_inventory'] for x in rows],'o-',label='Same child operator: retained step')
axes[1].semilogx(t,homogeneous,'o--',label='Same child operator: homogenized initial cell')
axes[1].set(xlabel='Time',ylabel='Whole right-phase scalar mass / area',title='Inventory identifiable; early layer still underresolved')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(a.output,dpi=180)
