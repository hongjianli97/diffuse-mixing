"""Direct exchange and parent-average errors on fixed local layer meshes."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);r=json.loads((root/'results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
old=r['previous_65_child_runs'];axes[0].loglog([x['time'] for x in old],[100*x['right_phase_inventory_relative_error'] for x in old],'ks-',label='Previous 65 children')
axes[1].loglog([x['time'] for x in old],[x['aggregated_error'] for x in old],'ks-',label='Previous 65 children')
for q in (2,4,8):
 rows=[x for x in r['runs'] if x['resolution']==q];label=f"q={q}, {rows[0]['cells']} cells"
 axes[0].loglog([x['time'] for x in rows],[100*x['right_inventory_absolute_relative_error'] for x in rows],'o-',label=label)
 axes[1].loglog([x['time'] for x in rows],[x['aggregated_error'] for x in rows],'o-',label=label)
endpoints=[x for x in r['runs'] if x['resolution']==8 and 'BE_check' in x]
axes[0].scatter([x['time'] for x in endpoints],[100*abs(x['BE_check']['1024']['right_inventory_total_relative_error']) for x in endpoints],facecolors='none',edgecolors='C3',s=100,label='q=8, BE1024 endpoint checks')
axes[0].set(xlabel='Time',ylabel='Whole right-inventory error (%)',title='Known contact: directly observable exchange')
axes[1].set(xlabel='Time',ylabel='Error / parent initial fluctuation norm',title='Same original 64-cell observation grid')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(a.output,dpi=180)
