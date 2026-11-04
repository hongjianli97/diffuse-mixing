"""Time-error budgets and actual single-path physical observables."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);r=json.loads((root/'results.json').read_text());path=json.loads((root/'single-path/results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for t in sorted({x['time'] for x in r['forecasts']}):
 rows=[x for x in r['forecasts'] if x['time']==t]
 axes[0].loglog([x['steps'] for x in rows],[100*x['maximum_ratio'] for x in rows],'o-',label=f't={t:g}')
axes[0].axhline(10,color='gray',linestyle='--',label='10% numerical separation target')
axes[0].set(xlabel='Uniform BE steps to each horizon',ylabel='Maximum time / spatial error contribution (%)',title='Fixed 436-cell mesh; offline modal forecast')
axes[0].legend(fontsize=7)
t=[x['time'] for x in path['runs']]
spatial=[100*abs(x['right_inventory_signed_spatial_relative_error']) for x in r['runs']]
actual=[100*abs(x['right_inventory_signed_relative_error']) for x in path['runs']]
axes[1].loglog(t,spatial,'o-',label='Semidiscrete spatial limit')
axes[1].loglog(t,actual,'s--',label='Actual one-path BE total error')
axes[1].set(xlabel='Observation time',ylabel='Whole right-inventory relative error (%)',title='One uninterrupted positive, conservative BE path')
axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
