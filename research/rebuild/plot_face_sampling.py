"""Separate physical face-history sampling from scalar solver time steps."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
run=Path(a.run);r=json.loads((run/'results.json').read_text())
geometry=r['geometry_controls'];rows=r['final_comparison'];n=[x['sampling_intervals'] for x in rows]
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
axes[0,0].loglog(n,[100*x['max_relative_volume_error'] for x in geometry],'o-',label='Volume interpolation error')
axes[0,0].loglog(n,[100*x['max_actual_fluid_mass_relative_mismatch'] for x in geometry],'s-',label='Actual fluid mass mismatch')
axes[0,0].set(xlabel='Face sampling intervals',ylabel='Maximum relative error (%)',title='Same material identities at all resolutions')
axes[0,0].legend(fontsize=8)
for intervals in n:
    with np.load(run/f'geometry-{intervals}.npz') as z:
        t=z['time'];mass=z['mass'];actual=z['true_fluid_mass_in_modeled_cells']
    axes[0,1].plot(t,100*np.max(abs(actual-mass)/mass,axis=1),label=f'{intervals} intervals')
axes[0,1].set(xlabel='Time',ylabel='Maximum fluid mass mismatch (%)',title='Knot accuracy does not remove between-knot error')
axes[0,1].legend(fontsize=8)
axes[1,0].loglog(n,[q['extra_error'] for q in rows],'o-',label='Added concentration error')
axes[1,0].loglog(n,[q['paired_time_difference_in_sampling_change'] for q in rows],'s--',label='Paired BE time difference')
axes[1,0].set(xlabel='Face sampling intervals',ylabel='Initial-fluctuation-normalized difference',title='Sampling contribution and its time check')
axes[1,0].legend(fontsize=8)
axes[1,1].semilogx(n,[q['total_error'] for q in rows],'o-',label='Sampled face record')
axes[1,1].axhline(r['unchanged_exact_face_baseline_error'],color='black',ls='--',label='Same FV, exact continuous face history')
axes[1,1].set(xlabel='Face sampling intervals',ylabel='Full concentration error',title='Existing spatial/density closure remains')
axes[1,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[1,1].legend(fontsize=8)
for panel in axes.flat:panel.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
