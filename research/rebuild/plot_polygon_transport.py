"""Fixed 2D mesh, sampled mass mismatch and exact geometric sweep checks."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
run=Path(a.run);r=json.loads((run/'results.json').read_text());rows=r['geometry_and_transport'];n=[q['sampling_intervals'] for q in rows]
fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
with np.load(run/'audit-32.npz') as z:
    x=z['positions'];edges=z['edge_vertices'];t=z['time'];mid=len(t)//2
for k,color,label in [(0,'tab:blue','t=0'),(mid,'tab:green','t=T/2'),(len(t)-1,'tab:orange','t=T')]:
    segments=x[k,edges];ax[0,0].add_collection(LineCollection(segments,colors=color,linewidths=.6,label=label))
ax[0,0].autoscale();ax[0,0].set_aspect('equal');ax[0,0].legend(fontsize=7)
ax[0,0].set(xlabel='x',ylabel='y',title='64 fixed material cells; shared edges retained')
ax[0,1].loglog(n,[100*q['max_relative_area_error'] for q in rows],'o-',label='Area interpolation error')
ax[0,1].loglog(n,[100*q['max_actual_fluid_mass_relative_mismatch'] for q in rows],'s-',label='Actual fluid mass mismatch')
ax[0,1].set(xlabel='Vertex sampling intervals',ylabel='Maximum sampled relative error (%)',title='Prescribed rotating/sheared continuity benchmark')
ax[0,1].legend(fontsize=8)
ax[1,0].loglog(n,[q['maximum_area_increment_residual'] for q in rows],'o-',label='Correct midpoint sweep')
ax[1,0].loglog(n,[q['left_endpoint_sweep_residual_negative_control'] for q in rows],'s--',label='Left-edge geometry negative control')
ax[1,0].set(xlabel='Vertex sampling intervals',ylabel='Area increment residual',title='Linear vertex paths give quadratic area')
ax[1,0].legend(fontsize=8)
for intervals in n:
    with np.load(run/f'audit-{intervals}.npz') as z:
        tt=z['time'];m=z['fixed_mass'];actual=z['actual_mass']
    ax[1,1].plot(tt,100*np.max(abs(actual-m)/m,axis=1),label=f'{intervals} intervals')
ax[1,1].set(xlabel='Time',ylabel='Maximum actual fluid mass mismatch (%)',title='GCL does not certify fluid materiality')
ax[1,1].legend(fontsize=8)
for panel in ax.flat:panel.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
