"""Plot recorded, propagated error components without new simulation."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);args=p.parse_args()
root=Path(args.run)
names={'ordinary_volume_center':'Geometric midpoint','ordinary_mass_center':'Mass midpoint','mass_average_resistance':'Average resistance'}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for color,(method,name) in zip(('C0','C1','C2'),names.items()):
 for steps in (256,512):
  with np.load(root/f'budget-n4096-k4096-{method}-k{steps}.npz') as z:
   m=z['mass'];c0=z['c0'];parts=z['contributions'];t=z['time']
   norm0=np.sqrt(m@((c0-m@c0/m.sum())**2))
   if steps==512:
    spatial=parts[:,:,1]+parts[:,:,2]
    axes[0].plot(t,np.sqrt((spatial**2)@m)/norm0,label=name,color=color)
   axes[1].plot(t,np.sqrt((parts[:,:,3]**2)@m)/norm0,label=f'{name}, {steps}',color=color,linestyle='--' if steps==256 else '-')
axes[0].set(title='Accumulated spatial-defect contribution (512 steps)',xlabel='Time',ylabel='Norm / initial fluctuation norm')
axes[1].set(title='Time remainder approximately halves',xlabel='Time',ylabel='Norm / initial fluctuation norm')
for ax in axes:
 ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(args.output,dpi=180)
