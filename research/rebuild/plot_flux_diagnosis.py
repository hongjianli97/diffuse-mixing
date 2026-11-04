"""Standalone figure of saved common-state flux diagnostics."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);report=json.loads((root/'results.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
names=['Geometric\ncenter','Mass\nmidpoint','Average\nresistance'];x=np.arange(3)
for off,key,label in ((-.18,'initial_exact','t=0, analytic flux'),(.18,'final_finest','t=0.1, numerical reference')):
 axes[0].bar(x+off,[100*r['relative_common_state_flux_error'] for r in report[key]],width=.36,label=label)
axes[0].set(xticks=x,xticklabels=names,ylabel='Relative common-state flux error (%)',title='Same concentration averages')
axes[0].legend(fontsize=7)
with np.load(root/'final-n4096-k4096.npz') as z:
 mu=z['mass_edges'][1:-1]/z['mass_edges'][-1]
 for ax,method,title in ((axes[1],'ordinary_mass_center','Mass midpoint: partial cancellation'),(axes[2],'ordinary_volume_center','Geometric center: reinforcement')):
  ax.plot(mu,z[method+'_average_bias'],label='Average-flux bias',lw=1.5)
  ax.plot(mu,z[method+'_resistance_scale_term'],label='Resistance-scale term',lw=1.5)
  ax.plot(mu,z[method+'_closure_error'],label='Sum = closure error',lw=1.5,linestyle='--')
  ax.axhline(0,color='gray',lw=.7)
  ax.set(xlabel='Mass coordinate / total mass',ylabel='Signed flux error',title=title)
  ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
  ax.legend(fontsize=7)
for ax in axes:ax.grid(alpha=.15)
fig.savefig(a.output,dpi=180)
