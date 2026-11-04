"""Same-input, semidiscrete comparison of new 1D kernel and material FV."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.run);r=json.loads((root/'results.json').read_text());rows=r['runs'];t=[x['time'] for x in rows]
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].loglog(t,[100*abs(x['kernel_inventory_signed_relative_error']) for x in rows],'o-',label='New 1D kernel, h_i=2 cell widths')
axes[0].loglog(t,[100*abs(x['FV_inventory_signed_relative_error']) for x in rows],'s-',label='Material FV baseline')
axes[0].set(xlabel='Time',ylabel='Whole right-inventory error (%)',title='Same masses, initial cell averages, oracle rho, D')
axes[1].loglog(t,[x['kernel_full_error'] for x in rows],'o-',label='New 1D kernel')
axes[1].loglog(t,[x['FV_full_error'] for x in rows],'s-',label='Material FV baseline')
axes[1].set(xlabel='Time',ylabel='Mass-weighted error / initial fluctuation norm',title='Full concentration error: both semidiscrete')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(a.output,dpi=180)
