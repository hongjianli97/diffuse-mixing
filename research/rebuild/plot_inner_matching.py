"""Independent inner shape and unfitted comparisons at two saved widths."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args()
run=Path(a.run);r=json.loads((run/'results.json').read_text());F=r['exact_reference_flux_at_T']
physical=json.loads((run/'config.json').read_text())['physical_configuration'];D=physical['D']
fig,ax=plt.subplots(1,3,figsize=(16,4),layout='constrained')
with np.load(run/'inner-control2.npz') as z:
    x=z['positions'];psi=z['localized_corrector']
ax[0].plot(x,psi)
ax[0].set(xlim=(-4,4),xlabel='z=x/h',ylabel='Localized inner corrector psi',
          title='Unit-flux cell problem; no measured-error fit')
ax[0].axhline(0,color='black',lw=.6)
for n,color in [(512,'tab:orange'),(2048,'tab:blue')]:
    with np.load(run/f'prediction-N{n}.npz') as z:
        x=z['positions'];h=float(z['h']);idx=z['inner_selected_indices']
        observed=z['observed_interface_response'];predicted=z['inner_composite_prediction']
    ax[1].plot(x[idx]/h,observed[idx]/(F*h/D),color=color,label=f'Saved response N={n}')
    ax[1].plot(x[idx]/h,predicted/(F*h/D),'--',color=color,label=f'Prediction N={n}')
ax[1].set(xlabel='x/h',ylabel='Interface response / (F(T) h / D)',title='Same independent coefficient at both widths')
ax[1].legend(fontsize=7)
rows=r['saved_width_comparisons'];h=[q['h'] for q in rows]
ax[2].plot(h,[q['observed_inventory_per_h'] for q in rows],'o-',label='Observed density-interface response')
reduced=physical['rho_left']*physical['rho_right']/(physical['rho_left']+physical['rho_right'])
ax[2].axhline(r['independent_coefficient']['kappa']*reduced**2*(physical['initial_left']-physical['initial_right']),
              color='black',ls='--',label='Unfitted leading prediction')
ax[2].set(xlabel='Kernel width h',ylabel='Right-inventory contribution / h',
          title='Two saved widths; no fitted convergence order')
ax[2].legend(fontsize=7)
for panel in ax:panel.grid(alpha=.2)
fig.savefig(a.output,dpi=180)
