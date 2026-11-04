"""Joint-path contact errors and separate fixed-h nonuniform quadrature."""
from pathlib import Path
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--output', required=True)
a = p.parse_args(); data = json.loads((Path(a.run)/'results.json').read_text())
rows = data['physical_runs']; ctrl = data['nonuniform_fixed_h_controls']
n = [r['cells'] for r in rows]; s = [r['subdivision_factor'] for r in ctrl]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
axes[0, 0].loglog(n, [r['kernel_full_error'] for r in rows], 'o-', label='New 1D kernel')
axes[0, 0].loglog(n, [r['FV_full_error'] for r in rows], 's-', label='Matched material FV')
axes[0, 0].set(xlabel='Material cells N', ylabel='Error / initial fluctuation norm',
               title='Same contact, T=0.1; h=0.25 sqrt(dx)')
axes[0, 0].legend()
axes[0, 1].semilogx(n, [100*r['kernel_inventory_signed_relative_error'] for r in rows], 'o-', label='Kernel')
axes[0, 1].semilogx(n, [100*r['FV_inventory_signed_relative_error'] for r in rows], 's-', label='FV')
axes[0, 1].axhline(0, color='black', lw=.7)
axes[0, 1].set_yscale('symlog', linthresh=.01)
axes[0, 1].set(xlabel='Material cells N', ylabel='Signed right-inventory error (%)',
               title='Inventory is not monotone on the joint path')
axes[0, 1].legend()
axes[1, 0].loglog(s, [r['interior_first_moment_max_abs'] for r in ctrl], 'o-')
axes[1, 0].set(xlabel='Subdivisions per original material cell', ylabel='max |first moment|',
               title='Nonuniform quadrature only, fixed h=0.03125')
axes[1, 1].loglog(s, [max(abs(r['interior_diffusivity_ratio_min']-1),
                              abs(r['interior_diffusivity_ratio_max']-1)) for r in ctrl], 'o-')
axes[1, 1].set(xlabel='Subdivisions per original material cell', ylabel='max |D_eff / D - 1|',
               title='Same old 436 material intervals retained')
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.savefig(a.output, dpi=180)
