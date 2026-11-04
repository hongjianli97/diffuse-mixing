"""Weak affine residual and the unchanged saved kernel's error responses."""
from pathlib import Path
import argparse, json
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from positive_kernel_1d import slope_over_radius

p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--output', required=True)
a = p.parse_args(); run = Path(a.run)
data = json.loads((run/'results.json').read_text()); o = data['observed']
with np.load(run/'decomposition.npz') as z:
    x = z['positions']; h = float(z['h']); total = z['total_error']
    interface = z['interface_response']; homogeneous = z['homogeneous_response']
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
u = np.linspace(-2., 2., 161)
shape = [-2*quad(lambda r: float(slope_over_radius(r, 1.))*(r-2*abs(v)),
                 abs(v), 2., epsabs=1e-12)[0] for v in u]
axes[0, 0].plot(u, shape)
axes[0, 0].axhline(0, color='black', lw=.6)
axes[0, 0].set(xlabel='x / h', ylabel='h R / (F beta)', title='Continuous affine patch: zero signed mass')
axes[0, 1].plot(x/h, total, label='Saved kernel error', color='black', lw=1.8)
axes[0, 1].plot(x/h, interface, label='Density-interface response')
axes[0, 1].plot(x/h, homogeneous, label='Homogeneous-profile response')
axes[0, 1].set(xlim=(-8, 8), xlabel='x / h (detail; all cells in norms)',
               ylabel='Signed concentration error', title='Exact forcing split, same saved operator')
axes[0, 1].legend(fontsize=8)
labels = ['Interface', 'Homogeneous', 'Total']
fine = [o['interface_response_norm'], o['homogeneous_response_norm'], o['complete_error']]
parent = [o['parent_128_interface_norm'], o['parent_128_homogeneous_norm'], o['parent_128_total_error']]
indices = np.arange(3)
axes[1, 0].bar(indices-.18, fine, width=.36, label='Fine 2048 averages')
axes[1, 0].bar(indices+.18, parent, width=.36, label='Nested 128 averages')
axes[1, 0].set(xticks=indices, xticklabels=labels, yscale='log',
               ylabel='Error / initial fluctuation norm', title='Fixed material observables; norms are not additive')
axes[1, 0].legend(fontsize=8)
values = 100*np.array([o['interface_inventory_relative_error_contribution'],
                       o['homogeneous_inventory_relative_error_contribution'],
                       o['total_inventory_signed_relative_error']])
bars = axes[1, 1].bar(labels, values, color=['tab:blue', 'tab:orange', 'black'])
axes[1, 1].axhline(0, color='black', lw=.6)
axes[1, 1].bar_label(bars, labels=[f'{v:+.4g}' for v in values], padding=3)
axes[1, 1].margins(y=.2)
axes[1, 1].set(ylabel='Signed right-inventory error (%)', title='Signed inventory contributions are additive')
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.savefig(a.output, dpi=180)
