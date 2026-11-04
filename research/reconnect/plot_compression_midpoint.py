"""Single coupled midpoint and actual-state endpoint commutator observations."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'compression-midpoint.npz') as z:
        labels=z['material_label_centers'];N=int(np.sqrt(len(labels)));a=labels[:N,0]
        rho=z['rho'][1,:N];area_error=(z['quadrature_area']/z['true_cell_area_from_unwrapped_faces']-1)[:N]
        ref=z['reference_rate'];rate=z['kernel_rate'];drift=z['gauss16_drift_rate']
    with np.load(source/'endpoint-actions.npz') as z:comm=z['commutator'];pair=z['pair_commutator']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];ax.plot(a,rho);ax.axhline(1,color='k',lw=.7);ax.set_xlabel('Material x label [m]');ax.set_ylabel('Oracle rho0/J [kg/m3]')
    ax.set_title('Predeclared midpoint: nonuniform density');ax.grid(alpha=.2)
    ax=axes[0,1];ax.plot(a,1e6*area_error);ax.set_xlabel('Material x label [m]');ax.set_ylabel('(m/rho_point)/true area - 1 [ppm]')
    ax.set_title('Raw quadrature-area discrepancy retained');ax.grid(alpha=.2)
    ax=axes[1,0];ax.scatter(ref,rate,s=4,alpha=.45);ax.plot([ref.min(),ref.max()],[ref.min(),ref.max()],'k--',lw=.7)
    ax.set_xlabel('Full local PDE mass-average rate [1/s]');ax.set_ylabel('Unmodified kernel action [1/s]')
    ax.set_title(f'Frozen rate error {100*r["relative_mass_weighted_instantaneous_rate_error"]:.4f}%');ax.grid(alpha=.2)
    ax=axes[1,1];ax.plot(a,comm[:N],label='Nested sparse actions');ax.plot(a,pair[:N],'--',label='Independent shared-flux scatter')
    ax.set_xlabel('Material x label at first y row [m]');ax.set_ylabel('[Q_end, Q_start] c0 [1/s2]')
    ax.set_title('Endpoint noncommutativity; no time integration');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('One frozen midpoint + saved endpoint vector actions; no compression transient or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
