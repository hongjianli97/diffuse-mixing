"""One 2D shear state: raw geometry, full-PDE rate and essential couplings."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'shear-frozen.npz') as z:
        x=z['positions'][1];rho=z['rho'][1];labels=z['material_label_centers'];ref=z['reference_rate'];rate=z['kernel_rate'];error=z['rate_error']
    N=int(np.sqrt(len(ref)));fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];selected=(np.arange(len(ref))//N)%4==0;im=ax.scatter(x[selected,0],x[selected,1],c=rho[selected],s=5,cmap='viridis')
    fig.colorbar(im,ax=ax,label='Oracle point rho [kg/m3]');ax.set_aspect('equal');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel('Physical x [m]');ax.set_ylabel('Physical y [m]');ax.set_title('Sheared cloud (every fourth row shown)')
    ax=axes[0,1];ax.scatter(ref,rate,s=4,alpha=.4);ax.plot([ref.min(),ref.max()],[ref.min(),ref.max()],'k--',lw=.8)
    ax.set_xlabel('Full local PDE mass-average rate [1/s]');ax.set_ylabel('Unmodified kernel action [1/s]')
    ax.set_title(f'Frozen rate error {100*r["relative_mass_weighted_instantaneous_rate_error"]:.4f}%');ax.grid(alpha=.2)
    ax=axes[1,0];vmax=np.max(abs(error));im=ax.imshow(error.reshape(N,N),origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=-vmax,vmax=vmax)
    fig.colorbar(im,ax=ax,label='Kernel rate - full reference [1/s]');ax.set_xlabel('Material a [m]');ax.set_ylabel('Material b [m]');ax.set_title('Raw material-cell rate residual')
    ax=axes[1,1];parts=r['PDE_component_norms_over_full_rate_norm'];keys=['metric_correction','mixed_derivative','shear_gradient','density_drift']
    ax.bar(['Metric\ncorrection','Mixed\nderivative','Shear\ngradient','Density\ndrift'],[100*parts[key] for key in keys])
    ax.set_ylabel('Component norm / full reference norm [%]');ax.set_title('Reference terms, not kernel error shares');ax.grid(axis='y',alpha=.2)
    fig.suptitle('One frozen periodic shear, original mass/c0; no shear transient or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
