"""One prescribed compression snapshot: geometry and physical rate target."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'compression-kernel2d.npz') as z:
        labels=z['material_label_centers'];x=z['positions'][1];rho=z['rho'];reference=z['reference_rate'];rate=z['kernel_rate']
        width=z['true_cell_width'];offset=z['node_minus_geometric_center_x'];N=int(np.sqrt(len(labels)))
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];select=(np.arange(len(labels))//N)%4==0;im=ax.scatter(x[select,0],x[select,1],c=width[select],s=5,cmap='viridis')
    ax.set_aspect('equal');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    ax.set_title('Deformed point cloud (every fourth row shown)');fig.colorbar(im,ax=ax,label='True cell width [m]',shrink=.85)
    ax=axes[0,1];ax.scatter(reference,rate,s=5,alpha=.45);lo=reference.min();hi=reference.max();ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    ax.set_xlabel('True mass-average local PDE rate [1/s]');ax.set_ylabel('Unmodified kernel action [1/s]')
    ax.set_title(f'Frozen rate error {100*r["relative_mass_weighted_instantaneous_rate_error"]:.4f}%');ax.grid(alpha=.2)
    ax=axes[1,0];a=labels[:N,0];ax.plot(a,rho[0,:N],label='Initial oracle rho0');ax.plot(a,rho[1,:N],label='Frozen rho0/J')
    ax.set_xlabel('Material x label [m]');ax.set_ylabel('Point density [kg/m3]');ax.set_title('Uniform final density follows the prescribed map');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];ax.plot(a,1e6*offset[:N]);ax.axhline(0,color='k',lw=.8);ax.set_xlabel('Material x label [m]')
    ax.set_ylabel('Trajectory point - geometric center [micrometre]');ax.set_title('Raw offset retained; no recentering');ax.grid(alpha=.2)
    fig.suptitle('One smooth compression geometry control; saved mass/c0 retained, no transient or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
