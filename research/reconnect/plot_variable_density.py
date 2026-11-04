"""Frozen smooth-density control; full PDE, drift and raw input-weight audit."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'variable-density-kernel2d.npz') as z:
        x=z['positions'][1];rho=z['rho'][1];reference=z['reference_rate'];rate=z['kernel_rate'];lap=z['reference_laplacian_rate']
        labels=z['material_label_centers'];area_error=z['quadrature_area_relative_error'];N=int(np.sqrt(len(rho)))
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];im=ax.scatter(x[:,0],x[:,1],c=rho,cmap='viridis',s=4,marker='s');ax.set_aspect('equal');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title('Oracle point density, same translated identities');fig.colorbar(im,ax=ax,label='rho [kg/m3]',shrink=.85)
    ax=axes[0,1];ax.scatter(reference,rate,s=7,alpha=.5);lo=reference.min();hi=reference.max();ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    ax.set_xlabel('Full PDE mass-average rate [1/s]');ax.set_ylabel('Unmodified kernel rate [1/s]');ax.set_title(f'Full variable-density error {100*r["relative_mass_weighted_instantaneous_rate_error"]:.4f}%');ax.grid(alpha=.2)
    ax=axes[1,0];row=N//8;selection=slice(row*N,(row+1)*N)
    ax.plot(labels[selection,0],reference[selection],label='Full PDE');ax.plot(labels[selection,0],lap[selection],'--',label='Laplacian-only diagnostic')
    ax.plot(labels[selection,0],rate[selection],':',label='Point kernel');ax.set_xlabel('Material x label [m]');ax.set_ylabel('Rate [1/s]')
    ax.set_title('Density drift retained in the physical target');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];selection=slice(0,N);ax.plot(labels[selection,0],100*area_error[selection]);ax.axhline(0,color='k',lw=.8)
    ax.set_xlabel('Material x label [m]');ax.set_ylabel('Quadrature area / true area - 1 [%]');ax.set_title('Point-density weight discrepancy kept raw');ax.grid(alpha=.2)
    fig.suptitle('One periodic smooth-density snapshot, no state/weight fit; fluid ORC remains disabled',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
