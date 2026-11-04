"""One 2D kernel probe and analytic finite-width error attribution."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'periodic-kernel2d.npz') as z:
        x=z['positions'][1];state=z['c0'];reference=z['reference_rate'];rate=z['kernel_rate']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.5),constrained_layout=True)
    ax=axes[0,0];im=ax.scatter(x[:,0],x[:,1],c=state,cmap='viridis',s=10,marker='s');ax.set_aspect('equal');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title('Exact material-square mass averages');fig.colorbar(im,ax=ax,shrink=.85)
    ax=axes[0,1];ax.scatter(reference,rate,s=10,alpha=.6);a=reference.min();b=reference.max();ax.plot([a,b],[a,b],'k--',lw=.8)
    ax.set_xlabel('Local PDE mass-average rate [1/s]');ax.set_ylabel('Point-kernel action [1/s]');ax.set_title('One grid, one h, no transient');ax.grid(alpha=.2)
    ax=axes[1,0];values=100*np.array([r['relative_mass_weighted_instantaneous_rate_error'],r['finite_width_rate_bias_relative_to_local_PDE'],r['node_quadrature_rate_difference_relative_to_local_PDE']])
    ax.bar(range(3),values,color=['C0','C1','C2']);ax.set_xticks(range(3),['Total','Finite width','Node quadrature'])
    for i,v in enumerate(values):ax.annotate(f'{v:.5f}%',(i,v),xytext=(0,5),textcoords='offset points',ha='center')
    ax.set_ylim(0,3.9);ax.set_ylabel('Rate norm / local PDE rate norm [%]');ax.set_title('Signed mode errors retained separately');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];values=[r['continuum_local_decay_per_s'],r['continuum_finite_width_decay_per_s']['W128'],r['discrete_Fourier_symbols_plus_minus_per_s'][0]]
    ax.bar(range(3),values,color=['.5','C1','C0']);ax.set_xticks(range(3),['Local PDE','Continuous kernel','Discrete kernel'])
    for i,v in enumerate(values):ax.annotate(f'{v:.7f}',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    ax.set_ylim(1.50,1.60);ax.set_ylabel('Mode decay rate [1/s] (zoomed axis)');ax.set_title('Independent radial reference and stencil symbol');ax.grid(axis='y',alpha=.2)
    fig.suptitle('2D periodic oracle-density control: nonnegative exchange, finite-width bias; fluid ORC disabled',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
