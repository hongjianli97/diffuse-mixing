"""One contact frozen-rate comparison; all regions share the global norm."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'contact-observations.npz') as z:
        labels=z['material_label_centers'];candidate=z['kernel_rate'];reference=z['reference_rate'];error=z['rate_error']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];x=labels[:128,0]
    ax.plot(x,reference[:128],label='Integrated PDE cell rate',lw=1.7)
    ax.plot(x,candidate[:128],'--',label='Unchanged point kernel',lw=1.4)
    for left,right in [(0,.08),(.42,.58),(.92,1.)]:ax.axvspan(left,right,color='gray',alpha=.12)
    ax.set_xlabel('Material a [m]');ax.set_ylabel('Mass-average rate [1/s]');ax.set_title(f'Frozen row, b={labels[0,1]:.5f} m')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1];vmax=np.max(abs(error));im=ax.imshow(error.reshape(128,128),origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=-vmax,vmax=vmax)
    fig.colorbar(im,ax=ax,label='Kernel - PDE rate [1/s]');ax.axvline(.5,color='k',ls=':',lw=.8)
    ax.set_xlabel('Material a [m]');ax.set_ylabel('Material b [m]');ax.set_title('Raw rate error at density contacts')
    ax=axes[1,0];keys=['all','near_contact','far_from_contact']
    bars=ax.bar(['All cells','Within 2h','Remaining'],[100*r['regional_comparison'][key]['error_over_global_reference_norm'] for key in keys],color=['gray','tab:red','tab:blue'])
    ax.bar_label(bars,fmt='%.3f%%',padding=3,fontsize=9);ax.set_ylim(0,17)
    ax.set_ylabel('Error / full PDE rate norm [%]');ax.set_title('Same global norm for every region');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1]
    values=[r['reference_controls']['analytic_faces_vs_volume16_relative'],r['reference_controls']['volume8_vs_volume16_relative'],r['raw_graph_controls']['pair_scatter_vs_CSR_relative'],r['regional_comparison']['all']['error_over_global_reference_norm']]
    ax.bar(['Face / volume','Gauss 8 / 16','Pair / CSR','Kernel error'],values,color=['tab:green','tab:green','tab:purple','tab:red'])
    ax.set_yscale('log');ax.tick_params(axis='x',labelsize=8);ax.set_ylabel('Relative to full PDE rate norm');ax.set_title('Reference and implementation controls');ax.grid(axis='y',alpha=.2)
    fig.suptitle('One stationary 1:4 density contact: material mass averages, no transient',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
