"""Common original-cell shear rate errors from one saved joint refinement."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());c=r['comparison']
    with np.load(source/'shear-spatial.npz') as z:
        ref=z['parent_reference_rate'];old=z['parent_kernel_rate'];new=z['common_kernel_rate']
    errors=[old-ref,new-ref];vmax=max(np.max(abs(e)) for e in errors)
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];ax.bar(['N128 / h .04','N256 / h .03\nobserved on N128'],100*np.array([c['coarse_error_common_cells'],c['fine_error_common_cells']]))
    ax.set_ylabel('Relative full-PDE rate error [%]');ax.set_title('Same saved reference norm and original mass');ax.grid(axis='y',alpha=.2)
    ax=axes[0,1];ax.bar(['Error reduction','Reference + arithmetic\ndifference indicator'],[c['absolute_error_reduction'],c['combined_empirical_indicator']])
    ax.set_yscale('log');ax.set_ylabel('Relative norm');ax.set_title('Empirical differences; no rigorous bound');ax.grid(axis='y',alpha=.2)
    for ax,error,title in [(axes[1,0],errors[0],'N128 final frozen-rate residual'),(axes[1,1],errors[1],'N256 residual on original cells')]:
        im=ax.imshow(error.reshape(128,128),origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=-vmax,vmax=vmax)
        fig.colorbar(im,ax=ax,label='Kernel rate - full PDE reference [1/s]')
        ax.set_xlabel('Original material a [m]');ax.set_ylabel('Original material b [m]');ax.set_title(title)
    fig.suptitle('One periodic shear joint control; raw mass averages, no transient or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
