"""Paired actual-flow errors on the same original material cells."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());rows=r['comparison_common_original_cells'][1:]
    t=np.array([v['time_s'] for v in rows])
    with np.load(source/'shear-transient-spatial.npz') as z:
        coarse_error=z['coarse_k8_observations'][-1]-z['parent_reference_states'][-1]
        fine_error=z['k8_parent_states'][-1]-z['parent_reference_states'][-1]
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0]
    ax.plot(t,[v['coarse_error_common_cells'] for v in rows],'o-',label='N128 / h .04')
    ax.plot(t,[v['fine_error_common_cells'] for v in rows],'o-',label='N256 / h .03, observed on N128')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Error / original initial fluctuation norm');ax.set_title('Same-cell concentration errors');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1];ax.plot(t,[v['absolute_error_reduction'] for v in rows],'o-',label='Observed error reduction')
    ax.plot(t,[v['combined_empirical_indicator'] for v in rows],'s--',label='Temporal + reference differences')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Relative norm');ax.set_title('Error reduction and measured differences');ax.legend(fontsize=8);ax.grid(alpha=.2)
    vmax=max(np.max(abs(coarse_error)),np.max(abs(fine_error)))
    for ax,error,title in [(axes[1,0],coarse_error,'N128 / h .04 final error'),(axes[1,1],fine_error,'N256 / h .03 final common-cell error')]:
        im=ax.imshow(error.reshape(128,128),origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=-vmax,vmax=vmax)
        ax.set_xlabel('Original material x label [m]');ax.set_ylabel('Original material y label [m]');ax.set_title(title)
        fig.colorbar(im,ax=ax,label='c - independent reference')
    fig.suptitle('One 2D shear spatial control: original material mass-average observations',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
