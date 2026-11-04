"""Full 2D reference solution, original-cell mass mixing and raw controls."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'shear-reference.npz') as z:
        times=z['snapshot_time'];states=z['N128_best_reference_states'];accepted_t=z['M127-k256_accepted_time'];inventory=z['M127-k256_weighted_inventory']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];im=ax.imshow(states[-1].reshape(128,128),origin='lower',extent=[0,1,0,1],cmap='viridis')
    fig.colorbar(im,ax=ax,label='Original-cell mass-average c');ax.set_xlabel('Material a [m]');ax.set_ylabel('Material b [m]');ax.set_title('Independent reference at T=0.05 s')
    ax=axes[0,1];ax.plot(times,[v['H'] for v in r['best_reference_metrics_N128']],'o-',label='Original N128 cells')
    ax.plot(times,[v['H'] for v in r['best_reference_metrics_N256']],'x--',label='Saved N256 cells')
    ax.set_xlabel('Time [s]');ax.set_ylabel('H = 1 - variance / original initial variance');ax.set_title('Reference mixing at three observation times');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];c=r['reference_uncertainty_controls']['128']
    ax.bar(['Space 63 vs 127','Max step T/128\nvs T/256'],[c['space_63_vs127_relative_to_original_fluctuation'][-1],c['max_step_halving_127_relative_to_original_fluctuation'][-1]])
    ax.set_ylabel('State difference / original fluctuation norm');ax.set_title('Differences at floating precision; no error bound');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];ax.plot(accepted_t,inventory-inventory[0]);ax.set_xlabel('Accepted time [s]');ax.set_ylabel('Raw reference component inventory change [kg]')
    ax.set_title('No inventory correction or state normalization');ax.grid(alpha=.2)
    fig.suptitle('Independent full 2D time-dependent shear PDE; no candidate propagation or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
