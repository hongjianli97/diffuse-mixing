"""Actual saved c0 evolution, mass mixing and independent PDE comparison."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());rows=r['runs']
    with np.load(source/'density-transient.npz') as z:
        times=z['time'];states=z['candidate_states'];reference=z['reference512_states'];labels=z['material_label_centers'];c0=z['c0']
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    ax=axes[0];ax.plot(times,[v['candidate_metrics']['H'] for v in rows],'o-',label='Saved point kernel')
    ax.plot(times,[v['reference_metrics_same_saved_initial_E']['H'] for v in rows],'s--',label='Independent local PDE')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Cell-scale mass mixing H');ax.set_title('Actual initial scalar, no normalization repair');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1];ax.semilogy(times[1:],100*np.array([v['relative_concentration_error_initial_fluctuation_norm'] for v in rows[1:]]),'o-',label='Candidate vs local PDE')
    ax.semilogy(times[1:],100*np.array([v['reference_256_vs_512_relative_difference'] for v in rows[1:]]),'s--',label='Reference 256 vs 512')
    ax.scatter([times[-1]],[100*r['algorithm_controls']['Q_vs_mass_symmetric_initial_fluctuation_norm']],marker='x',label='Q vs mass-symmetric action')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Initial-fluctuation normalized difference [%]');ax.set_title('Reference/algorithm checks kept separate');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[2];N=int(np.sqrt(len(c0)));row=N//8;slice_=slice(row*N,(row+1)*N)
    ax.plot(labels[slice_,0],c0[slice_],':',label='Saved c0');ax.plot(labels[slice_,0],reference[-1,slice_],label='Local PDE at .1 s')
    ax.plot(labels[slice_,0],states[-1,slice_],'--',label='Point kernel at .1 s')
    ax.set_xlabel('Material x label [m]');ax.set_ylabel('True cell mass-average concentration');ax.set_title('Same material-cell observations');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('One smooth periodic finite-time research case; no full P, clipping, or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
