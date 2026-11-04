"""Independent compression reference: states, controls and mass mixing."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'compression-reference.npz') as z:
        axis=z['n1024-k1024_axis'];A=z['n1024-k1024_A_snapshots'];times=z['snapshot_time'];states=z['best_projected_states']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0]
    for t,a in zip(times,A):ax.plot(axis,a,label=f't={t:g} s')
    ax.set_xlabel('Material x label [m]');ax.set_ylabel('A(a,t), material-column mean');ax.set_title('Independent full material PDE');ax.legend();ax.grid(alpha=.2)
    ax=axes[0,1];im=ax.imshow(states[-1].reshape(128,128),origin='lower',extent=[0,1,0,1],cmap='viridis')
    fig.colorbar(im,ax=ax,label='Original-cell mass-average c');ax.set_xlabel('Material x label [m]');ax.set_ylabel('Material y label [m]');ax.set_title('Reference at T=0.05 s, saved mass denominator')
    ax=axes[1,0];d=r['uncertainty_controls']['state_differences_relative_to_original_mass_weighted_fluctuation']
    v=[d[k][-1] for k in ['space_512_vs_1024_at_256_steps','time_256_vs_512_at_1024_columns','time_512_vs_1024_at_1024_columns']]
    ax.bar(['Space\n512 vs 1024','Time\n256 vs 512','Time\n512 vs 1024'],v);ax.set_yscale('log');ax.set_ylabel('State difference / original fluctuation norm')
    ax.set_title('Empirical differences; no rigorous bound');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];ax.plot(times,[a['H'] for a in r['best_raw_reference_metrics']],'o-')
    ax.set_xlabel('Time [s]');ax.set_ylabel('H = 1 - mass variance / initial variance');ax.set_title('Reference mixing, three recorded snapshots');ax.grid(alpha=.2)
    fig.suptitle('Independent compression-flow reference only; no candidate propagation or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
