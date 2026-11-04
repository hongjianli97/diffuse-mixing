"""Actual chronological scalar propagation compared with saved PDE reference."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'ordered-compression.npz') as z:
        ref=z['reference_states'];times=z['reference_times'];error=z['k8_states'][-1]-ref[-1]
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0]
    for K in [4,8]:
        case=r['cases'][str(K)];t=[0.]+[v['right_time_s'] for v in case['steps']];H=[0.]+[v['end_metrics']['H'] for v in case['steps']]
        ax.plot(t,H,'o-',ms=3,label=f'Ordered {K} intervals')
    ax.plot(times,[v['reference_metrics']['H'] for v in r['cases']['8']['observations']],'kx',label='Independent PDE')
    ax.set_xlabel('Time [s]');ax.set_ylabel('Mass mixing H');ax.set_title('Actual scalar, same saved mass/c0');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1]
    for K in [4,8]:ax.plot(times[1:],[v['relative_concentration_error_initial_fluctuation_norm'] for v in r['cases'][str(K)]['observations'][1:]],'o-',label=f'{K} intervals vs PDE')
    ax.plot(times[1:],r['time_and_reference_controls']['relative_state_difference_initial_fluctuation_norm'][1:],'s--',label='4 vs 8 temporal difference')
    ax.set_yscale('log');ax.set_xlabel('Time [s]');ax.set_ylabel('Difference / original fluctuation norm');ax.set_title('Concentration and necessary temporal control');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];vmax=np.max(abs(error));im=ax.imshow(error.reshape(128,128),origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=-vmax,vmax=vmax)
    fig.colorbar(im,ax=ax,label='c_8 - reference c');ax.set_xlabel('Material x label [m]');ax.set_ylabel('Material y label [m]');ax.set_title('Raw final mass-average concentration error')
    ax=axes[1,1]
    for K in [4,8]:ax.plot(times[1:],100*np.array([v['H_candidate_minus_reference'] for v in r['cases'][str(K)]['observations'][1:]]),'o-',label=f'{K} intervals')
    ax.set_xlabel('Time [s]');ax.set_ylabel('H_candidate - H_reference [percentage points]');ax.set_title('Mixing bias; no state/weight correction');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('One prescribed compression flow: ordered exponential actions, no full P or fluid ORC',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
