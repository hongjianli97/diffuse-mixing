"""Graph fixture only: distinguish chronological propagation and mixing budget."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'supplied-graph-chain.npz') as z:
        time=z['time'];states=z['states'];zero=z['zero_exchange_states'];P=z['P'];averaged=z['exp_integrated_Q'];ordered=z['piecewise_exact_ordered_P']
    fig,ax=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for i in range(3):
        ax[0].plot(time,states[:,i],'o-',label=f'Material {i}')
        ax[0].plot(time,zero[:,i],':',color=f'C{i}',alpha=.6)
    ax[0].set_xlabel('Time [s]');ax[0].set_ylabel('Supplied-graph mass fraction');ax[0].set_title('Solid: exchange; dotted: zero exchange');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    E=np.array([v['energy_kg'] for v in r['graph_fixture_metrics']]);b=r['graph_fixture_energy_balances']
    physical=np.r_[0,np.cumsum([v['dt_chi_endpoint_kg'] for v in b])];numerical=np.r_[0,np.cumsum([v['backward_euler_extra_dissipation_kg'] for v in b])]
    ax[1].plot(time,E[0]-E,'o-',label='Total E loss')
    ax[1].plot(time,physical,'s--',label='Sum dt chi at endpoint')
    ax[1].plot(time,numerical,'x:',label='BE extra dissipation')
    ax[1].set_xlabel('Time [s]');ax[1].set_ylabel('Energy budget [kg]');ax[1].set_title('Numerical loss kept separate');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
    im=ax[2].imshow(ordered-averaged,cmap='RdBu_r',vmin=-np.max(abs(ordered-averaged)),vmax=np.max(abs(ordered-averaged)))
    ax[2].set_xticks(range(3));ax[2].set_yticks(range(3));ax[2].set_xlabel('Initial material j');ax[2].set_ylabel('Final material i')
    ax[2].set_title('Exact ordered P minus exp(integrated Q)');fig.colorbar(im,ax=ax[2],shrink=.85)
    fig.suptitle('Three-node supplied-graph interface check; no fluid PDE acceptance, no fluid ORC',fontsize=11)
    fig.savefig(output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
