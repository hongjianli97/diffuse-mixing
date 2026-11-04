"""Fixed-width constitutive deviation and signed error-budget cross term."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());budget=r['continuous_runs']['fine']['budget']['all']
    with np.load(source/'contact-continuous.npz') as z:
        x=z['material_label_centers'][:128,0];pde=z['reference_rate'][:128];discrete=z['kernel_rate'][:128];continuous=z['fine_rate'][:128]
        C=z['fixed_width_deviation'][:128];R=z['discrete_mean_closure_remainder'][:128]
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];ax.plot(x,pde,label='PDE material mean',lw=1.7);ax.plot(x,continuous,label='Continuous kernel, observed',lw=1.4);ax.plot(x,discrete,'--',label='Saved discrete kernel',lw=1.2)
    ax.set_xlabel('Material a [m]');ax.set_ylabel('Rate [1/s]');ax.set_title('Same original material row');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1];ax.plot(x,C,label='C: continuous kernel - PDE');ax.plot(x,R,label='R: discrete - continuous kernel')
    ax.axhline(0,color='k',lw=.7);ax.set_xlabel('Material a [m]');ax.set_ylabel('Rate difference [1/s]');ax.set_title('Raw constitutive and combined remainder');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];values=[100*budget[k] for k in ['total_error','continuous_fixed_width_error','discrete_mean_closure_remainder']]
    bars=ax.bar(['Total error','C norm','R norm'],values,color=['gray','tab:red','tab:blue']);ax.bar_label(bars,fmt='%.4f%%',padding=3,fontsize=9)
    ax.set_ylim(0,max(values)*1.18);ax.set_ylabel('Norm / original PDE rate norm [%]');ax.set_title('Component norms are not additive');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];values=[budget['total_error']**2,budget['continuous_fixed_width_error']**2,budget['discrete_mean_closure_remainder']**2,budget['signed_twice_cross_term_over_reference_norm_squared']]
    ax.bar(['Total squared','C squared','R squared','2<C,R>'],values,color=['gray','tab:red','tab:blue','tab:purple'])
    ax.axhline(0,color='k',lw=.8);ax.set_ylabel('Squared norm / PDE norm squared');ax.set_title('Negative cross term reduces total error');ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same h=0.04 contact: continuous-kernel mass averages on original cells',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
