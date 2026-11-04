"""Two predeclared levels; identical-parent comparison and analytic prediction."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());c=r['comparison']
    with np.load(source/'same-parent-comparison.npz') as z:
        old=z['parent_kernel_rate']-z['parent_reference_rate'];new=z['restricted_kernel_rate']-z['parent_reference_rate']
        change=z['action_change'];prediction=z['continuous_change']+z['quadrature_change']
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),constrained_layout=True)
    ax=axes[0];x=np.arange(3);before=100*np.array([c['old_total_rate_error'],c['old_finite_width_error'],c['old_node_quadrature_error']]);after=100*np.array([c['new_total_rate_error'],c['new_finite_width_error'],c['new_node_quadrature_error']])
    ax.bar(x-.18,before,width=.36,label='N=32, h=.09');ax.bar(x+.18,after,width=.36,label='N=64, h=.06')
    ax.set_yscale('log');ax.set_ylim(1e-4,12);ax.set_xticks(x,['Total','Finite width','Quadrature'])
    for xx,vals in [(x-.18,before),(x+.18,after)]:
        for i,v in zip(xx,vals):ax.annotate(f'{v:.5g}',(i,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.set_ylabel('Relative rate error [%], logarithmic axis');ax.set_title('One predeclared additional level');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1];ax.scatter(old,new,s=9,alpha=.7);limit=max(abs(old).max(),abs(new).max());ax.plot([-limit,limit],[-limit,limit],'k--',lw=.8)
    ax.set_xlabel('Saved coarse rate error [1/s]');ax.set_ylabel('Restricted fine rate error [1/s]');ax.set_title('Same saved parent material cells');ax.grid(alpha=.2)
    ax=axes[2];ax.scatter(prediction,change,s=10,alpha=.6);lo=prediction.min();hi=prediction.max();ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    ax.set_xlabel('Continuous-kernel + quadrature change [1/s]');ax.set_ylabel('Observed restricted operator change [1/s]');ax.set_title('Raw signed-vector decomposition');ax.grid(alpha=.2)
    fig.suptitle('2D point exchange: joint refinement reduces one-mode error; no transient or fluid ORC acceptance',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
