"""Saved smooth-density levels: own/parent errors and signed prediction chain."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());c=r['comparison'];old=c['old_budget'];new=r['budget']
    with np.load(source/'same-parent-comparison.npz') as z:
        before=z['parent_kernel_rate']-z['parent_reference_rate'];after=z['restricted_kernel_rate']-z['parent_reference_rate']
        observed=z['actual_change'];predicted=z['physical_change']+z['closure_change']+z['quadrature_change']
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    keys=['total_relative_error','finite_width_relative_norm','averaged_input_point_density_closure_relative_norm','node_quadrature_relative_norm'];x=np.arange(4)
    ax=axes[0];a=100*np.array([old[k] for k in keys]);b=100*np.array([new[k] for k in keys])
    ax.bar(x-.18,a,.36,label='N=64, h=.06');ax.bar(x+.18,b,.36,label='N=128, h=.04');ax.set_yscale('log');ax.set_ylim(5e-5,8)
    ax.set_xticks(x,['Total','Width','Closure','Quadrature']);ax.tick_params(axis='x',labelsize=8)
    for xx,vals in [(x-.18,a),(x+.18,b)]:
        for pos,value in zip(xx,vals):ax.annotate(f'{value:.4g}',(pos,value),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
    ax.set_ylabel('Own-cell relative rate norm [%], log axis');ax.set_title('One additional predeclared level');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1];ax.scatter(before,after,s=5,alpha=.5);lo=before.min();hi=before.max();ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    ax.set_xlabel('Saved parent-grid rate error [1/s]');ax.set_ylabel('Restricted fine rate error [1/s]');ax.set_title('Same saved parent masses');ax.grid(alpha=.2)
    ax=axes[2];ax.scatter(predicted,observed,s=5,alpha=.5);lo=predicted.min();hi=predicted.max();ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    ax.set_xlabel('Continuous + closure + quadrature change [1/s]');ax.set_ylabel('Observed operator change [1/s]');ax.set_title('Independent-reference signed budget');ax.grid(alpha=.2)
    fig.suptitle('Smooth variable density: error reduction with raw mass rounding retained; fluid ORC disabled',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
