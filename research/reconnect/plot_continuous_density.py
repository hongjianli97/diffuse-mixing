"""Fixed error vectors and reference-resolution evidence, no new graph result."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());b=r['budget'];a=r['reference_audit']
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    values=100*np.array([b['total_relative_error'],b['finite_width_relative_norm'],b['averaged_input_point_density_closure_relative_norm'],b['node_quadrature_relative_norm']])
    ax=axes[0];ax.bar(range(4),values,color=['.5','C0','C1','C2']);ax.set_yscale('log');ax.set_ylim(1e-4,10)
    ax.set_xticks(range(4),['Saved total','Finite width','Input closure','Node quadrature']);ax.tick_params(axis='x',labelsize=9)
    for i,v in enumerate(values):ax.annotate(f'{v:.6g}%',(i,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    ax.set_ylabel('Relative mass-weighted rate norm [%], log axis');ax.set_title('Comparator-chain norms; not additive');ax.grid(axis='y',alpha=.2)
    keys=['finite_width','averaged_input_point_density_closure','node_quadrature'];ratios=[a['component_rule_difference_to_component_norm'][key] for key in keys]
    ax=axes[1];ax.bar(range(3),ratios,color=['C0','C1','C2']);ax.set_yscale('log');ax.set_ylim(1e-14,1e-9)
    ax.set_xticks(range(3),['Finite width','Input closure','Node quadrature'])
    for i,v in enumerate(ratios):ax.annotate(f'{v:.3g}',(i,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    ax.set_ylabel('Rule difference / component norm, log axis');ax.set_title('Empirical integration control, not certified bounds');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same saved smooth-density state: dominant finite-width bias; graph unchanged, fluid ORC disabled',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
