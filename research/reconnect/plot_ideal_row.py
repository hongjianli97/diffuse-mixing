"""Saved one-row response and raw positive closure integrand."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text());fine=r['rules']['fine']
    with np.load(source/'ideal-row.npz') as z:s=z['fine_s'];integrand=z['fine_eta_integrand']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    ax=axes[0];values=[r['auxiliary_PDE_rate_per_s'],r['rules']['coarse']['raw_direct_aggregate_rate_per_s'],fine['raw_direct_aggregate_rate_per_s']]
    bars=ax.bar(np.arange(3),values,color=['grey','tab:orange','tab:blue'])
    ax.bar_label(bars,fmt='%.8f',padding=4,fontsize=9);ax.set_xticks(np.arange(3),['PDE','Ideal 32/32','Ideal 64/64'])
    ax.set_ylim(0,.024);ax.set_ylabel('Unit-curvature rate [1/s]');ax.set_title(f"Ideal closure excess: {r['fine_eta_percent']:.6f}%");ax.grid(axis='y',alpha=.2)
    ax=axes[1];ax.plot(s/.04,.04*integrand,lw=1.2)
    ax.set_xlabel('Normal displacement s/h');ax.set_ylabel('Marginal excess density');ax.grid(alpha=.2)
    ax.set_title('Same raw kernel, one fixed cell-width ratio')
    ax.text(.03,.95,f"Integral eta = {fine['eta']:.10f}\nRule difference = {r['coarse_fine_absolute_differences']['eta']:.2e}",transform=ax.transAxes,va='top',fontsize=9)
    fig.suptitle('Original N=128, h=0.04, D=0.02: one ideal material-mean row',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
