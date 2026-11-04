"""Saved responses and a signed budget for one unchanged point row."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    r=json.loads((Path(source)/'results.json').read_text());b=r['signed_budget']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    ax=axes[0];values=[r['PDE_unit_curvature_response_per_s'],r['raw_point_response_per_s'],r['saved_ideal_direct_response_per_s']]
    bars=ax.bar(np.arange(3),values,color=['grey','tab:blue','tab:orange'])
    ax.bar_label(bars,fmt='%.11f',padding=4,fontsize=8);ax.set_xticks(np.arange(3),['PDE','Saved point row','Saved ideal row'])
    ax.set_ylim(0,.024);ax.set_ylabel('Unit-curvature response [1/s]');ax.set_title('Same material-mean quadratic probe');ax.grid(axis='y',alpha=.2)
    ax=axes[1];values=[100*b[k] for k in ['point_minus_ideal_relative_to_PDE','ideal_minus_PDE_relative_to_PDE','point_minus_PDE_relative_to_PDE']]
    bars=ax.bar(np.arange(3),values,color=['tab:blue','tab:orange','tab:red'])
    ax.bar_label(bars,labels=[f'{v:+.7f}%' for v in values],padding=4,fontsize=8)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(np.arange(3),['Point - ideal','Ideal - PDE','Point - PDE'])
    ax.set_ylim(-3.6,3.6);ax.set_ylabel('Signed error / PDE response [%]');ax.set_title('Specified intermediate: large opposite terms');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Read-only existing row 31: N=128, h=0.04, D=0.02',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
