"""Interface defects and the distinction between material and arithmetic means."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    r=json.loads((Path(source)/'results.json').read_text());region=r['two_cell_region']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    ax=axes[0];values=[.02,r['rows']['left']['row']['raw_CSR_quadratic_response_per_s'],r['rows']['right']['row']['raw_CSR_quadratic_response_per_s']]
    bars=ax.bar(np.arange(3),values,color=['grey','tab:blue','tab:orange']);ax.bar_label(bars,fmt='%.8f',padding=4,fontsize=9)
    ax.set_xticks(np.arange(3),['PDE, both sides','Row 63, rho=1','Row 64, rho=4']);ax.set_ylim(0,.029)
    ax.set_ylabel('Unit-curvature response [1/s]');ax.set_title('Same saved graph: opposite interface defects');ax.grid(axis='y',alpha=.2)
    ax=axes[1];values=[region['mass_average_reference_per_s'],region['arithmetic_mean_diagnostic_only_per_s'],region['mass_average_rate_per_s']]
    bars=ax.bar(np.arange(3),values,color=['grey','lightgrey','tab:red']);ax.bar_label(bars,fmt='%.8f',padding=4,fontsize=9)
    ax.set_xticks(np.arange(3),['PDE','Arithmetic mean\n(diagnostic only)','Material mass mean']);ax.set_ylim(0,.025)
    ax.set_ylabel('Two-cell mean response [1/s]');ax.set_title(f"Original masses 1:4; material error {100*region['mass_average_relative_error']:.4f}%");ax.grid(axis='y',alpha=.2)
    fig.suptitle('Zero-interface-flux quadratic mass means: N=128, h=0.04, D=0.02',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
