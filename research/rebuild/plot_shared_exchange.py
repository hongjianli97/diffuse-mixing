"""One fixed coupled system and its two-entry exact separating functional."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);result=json.loads((source/'results.json').read_text());z=np.load(source/'shared-exchange-system.npz')
    A=csr_matrix((z['A_data'],z['A_indices'],z['A_indptr']),shape=tuple(z['A_shape']));inside=z['interior_cells'];columns=z['columns'][inside]
    first=np.unique(columns,return_index=True)[1];fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ax=axes[0,0];ax.plot(columns[first],z['drift_margins'][first],'o-',ms=4,label='Same value in both interior rows')
    ax.axhline(0,color='black',lw=.8);old=int(np.flatnonzero(inside==result['previous_child']).item());bad=int(np.flatnonzero(inside==result['worst_cell']).item())
    ax.plot(columns[old],z['drift_margins'][old],'go',ms=8,label='Previous column 16');ax.plot(columns[bad],z['drift_margins'][bad],'ro',ms=8,label='Obstructing column 17')
    ax.set_xlabel('Stream column');ax.set_ylabel('Gamma_yy - h |beta_y| [m^2/s]');ax.set_title('All 60 interior cells checked, fixed h = 0.25 m')
    ax.set_ylim(-.004,.044);ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1];ax.spy(A,markersize=.7,aspect='auto')
    for row in z['certificate_dual_rows']:ax.axhline(row,color='C3',lw=.7)
    ax.set_xlabel('One variable per unordered pair');ax.set_ylabel('Mass-scaled moment equation')
    ax.set_title('300 equations / 1806 shared conductances')
    ax=axes[1,0];y=np.zeros(A.shape[0]);y[z['certificate_dual_rows']]=z['certificate_dual_values'];slack=np.asarray(A.T@y)
    ax.plot(np.arange(A.shape[1]),slack,'.',ms=3);ax.axhline(0,color='black',lw=.7)
    ax.set_xlabel('Pair-variable index');ax.set_ylabel('Dual value on each pair column [m^2]')
    ax.set_title('All pair slacks >= 0; target dual value < 0')
    gap=result['exact_embedded_separator']['exact_target_dot_dual']['float']
    ax.text(.03,.65,f'Exact target gap: {gap:.6g} kg m^2/s\nRational check covers every pair',transform=ax.transAxes,fontsize=9);ax.set_ylim(-.01,.15);ax.grid(alpha=.2)
    ax=axes[1,1];values=[result['worst_unscaled_drift_margin_m2_per_s'],*result['independent_boundary_drift_margins']]
    ax.plot(range(3),values,'o',ms=8);ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(range(3),['Mass primitive','Boundary Gauss 32','Boundary Gauss 64']);ax.set_ylabel('Local separating gap [m^2/s]')
    ax.set_title('Independent physical target control agrees')
    ax.text(.05,.2,'Gap ~ -6.75e-4\nTarget discrepancy ~1e-16',transform=ax.transAxes,fontsize=10);ax.grid(alpha=.2)
    fig.suptitle('Fixed shared-exchange exact-moment system rejected; no global diffusion acceptance',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
