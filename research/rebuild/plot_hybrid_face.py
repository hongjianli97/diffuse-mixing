"""Plot one fixed hybrid real-face baseline; no new physical computations."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'hybrid-face-baseline.npz') as z:
        ref=z['physical_reference_rate'];hybrid=z['hybrid_rate'];graph=z['saved_graph_rate'];quads=z['quads']
        interior=~z['topology_boundary_mask'];area=z['operator_face_area']
        qr=z['physical_reference_flux'][interior]/area[interior];qh=z['hybrid_face_flux'][interior]/area[interior]
        derivative=z['basis_rate'];centers=z['mass_centroids']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ax=axes[0,0];ax.scatter(ref,graph,s=13,label='Saved diagnostic graph',alpha=.65)
    ax.scatter(ref,hybrid,s=13,label='Hybrid real-face FV',alpha=.75)
    lo=min(ref.min(),graph.min(),hybrid.min());hi=max(ref.max(),graph.max(),hybrid.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=.8);ax.set_xlabel('Physical reference rate [1/s]');ax.set_ylabel('Discrete rate [1/s]')
    ax.set_title('Same mass-average state, no time integration');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1];ax.scatter(qr,qh,s=12)
    lo=min(qr.min(),qh.min());hi=max(qr.max(),qh.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    ax.set_xlabel('Physical normal flux density [kg/(m2 s)]');ax.set_ylabel('Native FV normal flux density [kg/(m2 s)]')
    ax.set_title(f'220 internal real faces: error {100*r["internal_face_flux_density_relative_L2_error"]:.2f}%');ax.grid(alpha=.2)
    ax=axes[1,0];x=np.arange(3);keys=['all_cells','interior_cells','boundary_cells']
    old=json.loads((Path(json.loads((source/'config.json').read_text())['source'])/'results.json').read_text())
    a=[100*old[k]['relative_mass_weighted_rate_error'] for k in keys]
    b=[100*r['hybrid_'+k]['relative_mass_weighted_rate_error'] for k in keys]
    ax.bar(x-.18,a,width=.36,label='Saved graph');ax.bar(x+.18,b,width=.36,label='Hybrid FV')
    for xx,vals in [(x-.18,a),(x+.18,b)]:
        for pos,val in zip(xx,vals):ax.annotate(f'{val:.2f}',(pos,val),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(x,['All 128','Interior 60','Boundary 68']);ax.set_ylabel('Relative mass-weighted rate error [%]')
    ax.set_ylim(0,85);ax.set_title('Rate accuracy, not terminal concentration');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];limit=float(np.max(abs(derivative)))
    pc=PolyCollection(quads,array=derivative,cmap='RdBu_r',clim=(-limit,limit),edgecolors='.6',linewidths=.3)
    ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');bad=r['basis_initial_derivative_counterexample'];j=bad['initial_cell'];i=bad['observed_cell']
    ax.scatter(*centers[j],marker='o',facecolors='none',edgecolors='black',label=f'Initial unit cell {j}')
    ax.scatter(*centers[i],marker='x',c='black',label=f'Observed zero cell {i}')
    ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title(f'Basis derivative: Q[{i},{j}]={bad["raw_initial_derivative_per_s"]:.4f}/s')
    ax.legend(fontsize=8,loc='upper left');fig.colorbar(pc,ax=ax,label='Initial derivative [1/s]',shrink=.85)
    fig.suptitle('Constitutive real-face baseline: improved frozen accuracy, positivity fails; fluid ORC disabled',fontsize=11)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
