"""Raw candidate results and the independent homogeneous support obstruction."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);rows=json.loads((source/'results.json').read_text())['frozen_snapshots']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);t=np.array([r['time'] for r in rows])
    ax=axes[0,0]
    ax.semilogy(t,[r['linear_patch_with_matching_Neumann_data']['maximum_joint_basis_relative_flux_error'] for r in rows],'o-',label='Matching boundary data')
    ax.semilogy(t,[r['linear_patch_with_boundary_data_disabled_incompatible_control']['maximum_joint_basis_relative_flux_error'] for r in rows],'s--',label='Boundary data disabled (incompatible)')
    ax.set_xlabel('Frozen time [s]');ax.set_ylabel('Max joint-basis face-flux error');ax.set_title('Linear consistency: boundary data are explicit');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1]
    z=np.load(source/'mpfa-t0.05.npz');response=z['basis_response'];counter=rows[1]['closed_operator']['basis_counterexample']
    ax.plot(np.arange(len(response)),response,'o',ms=3);ax.axhline(0,color='black',lw=.7)
    i=counter['observed_cell'];j=counter['initial_cell'];ax.plot(i,response[i],'ro',label=f'c[{i}] = {response[i]:.4g}')
    ax.set_ylim(min(response)*1.25,abs(min(response))*3);ax.set_xlabel('Cell index');ax.set_ylabel('Raw mass fraction (near-zero zoom)')
    ax.set_title(f'Closed exp(1e-4 Q) e_{j}: negative response');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];control=json.loads((source/'stencil-moments.json').read_text());m=np.load(source/'stencil-moments.npz')
    cell=int(m['observed_cell']);d=m['displacements'];rates=m['Q'][cell];js=np.flatnonzero(rates);js=js[js!=cell]
    colors=np.where(rates[js]<0,'C3','C0');ax.axhline(0,color='.7',lw=.5);ax.axvline(0,color='.7',lw=.5)
    for a in js:ax.plot([0,d[a,0]],[0,d[a,1]],color='C3' if rates[a]<0 else 'C0',alpha=.4)
    ax.scatter(d[js,0],d[js,1],c=colors,s=50);ax.scatter([0],[0],c='black',s=25)
    ax.text(.03,.97,'Blue: positive rate\nRed: negative rate',transform=ax.transAxes,va='top',fontsize=9)
    ax.set_aspect('equal');ax.set_xlabel('dx [m]');ax.set_ylabel('dy [m]');ax.set_title('Homogeneous control: signed interior rates [1/s]')
    ax=axes[1,1];cert=control['certificate'];reach=np.arange(1,7)
    low=cert['shear']-reach*cert['hx']/cert['hy'];high=cert['shear']+reach*cert['hx']/cert['hy']
    ax.vlines(reach,low,high,color='C0',lw=6,alpha=.6);ax.axhline(0,color='black',lw=1,label='Isotropic target Axy / Ayy = 0')
    ax.plot(1,low[0],'ro');ax.annotate('One-column reach excludes 0',xy=(1,low[0]),xytext=(1.25,.18),arrowprops={'arrowstyle':'->'},fontsize=9)
    ax.set_xticks(reach);ax.set_xlabel('Column reach R (analytic necessary bound)');ax.set_ylabel('Allowed Axy / Ayy for nonnegative rates')
    ax.set_title('At least 5 columns required by this bound alone');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2)
    fig.suptitle('One MPFA-type candidate: consistency passes, positivity fails; no transient acceptance',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
