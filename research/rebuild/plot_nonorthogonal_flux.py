"""Plot raw nonorthogonal patch errors and the retained sign counterexample."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def main(source,output):
    source=Path(source);rows=json.loads((source/'results.json').read_text())['frozen_snapshots']
    time=np.array([r['time'] for r in rows]);fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    record=np.load(source/'patch-t0.05.npz');positions=record['positions'];centers=record['mass_centroids']
    nx=32;cells=np.array([[j*(nx+1)+i,j*(nx+1)+i+1,(j+1)*(nx+1)+i+1,(j+1)*(nx+1)+i] for j in range(2) for i in range(nx)])
    ax=axes[0,0];ax.add_collection(PolyCollection(positions[cells],facecolors='none',edgecolors='.55',linewidths=.55))
    ax.scatter(centers[:,0],centers[:,1],s=5,color='black',label='True mass centroids')
    a,b=record['edge_owner'],record['edge_other'];e=len(a)//2;mid=(centers[a[e]]+centers[b[e]])/2
    d=record['mass_centroid_displacement'][e];n=record['face_normal'][e]
    ax.quiver(*mid,*(.2*d/np.linalg.norm(d)),angles='xy',scale_units='xy',scale=1,color='C1',label='Centroid displacement')
    ax.quiver(*mid,*(.2*n),angles='xy',scale_units='xy',scale=1,color='C0',label='Face normal')
    ax.autoscale();ax.set_aspect('equal');ax.set_title('Same 64 cells, frozen t = 0.05 s');ax.legend(fontsize=8,loc='upper left')
    ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    ax=axes[0,1]
    for method,label in [('normal_distance','Two-point: normal distance'),('center_distance','Two-point: centroid distance'),('linear_reconstruction_reference','Linear reconstruction reference')]:
        error=[r['methods'][method]['linear_patch']['maximum_joint_basis_relative_flux_error'] for r in rows]
        ax.plot(time,error,'o-',label=label)
    ax.plot(time,[r['maximum_any_scalar_conductance_error_lower_bound'] for r in rows],'k--',label='Any scalar conductance: lower bound')
    ax.set_xlabel('Frozen time [s]');ax.set_ylabel('Max joint-basis face-flux error');ax.set_title('Face consistency, independent of density closure')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];raw=record['reconstruction_Q'].copy();np.fill_diagonal(raw,np.nan)
    limit=np.nanmax(abs(raw));im=ax.imshow(raw,cmap='RdBu_r',vmin=-limit,vmax=limit,origin='lower',interpolation='nearest')
    ax.set_xlabel('Input cell j');ax.set_ylabel('Observed cell i');ax.set_title('Reference generator off-diagonals [1/s]')
    fig.colorbar(im,ax=ax,shrink=.8)
    ax=axes[1,1];response=record['reconstruction_basis_response'];audit=rows[1]['methods']['linear_reconstruction_reference']['closed_operator']
    counter=audit['basis_counterexample'];i=counter['observed_cell'];j=counter['initial_cell']
    ax.plot(np.arange(len(response)),response,'o',ms=3);ax.axhline(0,color='black',lw=.7)
    ax.plot(i,response[i],'ro',label=f'c[{i}] = {response[i]:.3g}')
    ax.set_ylim(min(response)*1.3,abs(min(response))*3);ax.set_xlabel('Cell index');ax.set_ylabel('Raw mass fraction (zoom near zero)')
    ax.set_title(f'exp(1e-4 Q) e_{j}: negative response retained');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Frozen diagnostic only: no 2D transient acceptance; fluid ORC disabled',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
