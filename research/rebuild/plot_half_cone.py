"""One refined interior row, physical partition and exact positive witness."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def main(source,output):
    source=Path(source);result=json.loads((source/'results.json').read_text());cfg=json.loads((source/'config.json').read_text())
    coarse=Path(cfg['source']);old_audit=json.loads((coarse/'drift-certificates.json').read_text())
    z=np.load(source/'half-transverse-cone.npz');fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    origin=z['origin'];frame=z['axes'];quads=(z['child_record_quads']-origin)@frame
    centers=(z['child_record_centroids']-origin)@frame;selected=int(z['selected_child_index']);neighbours=z['neighbour_child_indices']
    ax=axes[0,0];window=(z['child_column']>=9)&(z['child_column']<=23)&(z['child_row']<=2)
    ax.add_collection(PolyCollection(quads[window],facecolors='none',edgecolors='.75',linewidths=.6))
    ax.scatter(centers[neighbours,0],centers[neighbours,1],s=15,c='.6',label='38 actual neighbours')
    if 'raw_rates' in z:
        active=np.flatnonzero(z['raw_rates']!=0);points=centers[neighbours[active]]
        for p in points:ax.plot([0,p[0]],[0,p[1]],color='C0',alpha=.5)
        ax.scatter(points[:,0],points[:,1],s=40,c='C0',label='5 positive witness rates')
    ax.scatter([0],[0],marker='*',s=90,c='black',label=f'Interior child {selected}')
    ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('Corotated physical dx [m]');ax.set_ylabel('dy [m]')
    ax.set_title('Derived partition; all support cells inside domain');ax.legend(fontsize=8,loc='upper left')
    ax=axes[0,1];values=[old_audit['Gamma_yy_minus_h_abs_beta_y_m2_per_s'],result['drift_necessary_condition']['exact_target_dot_functional']['float']]
    ax.bar([0,1],values,color=['C3','C0'],width=.5);ax.axhline(0,color='black',lw=.7)
    ax.set_xticks([0,1],['Original h = 0.5 m','Declared half h = 0.25 m']);ax.set_ylabel('Gamma_yy - h |beta_y| [m^2/s]')
    ax.set_title('Necessary drift obstruction removed')
    for i,v in enumerate(values):ax.annotate(f'{v:.6g}',(i,v),xytext=(0,-15 if v<0 else 5),textcoords='offset points',ha='center',fontsize=9)
    ax.set_ylim(-.003,.024);ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    if 'raw_rates' in z:
        rates=z['raw_rates'];active=np.flatnonzero(rates!=0);labels=z['support_neighbour_column_and_virtual_row'][active]
        ax.bar(range(len(active)),rates[active],color='C0');ax.set_xticks(range(len(active)),[f'{j} / {q:+d}' for j,q in labels])
        for i,v in enumerate(rates[active]):ax.annotate(f'{v:.6f}',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xlabel('Stream column / relative row');ax.set_ylabel('Unmodified LP witness rate [1/s]');ax.set_title('Independent rational basis check passed')
    ax.set_ylim(0,float(z['raw_rates'].max())*1.18 if 'raw_rates' in z else 1);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];parent=cfg['selected_parent'];S=z['child_record_parent_covariance'][parent];C=z['child_record_covariance'][selected]
    v=frame[:,1];offset=z['child_record_centroid_offsets'][selected];parent_y=float(v@S@v);child_y=float(v@C@v);between=float((offset@v)**2)
    ax.bar([0],[parent_y],color='.6',width=.5,label='Parent covariance')
    ax.bar([1],[child_y],color='C0',width=.5,label='Within-child covariance');ax.bar([1],[between],bottom=[child_y],color='C1',width=.5,label='Between-child centroids')
    ax.set_xticks([0,1],['Parent','Mass-weighted child decomposition']);ax.set_ylabel('Transverse mass covariance [m^2]')
    ax.set_title('Quadratic average restriction keeps both terms');ax.legend(fontsize=8,loc='upper right');ax.set_ylim(0,.033);ax.grid(axis='y',alpha=.2)
    fig.suptitle('One refined interior moment row: feasible certificate, no global diffusion acceptance',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
