"""Plot the two stored supports and independently verified moment obstructions."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);result=json.loads((source/'results.json').read_text());audit=json.loads((source/'drift-certificates.json').read_text())
    z=np.load(source/'mass-moment-cones.npz');fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ax=axes[0,0]
    for name,label,style in [('geometry_selected_wide','Geometry-selected wide: 38 neighbours','o'),('narrow','Adjacent columns: 8 neighbours','x')]:
        d=z[name+'_displacements'];ax.scatter(d[:,0],d[:,1],marker=style,s=35,label=label,alpha=.8)
    ax.scatter([0],[0],color='black',s=35);ax.axhline(-.5,color='gray',linestyle=':',label='Virtual lower row')
    ax.set_xlabel('Corotated physical dx [m]');ax.set_ylabel('dy [m]');ax.set_aspect('equal');ax.set_title('One selected column; bulk extension only');ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(.01,.72))
    ax=axes[0,1]
    for i,name in enumerate(('narrow','geometry_selected_wide')):
        a,b=result['supports'][name]['cross_ratio_separator']['cross_ratio_interval']
        ax.plot([a,b],[i,i],'o-',lw=5,alpha=.65)
    ax.axvline(result['target_cross_transverse_ratio'],color='black',linestyle='--',label='Physical target')
    ax.set_yticks([0,1],['Adjacent columns','Geometry-selected wide']);ax.set_xlabel('Cross / transverse quadratic moment');ax.set_title('Widening resolves this necessary direction bound');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];h=audit['actual_transverse_spacing_m'];beta=abs(result['target_drift'][1]);Gamma=result['target_centered_quadratic_derivative'][1][1]
    ax.plot([0,1],[Gamma,h*beta],'o',ms=8);ax.set_xticks([0,1],['Physical Gamma_yy','Minimum h |beta_y|'])
    ax.set_xlim(-.4,1.4);ax.set_ylim(Gamma-.0002,h*beta+.0002)
    ax.set_ylabel('Centered quadratic derivative [m^2/s] (zoom)');ax.set_title('Drift requires more variance than the target')
    for i,v in enumerate((Gamma,h*beta)):ax.annotate(f'{v:.10f}',(i,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.grid(alpha=.2)
    ax=axes[1,1];key='geometry_selected_wide';values=[audit['certificates'][key]['exact_target_dot_functional']['float']]
    values +=[audit['independent_boundary_target_controls'][key][str(order)]['exact_target_dot_functional']['float'] for order in (32,64)]
    ax.plot(range(3),values,'o',ms=8);ax.axhline(0,color='black',lw=1);ax.set_xticks(range(3),['Mass primitive','Boundary Gauss 32','Boundary Gauss 64'])
    ax.set_ylabel('Separating functional on target [m^2/s]');ax.set_title('Negative gap survives independent target controls')
    ax.text(.05,.18,'Gap ~ -4.33e-4\nTarget controls differ by ~1e-16',transform=ax.transAxes,fontsize=10);ax.grid(alpha=.2)
    fig.suptitle('Bulk exact-moment diagnostic only: no new closed 64-cell backend',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
