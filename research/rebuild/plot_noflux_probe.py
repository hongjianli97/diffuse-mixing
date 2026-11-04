"""Independent nonlinear state, physical rate reference and unchanged graph."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def main(source,output):
    source=Path(source);result=json.loads((source/'results.json').read_text());z=np.load(source/'noflux-probe.npz')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);quads=z['quads'];centers=z['mass_centroids'];state=z['probe_average_order64']
    ax=axes[0,0];pc=PolyCollection(quads,array=state,cmap='viridis',edgecolors='.7',linewidths=.3,clim=(.5,.75))
    ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    ax.set_title('True material-cell mass averages');fig.colorbar(pc,ax=ax,label='Manufactured concentration',shrink=.85)
    isolated=[v['child'] for v in result['isolated_cells']];ax.scatter(centers[isolated,0],centers[isolated,1],s=30,marker='x',c='red',label='Saved isolated cells');ax.legend(fontsize=8,loc='upper left')
    ax=axes[0,1];ref=z['face_rate_order64'];graph=z['graph_rate'];boundary=z['boundary_cell_mask']
    ax.scatter(ref[~boundary],graph[~boundary],s=16,label='Interior');ax.scatter(ref[boundary],graph[boundary],s=16,label='Boundary')
    ax.scatter(ref[isolated],graph[isolated],s=40,marker='x',c='red',label='Isolated boundary')
    a=min(ref.min(),graph.min());b=max(ref.max(),graph.max());ax.plot([a,b],[a,b],'k--',lw=.8)
    ax.set_xlabel('Physical face-divergence rate [1/s]');ax.set_ylabel('Unmodified graph rate [1/s]')
    ax.set_title('Instantaneous action on one independent state');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];groups=['all_cells','interior_cells','boundary_cells'];values=[result[k]['relative_mass_weighted_rate_error']*100 for k in groups]
    ax.bar(range(3),values,color=['.5','C0','C1']);ax.set_xticks(range(3),['All 128\ncells','60 interior\ncells','68 boundary\ncells'])
    for i,v in enumerate(values):ax.annotate(f'{v:.2f}%',(i,v),xytext=(0,5),textcoords='offset points',ha='center')
    ax.set_ylabel('Relative mass-weighted rate error [%]');ax.set_title('Independent physical rate error');ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];error=z['rate_error'];limit=float(np.max(abs(error)))
    pc=PolyCollection(quads,array=error,cmap='RdBu_r',edgecolors='.7',linewidths=.3,clim=(-limit,limit))
    ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    ax.set_title('Signed graph minus physical rate');fig.colorbar(pc,ax=ax,label='Rate error [1/s]',shrink=.85)
    fig.suptitle('One frozen no-flux probe: independent reference, saved graph unchanged; fluid ORC disabled',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
