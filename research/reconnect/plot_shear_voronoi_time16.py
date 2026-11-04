"""Plot only saved actual states on one unchanged common material observation."""
from pathlib import Path
import argparse,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from voronoi_time_action import metrics


def main(directory,output=None):
    source=Path(directory).resolve();out=source if output is None else Path(output).resolve()
    if out!=source:out.mkdir(parents=True,exist_ok=False)
    for name in ['time-control.png','time-control.svg','plot-values.json']:assert not (out/name).exists()
    with np.load(source/'original-comparison.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ['parent_mass','parent_c0','parent_reference_states','parent_FV_k4_states','parent_FV_k8_states']}
    with np.load(source/'source-spatial-states.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in ['k4_parent_states','k8_parent_states']}
    with np.load(source/'n128/trajectory-states.npz',allow_pickle=False) as z:coarse16=z['states'][-1].copy()
    with np.load(source/'common-observations.npz',allow_pickle=False) as z:fine16=z['fine_parent_states16'][-1].copy()
    m,c0,ref=d['parent_mass'],d['parent_c0'],d['parent_reference_states'][-1]
    sigma=math.sqrt(metrics(m,c0,c0)['variance']);href=metrics(m,ref,c0)['H'];values={}
    for N,states in [(128,[d['parent_FV_k4_states'][-1],d['parent_FV_k8_states'][-1],coarse16]),
                     (256,[old['k4_parent_states'][-1],old['k8_parent_states'][-1],fine16])]:
        values[str(N)]=[{'K':K,'concentration_error':float(np.sqrt(m@((c-ref)**2))/sigma),
            'H_absolute_error':abs(metrics(m,c,c0)['H']-href)} for K,c in zip([4,8,16],states)]
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),constrained_layout=True)
    for ax,key,title in zip(axes,['concentration_error','H_absolute_error'],['Concentration error','Absolute H error']):
        for N,color in [(128,'#2463a5'),(256,'#d15f19')]:
            ax.plot([4,8,16],[v[key] for v in values[str(N)]],'-o',color=color,label=f'N={N}')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([4,8,16],['4','8','16'])
        ax.set_xlabel('Ordered midpoint intervals');ax.set_title(title);ax.grid(True,which='both',alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Relative to original initial fluctuation');axes[1].set_ylabel('|H - reference H|')
    fig.suptitle('Same saved N128 material regions, T=0.05, D=0.02',fontsize=11)
    fig.savefig(out/'time-control.png',dpi=180);fig.savefig(out/'time-control.svg');plt.close(fig)
    with (out/'plot-values.json').open('x') as f:json.dump({'scope':'saved actual states only; no fit or extrapolation','values':values},f,indent=2);f.write('\n')
    print('Saved common-observation time-control figure; no PDF, fitting or new propagation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--output');a=p.parse_args();main(a.directory,a.output)
