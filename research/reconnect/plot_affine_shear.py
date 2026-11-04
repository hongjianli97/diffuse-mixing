"""Saved sliding domains, raw mode observations, actual H and direction indicators."""
from fractions import Fraction as F
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save
from shear_union64_study import arrays
from voronoi_time_action import metrics


def main():
    out=ROOT/'runs/reconnect-affine-shear';r=read(out/'results-completed.json');world=read(out/'pure-inputs.json')
    for name,rows in [('observations.csv',r['observations']),('time-controls.csv',r['time_controls']),('frozen-symbol-controls.csv',r['frozen_symbol_controls'])]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    direction=[]
    for row in r['direction_observables']:
        direction.append({'time':row['time'],'actual_Iplus':row['raw_actual_observables'][0],'actual_Iminus':row['raw_actual_observables'][1],
          'reference_Iplus':row['raw_reference_observables'][0],'reference_Iminus':row['raw_reference_observables'][1],
          'actual_direction_contrast':row['actual_direction_contrast'],'reference_direction_contrast':row['reference_direction_contrast'],
          'fixed_positive_contrast_indicator':row['fixed_positive_contrast_indicator'],'direction_contrast_resolved':row['direction_contrast_resolved']})
    with (out/'direction-observables.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(direction[0]));w.writeheader();w.writerows(direction)
    fig,ax=plt.subplots(2,2,figsize=(11.2,7.6));domains={}
    for j,color in ((0,'tab:gray'),(16,'tab:blue'),(32,'tab:orange')):
        frame=world['cases']['16']['frames'][j];s=float(F(frame['shear_exact']));t=float(F(frame['time_exact']))
        corners=np.array([[0,0],[1,0],[1+s,1],[s,1],[0,0]])
        ax[0,0].plot(corners[:,0],corners[:,1],color=color,label=f't={t:g}');domains[str(t)]=corners.tolist()
    ax[0,0].set_aspect('equal');ax[0,0].set_xlabel('world x');ax[0,0].set_ylabel('world y');ax[0,0].set_title('Given sliding domain and periodic lattice');ax[0,0].legend(fontsize=7)
    t=[row['time'] for row in direction]
    for key,ref,color,label in [('actual_Iplus','reference_Iplus','tab:blue','eta=+1'),('actual_Iminus','reference_Iminus','tab:orange','eta=-1')]:
        ax[0,1].plot(t,[row[key] for row in direction],'o-',color=color,label=label+' actual raw I')
        ax[0,1].scatter(t,[row[ref] for row in direction],color=color,marker='x',s=45,label=label+' own PDE reference')
    ax[0,1].set_title('Two tilted modes: raw material observables');ax[0,1].set_xlabel('time');ax[0,1].set_ylabel('sum m (c - mu0) w');ax[0,1].legend(fontsize=7)
    data=arrays(out/'material-inputs.npz');path=arrays(out/'k16/trajectory-states.npz');times=np.arange(17)/320
    H=[metrics(data['mass'],c,data['c0'])['H'] for c in path['states']]
    ax[1,0].plot(times,H,label='actual H');ax[1,0].scatter(t,[row['reference_H'] for row in r['observations']],marker='x',color='black',label='own material PDE reference')
    ax[1,0].set_title('Actual whole-state mixing H');ax[1,0].set_xlabel('time');ax[1,0].set_ylabel('H');ax[1,0].legend(fontsize=8)
    x=np.arange(2);width=.31;rows=direction[1:]
    ax[1,1].bar(x-width/2,[row['actual_direction_contrast'] for row in rows],width,color='tab:green',label='actual response contrast')
    ax[1,1].bar(x+width/2,[row['fixed_positive_contrast_indicator'] for row in rows],width,color='tab:gray',label='fixed positive indicator')
    ax[1,1].scatter(x-width/2,[row['reference_direction_contrast'] for row in rows],marker='x',color='black',label='PDE reference contrast')
    ax[1,1].set_xticks(x,['t=0.025','t=0.05']);ax[1,1].set_title('Resolved shear direction response');ax[1,1].set_ylabel('raw I change contrast');ax[1,1].legend(fontsize=7)
    for panel in ax.flat:panel.grid(alpha=.2);panel.set_axisbelow(True)
    fig.suptitle('Registered N32 affine shear control; Stage 3 incomplete',fontsize=12);fig.tight_layout()
    fig.savefig(out/'affine-shear.png',dpi=180);fig.savefig(out/'affine-shear.svg');plt.close(fig)
    save(out/'plot-data.json',{'source_results':'results-completed.json','world_domains':domains,'direction_observables':direction,'actual_H':H,'time':times.tolist(),
      'fits_state_projection_or_normalization':False,'original_pre_completion_files_changed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
