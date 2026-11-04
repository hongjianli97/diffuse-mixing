"""Saved-data plots of distinct physical PDEs and the retained H-inconclusive checkpoint."""
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
    out=ROOT/'runs/reconnect-t3-affine';r=read(out/'results.json');d=arrays(out/'material-inputs.npz')
    models=('constant_D','constant_Gamma');colors=('tab:blue','tab:orange');labels=('constant D','constant Gamma')
    observations=[{'model':model,**row} for model in models for row in r['per_model_observations'][model]]
    time=[{'model':model,**row} for model in models for row in r['time_controls'][model]]
    comparison=[{k:v for k,v in row.items() if k!='indicator_components_absolute'} for row in r['model_comparisons']]
    for name,rows in [('observations.csv',observations),('time-controls.csv',time),('model-comparison.csv',comparison),('symbol-controls.csv',r['symbol_controls'])]:
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figure,ax=plt.subplots(2,2,figsize=(11,7.6));t=np.arange(17)*float(F(1,320));saved={}
    for model,color,label in zip(models,colors,labels):
        path=arrays(out/model/'k16/trajectory-states.npz');h=[metrics(d['mass'],c,d['c0'])['H'] for c in path['states']]
        reference=[v['reference_H'] for v in r['per_model_observations'][model]]
        ax[0,0].plot(t,h,color=color,label=label+' actual');ax[0,0].scatter([0,.025,.05],reference,color=color,marker='x',s=45,label=label+' PDE reference')
        saved[model]={'time':t.tolist(),'actual_H':h,'reference_H':reference}
    ax[0,0].set_title('Distinct physical models: actual H vs own PDE');ax[0,0].set_xlabel('time');ax[0,0].set_ylabel('H');ax[0,0].legend(fontsize=7)
    x=np.arange(2);width=.32
    for offset,model,color,label in zip((-width/2,width/2),models,colors,labels):
        rows=r['per_model_observations'][model][1:]
        ax[0,1].bar(x+offset,[q['reference_error_relative'] for q in rows],width,color=color,label=label+' actual error')
        ax[0,1].plot(x+offset,[q['fixed_total_indicator_relative'] for q in rows],'_',color='black',markersize=18)
    ax[0,1].set_title('Own PDE error; black mark = fixed indicator');ax[0,1].set_ylabel('mass L2 / initial fluctuation');ax[0,1].set_xticks(x,['t=0.025','t=0.05']);ax[0,1].legend(fontsize=7)
    rows=r['model_comparisons'][1:]
    for panel,value,indicator,title,ylabel in [
      (ax[1,0],'model_state_difference_relative','fixed_state_indicator_relative','Model state response: both checkpoints resolved','mass L2 / initial fluctuation'),
      (ax[1,1],'model_H_difference','fixed_H_indicator','Model H response: t=0.025 INCONCLUSIVE','H(constant D) - H(constant Gamma)')]:
        panel.bar(x-width/2,[q[value] for q in rows],width,label='actual model difference',color='tab:green')
        panel.bar(x+width/2,[q[indicator] for q in rows],width,label='fixed comparison indicator',color='tab:gray')
        panel.set_xticks(x,['t=0.025','t=0.05']);panel.set_title(title);panel.set_ylabel(ylabel);panel.legend(fontsize=7)
    for panel in ax.flat:panel.grid(axis='y',alpha=.25);panel.set_axisbelow(True)
    figure.suptitle('T3 planar affine pair: registered case unresolved; Stage 3 incomplete',fontsize=12)
    figure.tight_layout();figure.savefig(out/'t3-affine.png',dpi=180);figure.savefig(out/'t3-affine.svg');plt.close(figure)
    save(out/'plot-data.json',{'H_paths_and_own_references':saved,'checkpoint_model_comparisons':comparison,
      'fits_clipping_or_extrapolated_states':False,'unresolved_checkpoint_retained':True,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'})


if __name__=='__main__':main()
