"""Saved three-space-level errors and fine model response; no fits or extrapolated states."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-t3-affine-spatial';r=read(out/'results.json');levels=(16,32,64);models=('constant_D','constant_Gamma')
    observations=[]
    for key,values in r['per_grid_observations'].items():
        for row in values:observations.append({'grid_model':key,**row})
    spatial=[]
    for row in r['adjacent_spatial_comparisons']:
        item={key:row[key] for key in ('model','coarse_N','fine_N','time')}
        for group in ('gains','fixed_indicators','decisions'):
            item.update({group+'_'+key:value for key,value in row[group].items()})
        spatial.append(item)
    temporal=[]
    for N,result in r['per_grid_controls'].items():
        for model,values in result['time_controls'].items():
            temporal.extend({'N':int(N),'model':model,**row} for row in values)
    fine=[{key:value for key,value in row.items() if key!='indicator_components_absolute'} for row in r['per_grid_controls']['64']['model_comparisons']]
    for name,rows in [('observations.csv',observations),('spatial-comparisons.csv',spatial),('time-controls.csv',temporal),('fine-model-comparisons.csv',fine)]:
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(2,2,figsize=(11,7.4));values={}
    for model,color,label in zip(models,('tab:blue','tab:orange'),('constant D','constant Gamma')):
        rows=[r['per_grid_observations'][f'n{N}_{model}'][-1] for N in levels]
        native=[q['native_error_native_denominator'] for q in rows];common=[q['common_error'] for q in rows]
        herror=[abs(q['common_H']-q['common_reference_H']) for q in rows]
        ax[0,0].loglog(levels,native,'o-',base=2,color=color,label=label)
        ax[0,1].loglog(levels,common,'o-',base=2,color=color,label=label)
        ax[1,0].loglog(levels,herror,'o-',base=2,color=color,label=label)
        values[model]={'native_final_error_own_initial_fluctuation':native,'common_final_error_N16_initial_fluctuation':common,'common_final_H_error':herror}
    for panel in (ax[0,0],ax[0,1],ax[1,0]):
        panel.set_xticks(levels,[str(N) for N in levels]);panel.set_xlabel('material grid side N');panel.legend(fontsize=8);panel.grid(alpha=.25)
    ax[0,0].set_title('Final own-PDE error on native material cells');ax[0,0].set_ylabel('mass L2 / own initial fluctuation')
    ax[0,1].set_title('Final error on the same N16 material regions');ax[0,1].set_ylabel('mass L2 / original N16 fluctuation')
    ax[1,0].set_title('Final common-material H error');ax[1,0].set_ylabel('|actual H - own PDE reference H|')
    rows=r['per_grid_controls']['64']['model_comparisons'][1:];x=np.arange(2);width=.34
    ax[1,1].bar(x-width/2,[q['model_H_difference'] for q in rows],width,label='actual fine-grid H model gap',color='tab:green')
    ax[1,1].bar(x+width/2,[q['fixed_H_indicator'] for q in rows],width,label='unchanged original H indicator',color='tab:gray')
    ax[1,1].set_xticks(x,['t=0.025','t=0.05']);ax[1,1].set_ylabel('H(constant D) - H(constant Gamma)');ax[1,1].set_title('N64 model response; N32 decision retained')
    ax[1,1].legend(fontsize=7);ax[1,1].grid(axis='y',alpha=.25);ax[1,1].set_axisbelow(True)
    passed=r['T3_registered_three_space_level_affine_control_passed'];fig.suptitle(f'T3 registered affine spatial control: {"PASSED" if passed else "FAILED / UNRESOLVED"}; Stage 3 incomplete',fontsize=12)
    fig.tight_layout();fig.savefig(out/'t3-affine-spatial.png',dpi=180);fig.savefig(out/'t3-affine-spatial.svg');plt.close(fig)
    save(out/'plot-data.json',{'N_values':list(levels),'per_model_final_errors':values,'fine_H_model_comparisons':rows,
      'original_N32_inconclusive_retained':True,'fits_clipping_or_extrapolated_states':False,'new_reference_or_candidate_solve':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
