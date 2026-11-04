"""Saved native/common spatial evidence and raw fine-direction response; no new solves."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-affine-shear-spatial';r=read(out/'results.json');levels=(16,32,64)
    observations=[{'N':int(N),**row} for N,values in r['per_grid_observations'].items() for row in values]
    comparisons=[]
    for row in r['adjacent_spatial_comparisons']:
        item={key:row[key] for key in ('coarse_N','fine_N','time')}
        for group in ('gains','fixed_indicators','decisions'):item.update({group+'_'+key:value for key,value in row[group].items()})
        comparisons.append(item)
    temporal=[{'N':int(N),**row} for N,v in r['per_grid_controls'].items() for row in v['time_controls']]
    fine=r['per_grid_controls']['64']['direction_observables']
    fineflat=[{key:value for key,value in row.items() if not isinstance(value,list)} for row in fine]
    for name,rows in [('observations.csv',observations),('spatial-comparisons.csv',comparisons),('time-controls.csv',temporal),('fine-direction-comparisons.csv',fineflat)]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(2,2,figsize=(11,7.4));saved={}
    for t,index,color in ((.025,1,'tab:blue'),(.05,2,'tab:orange')):
        rows=[r['per_grid_observations'][str(N)][index] for N in levels]
        native=[v['native_error_native_denominator'] for v in rows];common=[v['common_error'] for v in rows]
        herror=[abs(v['common_H']-v['common_reference_H']) for v in rows]
        for panel,values in zip((ax[0,0],ax[0,1],ax[1,0]),(native,common,herror)):panel.loglog(levels,values,'o-',base=2,color=color,label=f't={t:g}')
        saved[str(t)]={'native_error_own_initial_fluctuation':native,'common_error_N16_initial_fluctuation':common,'common_H_error':herror}
    for panel in (ax[0,0],ax[0,1],ax[1,0]):
        panel.set_xticks(levels,[str(N) for N in levels]);panel.set_xlabel('material grid side N');panel.legend(fontsize=8);panel.grid(alpha=.25)
    ax[0,0].set_title('Own-PDE error on native material cells');ax[0,0].set_ylabel('mass L2 / own initial fluctuation')
    ax[0,1].set_title('Error on the same N16 material regions');ax[0,1].set_ylabel('mass L2 / original N16 fluctuation')
    ax[1,0].set_title('Common-material H error');ax[1,0].set_ylabel('|actual H - reference H|')
    x=np.arange(2);w=.26;rows=fine[1:]
    for shift,key,label,color in ((-w,'actual_direction_contrast','actual raw response','tab:green'),(0,'reference_direction_contrast','physical reference','tab:blue'),(w,'fixed_positive_contrast_indicator','fixed positive indicator','tab:gray')):
        ax[1,1].bar(x+shift,[v[key] for v in rows],w,label=label,color=color)
    ax[1,1].set_xticks(x,['t=0.025','t=0.05']);ax[1,1].set_ylabel('delta I+ - delta I-');ax[1,1].set_title('N64 two tilted-mode direction contrast');ax[1,1].legend(fontsize=7);ax[1,1].grid(axis='y',alpha=.25);ax[1,1].set_axisbelow(True)
    passed=r['registered_three_space_level_affine_shear_control_passed']
    fig.suptitle(f'Registered affine-shear spatial control: {"PASSED" if passed else "FAILED / UNRESOLVED"}; Stage 3 incomplete',fontsize=12)
    fig.tight_layout();fig.savefig(out/'affine-shear-spatial.png',dpi=180);fig.savefig(out/'affine-shear-spatial.svg');plt.close(fig)
    save(out/'plot-data.json',{'N_values':list(levels),'saved_spatial_errors':saved,'fine_raw_direction_response':fine,
      'fits_clipping_or_extrapolated_states':False,'new_reference_or_candidate_solve':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
