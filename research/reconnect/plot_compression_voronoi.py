"""Saved actual trajectories only: fixed-time control, errors and H; no fits."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-compression-voronoi-time';r=read(out/'results.json')
    rows=[dict(K=int(K),**row) for K,items in r['observations_by_intervals'].items() for row in items]
    flat=[]
    for row in rows:
        v={k:x for k,x in row.items() if not isinstance(x,dict)}
        for prefix in ('candidate','reference'):
            for key,value in row[prefix+'_metrics'].items():v[prefix+'_'+key]=value
        flat.append(v)
    with (out/'FV-observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    fields=['time','FV16_error','saved_kernel8_error','state_gain','state_indicator','state_decision',
      'FV16_H','saved_kernel8_H','reference_H','FV16_H_signed_error','H_gain','H_indicator','H_decision','saved_kernel8_over_FV16_error_ratio']
    with (out/'comparison-observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v[k] for k in fields} for v in r['comparison_observations'])
    with (out/'time-controls.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r['time_controls'][0]));w.writeheader();w.writerows(r['time_controls'])
    fig,axes=plt.subplots(1,3,figsize=(13,3.8));raw={}
    for i,t in [(1,.025),(2,.05)]:
        errors=[r['observations_by_intervals'][str(K)][i]['relative_concentration_error'] for K in (4,8,16)]
        axes[0].loglog([4,8,16],errors,'o-',label=f't={t:g}')
        raw[str(t)]={'intervals':[4,8,16],'actual_FV_errors':errors}
    axes[0].set_xlabel('Actual intervals K');axes[0].set_ylabel('mass error / original initial fluctuation');axes[0].set_title('Three actual time discretizations');axes[0].legend();axes[0].grid(alpha=.3)
    actual=r['comparison_observations'][1:];x=[0,1]
    for ax,label,title in [(axes[1],'state','Concentration gain / fixed indicator'),(axes[2],'H','H gain / fixed indicator')]:
        ax.bar([v-.18 for v in x],[v[label+'_gain'] for v in actual],width=.36,label='raw gain')
        ax.bar([v+.18 for v in x],[v[label+'_indicator'] for v in actual],width=.36,label='indicator')
        ax.set_xticks(x,['t=.025','t=.05']);ax.set_title(title);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(fontsize=8)
    fig.suptitle('Given smooth compression, same ordinary Voronoi FV; Stage 3 incomplete',fontsize=10);fig.tight_layout()
    fig.savefig(out/'compression-time-comparison.png',dpi=180);fig.savefig(out/'compression-time-comparison.svg');plt.close(fig)
    save(out/'plot-data.json',{'source':'results.json','raw_points':raw,'fits_or_extrapolated_states':False,
      'comparison_uses_saved_original_kernel_only':True,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
