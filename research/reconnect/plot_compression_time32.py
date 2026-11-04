"""Read-only actual four-time-level data and necessary16/32 control, no fits."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-compression-time32';r=read(out/'results.json')
    rows=[]
    for K,items in r['observations_by_intervals'].items():
        for row in items:
            v=dict(K=int(K),**{k:x for k,x in row.items() if not isinstance(x,dict)})
            for prefix in ('candidate','reference'):
                for key,value in row[prefix+'_metrics'].items():v[prefix+'_'+key]=value
            rows.append(v)
    with (out/'actual-observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fields=['time','FV32_error','saved_kernel8_error','state_gain','state_indicator','state_decision',
      'FV32_H','saved_kernel8_H','reference_H','FV32_H_signed_error','H_gain','H_indicator','H_decision','saved_kernel8_over_FV32_error_ratio']
    with (out/'comparison-observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v[k] for k in fields} for v in r['comparison_observations'])
    with (out/'time-controls.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r['time_controls'][0]));w.writeheader();w.writerows(r['time_controls'])
    fig,ax=plt.subplots(1,3,figsize=(13,3.8));raw={}
    for i,t in [(1,.025),(2,.05)]:
        errors=[r['observations_by_intervals'][str(K)][i]['relative_concentration_error'] for K in (4,8,16,32)]
        controls=[r['time_controls'][i][f'FV{C}_{N}_state_difference'] for C,N in ((4,8),(8,16),(16,32))]
        ax[0].loglog([4,8,16,32],errors,'o-',label=f't={t:g}')
        ax[1].semilogy([0,1,2],controls,'o-',label=f't={t:g}')
        raw[str(t)]={'intervals':[4,8,16,32],'actual_errors':errors,'raw_state_differences':controls}
    ax[0].set_xticks([4,8,16,32],['4','8','16','32']);ax[0].set_xlabel('Actual intervals K');ax[0].set_ylabel('mass error / original initial fluctuation');ax[0].set_title('Old4/8/16 reused, new32 actual')
    ax[1].set_xticks([0,1,2],['4 / 8','8 / 16','16 / 32']);ax[1].set_title('Observed time differences, no extrapolation')
    active=r['comparison_observations'][1:];x=[0,1]
    ax[2].bar([v-.18 for v in x],[v['state_gain'] for v in active],width=.36,label='raw gain')
    ax[2].bar([v+.18 for v in x],[v['state_indicator'] for v in active],width=.36,label='fixed indicator')
    ax[2].set_xticks(x,['t=.025','t=.05']);ax[2].set_title('FV32 vs saved kernel8');ax[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for a in ax:a.legend(fontsize=8);a.grid(alpha=.25)
    fig.suptitle('Same given smooth compression; raw states only; Stage 3 incomplete',fontsize=10);fig.tight_layout()
    fig.savefig(out/'compression-time32.png',dpi=180);fig.savefig(out/'compression-time32.svg');plt.close(fig)
    save(out/'plot-data.json',{'source':'results.json','raw_points':raw,'fits_or_extrapolated_states':False,
      'old_candidate_reruns':0,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
