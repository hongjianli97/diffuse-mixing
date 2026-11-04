"""Read-only plots and tables of actual N64/128/256 observations; no fitting."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-shear-union64';r=read(out/'results.json')
    rows=[dict(N=int(N),**row) for N,items in r['per_grid_observations'].items() for row in items]
    with (out/'grid-observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fields=['coarse_N','fine_N','time','coarse_common_error','fine_common_error','common_gain','common_indicator','common_decision',
      'native_coarse_error','native_fine_error','native_gain','native_indicator','native_decision',
      'coarse_H_common','fine_H_common','reference_H_common','H_gain','H_indicator','H_decision',
      'common_error_ratio','native_error_ratio','common_observed_log2_ratio','native_observed_log2_ratio']
    with (out/'adjacent-comparisons.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v[k] for k in fields} for v in r['adjacent_comparisons'])
    fig,ax=plt.subplots(1,3,figsize=(13,3.8));plot={}
    for t,color in [(.025,'tab:blue'),(.05,'tab:orange')]:
        points=[next(v for v in r['per_grid_observations'][str(N)] if v['time']==t) for N in (64,128,256)]
        native=[v['native_error_common_denominator'] for v in points];common=[v['common_error'] for v in points]
        plot[str(t)]={'N':[64,128,256],'native_error':native,'common_N64_error':common,
          'common_H_error':[abs(v['common_H']-v['reference_common_H']) for v in points]}
        ax[0].loglog([64,128,256],native,'o-',color=color,label=f't={t:g}, native')
        ax[0].loglog([64,128,256],common,'s--',color=color,label=f't={t:g}, common64')
    last=[v for v in r['adjacent_comparisons'] if v['time']==.05]
    for a,label,title in [(ax[1],'native','Native error gain / fixed indicator'),(ax[2],'H','Common H gain / fixed indicator')]:
        x=[0,1];gain=[v[label+'_gain'] for v in last];budget=[v[label+'_indicator'] for v in last]
        a.bar([v-.18 for v in x],gain,width=.36,label='raw gain');a.bar([v+.18 for v in x],budget,width=.36,label='indicator')
        a.set_xticks(x,['64 -> 128','128 -> 256']);a.set_title(title);a.ticklabel_format(axis='y',style='sci',scilimits=(0,0));a.legend(fontsize=8)
    ax[0].set_title('Actual K16 errors, same initial norm');ax[0].set_xlabel('N');ax[0].set_ylabel('mass norm / initial fluctuation');ax[0].legend(fontsize=7);ax[0].grid(alpha=.3)
    fig.suptitle('One given smooth shear; no extrapolation; Stage 3 incomplete',fontsize=10);fig.tight_layout()
    fig.savefig(out/'union64-comparison.png',dpi=180);fig.savefig(out/'union64-comparison.svg');plt.close(fig)
    save(out/'plot-data.json',{'source':'results.json','raw_points':plot,'fits_or_extrapolated_states':False,
      'native_and_common_errors_use_same_union_initial_denominator':True,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
