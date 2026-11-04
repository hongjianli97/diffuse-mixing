"""Only actual saved two-grid observations, native errors and fixed indicators."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-compression-spatial-fv';r=read(out/'results.json')
    rows=[dict(N=int(N),**row) for N,items in r['per_grid_observations'].items() for row in items]
    with (out/'grid-observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fields=['time','coarse32_error_common','fine32_error_common','common_gain','common_indicator','common_decision',
      'native_coarse32_error','native_fine32_error','native_gain','native_indicator','native_decision',
      'coarse32_H_common','fine32_H_common','reference_H_common','H_gain','H_indicator','H_decision','common_error_ratio','native_error_ratio']
    with (out/'comparison-observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v[k] for k in fields} for v in r['comparison_observations'])
    with (out/'time-controls.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r['time_controls'][0]));w.writeheader();w.writerows(r['time_controls'])
    fig,ax=plt.subplots(1,3,figsize=(13,3.8));raw={}
    for i,t in [(1,.025),(2,.05)]:
        values=[r['per_grid_observations'][str(N)][i] for N in (128,256)]
        native=[v['native_error_common_denominator'] for v in values];common=[v['common_error'] for v in values]
        ax[0].semilogy([128,256],native,'o-',label=f't={t:g}, native')
        ax[0].semilogy([128,256],common,'s--',label=f't={t:g}, common')
        raw[str(t)]={'N':[128,256],'native_errors':native,'common_errors':common}
    ax[0].set_xticks([128,256]);ax[0].set_xlabel('N per direction');ax[0].set_ylabel('mass error / original parent initial fluctuation');ax[0].set_title('Actual32, same original denominator');ax[0].legend(fontsize=7);ax[0].grid(alpha=.25)
    active=r['comparison_observations'][1:];x=[0,1]
    for a,label,title in [(ax[1],'native','Native error gain / fixed indicator'),(ax[2],'H','Common H gain / fixed indicator')]:
        a.bar([v-.18 for v in x],[v[label+'_gain'] for v in active],width=.36,label='raw gain')
        a.bar([v+.18 for v in x],[v[label+'_indicator'] for v in active],width=.36,label='fixed indicator')
        a.set_xticks(x,['t=.025','t=.05']);a.set_title(title);a.ticklabel_format(axis='y',style='sci',scilimits=(0,0));a.legend(fontsize=8)
    fig.suptitle('One given smooth compression; no fitted or extrapolated states; Stage 3 incomplete',fontsize=10);fig.tight_layout()
    fig.savefig(out/'compression-spatial-fv.png',dpi=180);fig.savefig(out/'compression-spatial-fv.svg');plt.close(fig)
    save(out/'plot-data.json',{'source':'results.json','raw_points':raw,'fits_or_extrapolated_states':False,
      'native_and_common_same_denominator':True,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
