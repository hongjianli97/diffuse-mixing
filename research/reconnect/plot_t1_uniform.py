"""Read-only T1 tables and actual-state comparison plots; no fits."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-t1-uniform';r=read(out/'results.json')
    rows=[dict(N=int(N),**v) for N,vs in r['per_grid_observations'].items() for v in vs]
    for name,items in [('grid-observations.csv',rows),('time-controls.csv',r['time_controls']),('symbol-controls.csv',r['discrete_symbol_controls'])]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    fields=[k for k in r['adjacent_comparisons'][0] if not k.endswith('_components')]
    with (out/'adjacent-comparisons.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v[k] for k in fields} for v in r['adjacent_comparisons'])
    fig,ax=plt.subplots(1,3,figsize=(13.4,3.8));raw={}
    for t,color in [(.025,'tab:blue'),(.05,'tab:orange')]:
        values=[next(v for v in r['per_grid_observations'][str(N)] if v['time']==t) for N in (16,32,64)]
        common=[v['common_error'] for v in values];native=[v['native_error_common_denominator'] for v in values]
        ax[0].semilogy([16,32,64],common,'s-',color=color,label=f't={t:g}, common16')
        ax[0].semilogy([16,32,64],native,'o--',color=color,label=f't={t:g}, native')
        raw[str(t)]={'N':[16,32,64],'common16_error':common,'native_error_common_denominator':native}
    x=list(range(6));labels=[];diffs=[];budgets=[]
    for N in (16,32,64):
        for A,B in ((4,8),(8,16)):
            v=next(v for v in r['time_controls'] if v['N']==N and v['coarse_K']==A and v['fine_K']==B and v['time']==.05)
            labels.append(f'{N}\n{A}/{B}');diffs.append(v['raw_difference_relative']);budgets.append(v['fixed_numerical_indicator_absolute']/r['common_initial_fluctuation_norm'])
    ax[1].semilogy(x,diffs,'o',label='actual time difference');ax[1].semilogy(x,budgets,'s',label='fixed numerical indicator')
    ax[1].set_xticks(x,labels);ax[1].set_xlabel('N and interval pair');ax[1].legend(fontsize=7)
    sym=[];sb=[]
    for N in (16,32,64):
        v=[v for v in r['discrete_symbol_controls'] if v['N']==N]
        sym.append(max(q['actual_vs_discrete_symbol_absolute'] for q in v));sb.append(max(q['fixed_indicator_absolute'] for q in v))
    ax[2].semilogy([16,32,64],sym,'o-',label='actual vs independent symbol')
    ax[2].semilogy([16,32,64],sb,'s--',label='fixed action indicator');ax[2].legend(fontsize=7)
    ax[0].set_title('Actual K16 spatial errors');ax[0].set_xlabel('N');ax[0].set_ylabel('mass norm / common initial fluctuation');ax[0].legend(fontsize=7)
    ax[1].set_title('Static time differences at T=0.05');ax[2].set_title('Independent discrete symbol check');ax[2].set_xlabel('N');ax[2].set_ylabel('absolute mass norm')
    for a in (ax[0],ax[2]):a.set_xticks([16,32,64])
    for a in ax:a.grid(alpha=.25)
    fig.suptitle('T1: one uniform static material-mean sine; Stage 3 incomplete',fontsize=11);fig.tight_layout()
    fig.savefig(out/'t1-uniform.png',dpi=180);fig.savefig(out/'t1-uniform.svg');plt.close(fig)
    save(out/'plot-data.json',{'raw_spatial_points':raw,'fits_or_extrapolated_states':False,'general_temporal_order_claimed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
