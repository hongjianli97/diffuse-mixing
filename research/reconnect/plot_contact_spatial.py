"""Saved three-grid contact curves; no fitted states or additional solves."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-contact-spatial';r=read(out/'results.json');levels=(16,32,64)
    obs=[{'N':int(N),**v} for N,rows in r['per_grid_observations'].items() for v in rows];spatial=[]
    for row in r['adjacent_spatial_comparisons']:
        v={key:row[key] for key in ('coarse_N','fine_N','time')}
        for field in ('gains','fixed_indicators','decisions'):v.update({field+'_'+k:x for k,x in row[field].items()})
        spatial.append(v)
    for name,rows in [('observations.csv',obs),('spatial-comparisons.csv',spatial)]:
        with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(2,2,figsize=(10.5,7));saved={}
    for j,t,color in ((1,.025,'tab:blue'),(2,.05,'tab:orange')):
        rows=[r['per_grid_observations'][str(N)][j] for N in levels]
        native=[v['native_error_native_denominator'] for v in rows];common=[v['common_error'] for v in rows]
        herror=[abs(v['common_H']-v['common_reference_H']) for v in rows]
        for a,v in zip((ax[0,0],ax[0,1],ax[1,0]),(native,common,herror)):a.loglog(levels,v,'o-',base=2,color=color,label=f't={t:g}')
        saved[str(t)]={'native_error_own_initial_denominator':native,'common_error_N16_denominator':common,'common_H_error':herror}
    for a in (ax[0,0],ax[0,1],ax[1,0]):a.set_xticks(levels,[str(N) for N in levels]);a.set_xlabel('material grid side N');a.legend(fontsize=8);a.grid(alpha=.25)
    ax[0,0].set_title('Native finite-cell PDE error');ax[0,0].set_ylabel('mass L2 / own initial fluctuation')
    ax[0,1].set_title('Same N16 material-region error');ax[0,1].set_ylabel('mass L2 / original N16 fluctuation')
    ax[1,0].set_title('Same material-region H error');ax[1,0].set_ylabel('|actual H - N16 PDE reference H|')
    rows=[v for v in r['adjacent_spatial_comparisons'] if v['time']==.05];xx=np.arange(2);w=.32
    ax[1,1].bar(xx-w/2,[v['gains']['common'] for v in rows],w,label='common error reduction')
    ax[1,1].bar(xx+w/2,[v['fixed_indicators']['common'] for v in rows],w,label='fixed positive indicator')
    ax[1,1].set_yscale('log');ax[1,1].set_xticks(xx,['16 to 32','32 to 64']);ax[1,1].set_ylabel('relative concentration error');ax[1,1].set_title('Final gains vs original positive indicators');ax[1,1].legend(fontsize=8);ax[1,1].grid(axis='y',alpha=.25)
    passed=r['registered_three_space_level_contact_oracle_control_passed'];fig.suptitle(f'Registered static-contact oracle space control: {"PASSED" if passed else "FAILED / UNRESOLVED"}; Stage 3 incomplete',fontsize=12)
    fig.tight_layout();fig.savefig(out/'contact-spatial.png',dpi=180);fig.savefig(out/'contact-spatial.svg');plt.close(fig)
    save(out/'plot-data.json',{'N_values':list(levels),'saved_errors':saved,'fits_clipping_or_extrapolation':False,'new_candidate_or_reference_solve':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
