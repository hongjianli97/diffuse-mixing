"""Raw translation objectivity plots and tables; no fitted states or thresholds."""
from fractions import Fraction as F
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-t2-translation';r=read(out/'results.json');p=r['coefficient_perturbations']
    rows=[{k:v for k,v in row.items() if k!='state_indicator_components_absolute'} for row in r['observations']]
    with (out/'observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'coefficient-perturbations.csv').open('x',newline='') as f:
        values=[dict(step=i,**row) for i,row in enumerate(p)];w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    fig,ax=plt.subplots(1,3,figsize=(12.8,3.7));active=r['observations'][1:];t=[v['time'] for v in active]
    ax[0].semilogy(t,[v['state_difference_relative'] for v in active],'o-',label='moving vs static')
    ax[0].semilogy(t,[v['state_indicator_relative'] for v in active],'s--',label='fixed total indicator')
    ax[0].set_title('Material concentration objectivity');ax[0].set_ylabel('mass norm / initial fluctuation')
    ax[1].semilogy(t,[v['H_difference'] for v in active],'o-',label='absolute H difference')
    ax[1].semilogy(t,[v['H_indicator'] for v in active],'s--',label='fixed H indicator');ax[1].set_title('Material mixing H objectivity')
    relative=[float(F(v['maximum_relative_static_face_difference_exact'])) for v in p]
    positive=[i for i,v in enumerate(relative) if v>0]
    ax[2].semilogy(positive,[relative[i] for i in positive],'o',label='raw nonzero relative face difference')
    ax[2].axhline(1e-10,linestyle='--',color='tab:orange',label='original algebra tolerance')
    ax[2].set_title('Raw coefficient perturbations');ax[2].set_xlabel('midpoint index')
    for a in ax[:2]:a.set_xticks(t);a.set_xlabel('time')
    for a in ax:a.grid(alpha=.25);a.legend(fontsize=7)
    fig.suptitle('T2: one rigid translation; Stage 3 incomplete',fontsize=11);fig.tight_layout()
    fig.savefig(out/'t2-translation.png',dpi=180);fig.savefig(out/'t2-translation.svg');plt.close(fig)
    save(out/'plot-data.json',{'source':'results.json','raw_relative_face_differences':relative,'exact_zero_graph_indices':[i for i,v in enumerate(relative) if v==0],
      'no_fits_or_extrapolated_states':True,'general_T2_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
