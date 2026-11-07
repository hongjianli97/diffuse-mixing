"""Saved moving-Becker diagnostic plots; no parameter scan or rerun."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save
from shear_union64_study import arrays


def main():
    out=ROOT/'runs/reconnect-becker-point';r=read(out/'results.json');col=arrays(out/'source-column-inputs.npz');diag=read(out/'density-volume-diagnostics.json')
    rows=r['observations'];tc=r['time_control'];K=[v['K'] for v in rows];fig,ax=plt.subplots(2,2,figsize=(10.5,7))
    ax[0,0].loglog(K,[v['column_observation_error_relative'] for v in rows],'o-',base=2);ax[0,0].set_xticks(K,[str(k) for k in K]);ax[0,0].set_xlabel('ordered midpoint intervals K');ax[0,0].set_ylabel('mass L2 / original initial fluctuation');ax[0,0].set_title('Final error vs saved PDE reference; one space grid')
    x=(col['mass_edges'][:-1]+col['mass_edges'][1:])/2;p=arrays(out/'k16/trajectory-states.npz')['states'][-1].reshape(8,32)[0]
    ax[0,1].plot(x,col['c0'],label='original material means, initial',color='gray');ax[0,1].plot(x,col['reference'],label='saved PDE means, T=0.1');ax[0,1].plot(x,p,'o',ms=3,label='actual K16 first transverse row');ax[0,1].set_xlabel('fixed material mass coordinate');ax[0,1].set_ylabel('cell mass average');ax[0,1].set_title('Original unknowns retained');ax[0,1].legend(fontsize=7)
    ax[1,0].bar([0,1,2],[tc['H4_8'],tc['H8_16'],tc['fixed_H_decrease_indicator']],color=['tab:blue','tab:orange','tab:red']);ax[1,0].set_xticks([0,1,2],['H4/8 gap','H8/16 gap','fixed decrease indicator']);ax[1,0].set_yscale('log');ax[1,0].set_ylabel('absolute H');ax[1,0].set_title('H time decrease remains UNRESOLVED')
    times=[v['time'] for v in diag]
    for key,label in [('maximum_point_vs_true_material_average_density_relative_difference','point rho vs true material average'),('maximum_point_Voronoi_vs_true_material_volume_relative_difference','point Voronoi vs true material volume')]:ax[1,1].plot(times,[v[key] for v in diag],label=label)
    ax[1,1].set_xlabel('midpoint physical time');ax[1,1].set_ylabel('maximum relative diagnostic difference');ax[1,1].set_title('Distinct approximations; no causal error allocation');ax[1,1].legend(fontsize=7)
    for a in ax.ravel():a.grid(alpha=.25,axis='y')
    fig.suptitle('Moving Becker point-FV oracle: state time control resolved, H unresolved; Stage 3 incomplete',fontsize=11);fig.tight_layout()
    fig.savefig(out/'becker-point.png',dpi=180);fig.savefig(out/'becker-point.svg');plt.close(fig)
    for name,values in [('observations.csv',rows),('density-volume-diagnostics.csv',diag)]:
        with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    save(out/'plot-data.json',{'source':'saved outputs only','state_modified_or_fitted':False,'new_candidate_or_reference_solve':False,'time_control':tc,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
