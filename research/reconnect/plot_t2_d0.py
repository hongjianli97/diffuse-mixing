"""Read-only plots of nonrigid density change and exact material identity."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save
from shear_union64_study import arrays


def main():
    out=ROOT/'runs/reconnect-t2-d0';r=read(out/'results.json');b=arrays(out/'builder-inputs.npz');d=arrays(out/'material-inputs.npz');p=arrays(out/'k16/trajectory-states.npz')
    for name,rows in [('observations.csv',r['observations']),('interval-zero-checks.csv',r['interval_zero_checks'])]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(1,3,figsize=(12.6,3.6))
    lo=b['rho'].min(axis=1);hi=b['rho'].max(axis=1)
    ax[0].plot(b['time'],lo,label='minimum given point rho');ax[0].plot(b['time'],hi,label='maximum given point rho')
    ax[0].set_title('Given nonrigid flow changes density');ax[0].set_xlabel('time');ax[0].set_ylabel('rho');ax[0].legend(fontsize=8);ax[0].grid(alpha=.25)
    ax[1].scatter(d['c0'],p['states'][-1],s=5,alpha=.6,label='actual final material means')
    x=[float(d['c0'].min()),float(d['c0'].max())];ax[1].plot(x,x,'k--',linewidth=1,label='identity reference')
    ax[1].set_title('D=0: original means preserved');ax[1].set_xlabel('original c0');ax[1].set_ylabel('actual final mean');ax[1].legend(fontsize=7);ax[1].grid(alpha=.25)
    checks=np.array([[v['actual_bitwise_initial'] for v in r['observations']],[v['independent_bitwise_initial'] for v in r['observations']]],dtype=int)
    ax[2].imshow(checks,aspect='auto',interpolation='nearest',cmap='Greens',vmin=0,vmax=1)
    ax[2].set_yticks([0,1],['actual','independent']);ax[2].set_xticks([0,4,8,12,16]);ax[2].set_xlabel('endpoint index (0...16)');ax[2].set_title('Every endpoint bitwise equals c0')
    ax[2].text(.5,-.35,'H = 0; raw inventory drift = 0',ha='center',transform=ax[2].transAxes,fontsize=9)
    fig.suptitle('T2: one D=0 nonrigid material path; Stage 3 incomplete',fontsize=11);fig.tight_layout()
    fig.savefig(out/'t2-d0.png',dpi=180);fig.savefig(out/'t2-d0.svg');plt.close(fig)
    save(out/'plot-data.json',{'given_density_minimum':lo.tolist(),'given_density_maximum':hi.tolist(),'endpoint_identity_matrix':checks.tolist(),
      'fits_clipping_or_extrapolated_states':False,'kappa_computed':False,'general_T2_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
