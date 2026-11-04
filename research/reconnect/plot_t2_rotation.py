"""Read-only continuous-rotation domain, actual-state and positive mixing plots."""
from fractions import Fraction as F
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save
from shear_union64_study import arrays
from voronoi_time_action import metrics


def main():
    out=ROOT/'runs/reconnect-t2-rotation';r=read(out/'results.json');world=read(out/'world-trajectory.json')
    d=arrays(out/'original-material-reference.npz');p=arrays(out/'k16/trajectory-states.npz');old=arrays(out/'static16-states.npz')
    for name,rows in [('observations.csv',r['observations']),('interval-covariance.csv',r['interval_covariance_checks'])]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(1,3,figsize=(12.8,3.8));domains={}
    for j,color in [(0,'tab:gray'),(16,'tab:blue'),(32,'tab:orange')]:
        frame=world['frames'][j];basis=np.array([[float(F(v)) for v in q] for q in frame['boundary']['basis_columns_exact']]);u,v=basis
        corners=np.array([[0.,0.],u,u+v,v,[0.,0.]])
        t=float(F(frame['time_exact']));ax[0].plot(corners[:,0],corners[:,1],color=color,label=f't={t:g}');domains[str(t)]=corners.tolist()
    ax[0].set_aspect('equal');ax[0].set_title('Domain and periodic lattice rotate');ax[0].set_xlabel('world x');ax[0].set_ylabel('world y');ax[0].legend(fontsize=8);ax[0].grid(alpha=.25)
    ax[1].scatter(old['states'][-1],p['states'][-1],s=6,alpha=.6,label='actual material means')
    x=[float(old['states'][-1].min()),float(old['states'][-1].max())];ax[1].plot(x,x,'k--',linewidth=1,label='static-path identity')
    ax[1].set_title('Final rotating vs static state');ax[1].set_xlabel('saved static final mean');ax[1].set_ylabel('rotating final mean');ax[1].legend(fontsize=7);ax[1].grid(alpha=.25)
    t=np.array([float(F(j,320)) for j in range(17)])
    h=[metrics(d['mass'],c,d['c0'])['H'] for c in p['states']];ho=[metrics(d['mass'],c,d['c0'])['H'] for c in old['states']]
    ref=[metrics(d['mass'],c,d['c0'])['H'] for c in d['reference']]
    ax[2].plot(t,h,'-',label='actual rotating H');ax[2].plot(t,ho,'--',label='saved static H');ax[2].scatter([0,.025,.05],ref,marker='x',color='black',label='saved analytic reference')
    ax[2].set_title('Positive diffusion: H stays nonzero');ax[2].set_xlabel('time');ax[2].set_ylabel('H');ax[2].legend(fontsize=7);ax[2].grid(alpha=.25)
    fig.suptitle('T2: exact-boundary continuous rotation; Stage 3 incomplete',fontsize=11);fig.tight_layout()
    fig.savefig(out/'t2-rotation.png',dpi=180);fig.savefig(out/'t2-rotation.svg');plt.close(fig)
    save(out/'plot-data.json',{'world_domain_corners':domains,'time':t.tolist(),'actual_rotating_H':h,'saved_static_H':ho,'saved_reference_H':ref,
      'fits_clipping_or_extrapolated_states':False,'world_input_points_or_lattice_modified_for_display':False,'fluid_orc_enabled':False,'stage3_physical_acceptance':'incomplete'})


if __name__=='__main__':main()
