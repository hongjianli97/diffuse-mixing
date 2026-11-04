"""Saved contact profiles, H and action/PDE comparison; no numerical reruns."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import ROOT,read,save
from shear_union64_study import arrays
from voronoi_time_action import metrics


def main():
    out=ROOT/'runs/reconnect-contact-oracle';r=read(out/'results.json');d=arrays(out/'material-reference.npz');p=arrays(out/'k16/trajectory-states.npz')
    with (out/'observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r['observations'][0]));w.writeheader();w.writerows(r['observations'])
    x=(np.arange(32)+.5)/32;times=np.arange(17)/320;H=[metrics(d['mass'],c,d['c0'])['H'] for c in p['states']]
    fig,ax=plt.subplots(2,2,figsize=(10,6.7))
    ax[0,0].plot(x,d['c0'].reshape(32,32)[0],label='initial cell means',color='gray')
    ax[0,0].plot(x,p['states'][-1].reshape(32,32)[0],'o',ms=3,label='point FV, T=0.05')
    ax[0,0].plot(x,d['reference'][-1].reshape(32,32)[0],'-',lw=1,label='PDE finite-cell reference')
    ax[0,0].axvline(.5,color='black',ls=':',lw=1);ax[0,0].set_title('Aligned periodic contact: first material row');ax[0,0].set_ylabel('mass-averaged concentration');ax[0,0].set_xlabel('material x');ax[0,0].legend(fontsize=7)
    image=ax[0,1].imshow((p['states'][-1]-d['reference'][-1]).reshape(32,32),origin='lower',extent=(0,1,0,1),aspect='equal',cmap='RdBu_r')
    ax[0,1].axvline(.5,color='black',ls=':',lw=1);ax[0,1].set_title('Raw FV minus PDE cell means, T=0.05');fig.colorbar(image,ax=ax[0,1],shrink=.8)
    ax[1,0].plot(times,H,label='actual full-state H')
    ax[1,0].plot([v['time'] for v in r['observations']],[v['reference_H'] for v in r['observations']],'x',label='PDE H')
    ax[1,0].set_xlabel('physical time');ax[1,0].set_ylabel('H');ax[1,0].set_title('Physical mixing diagnostic');ax[1,0].legend(fontsize=8)
    obs=r['observations'][1:];tt=[v['time'] for v in obs]
    ax[1,1].semilogy(tt,[v['reference_error_relative'] for v in obs],'o-',label='FV vs PDE spatial error')
    ax[1,1].semilogy(tt,[v['fixed_action_indicator_relative'] for v in obs],'s--',label='fixed action indicator')
    ax[1,1].semilogy(tt,[v['actual_vs_Cartesian_relative'] for v in obs],'x-',label='FV vs Cartesian action difference')
    ax[1,1].set_xticks(tt);ax[1,1].set_xlabel('physical time');ax[1,1].set_ylabel('mass L2 / initial fluctuation');ax[1,1].set_title('Spatial error separated from action error');ax[1,1].legend(fontsize=7)
    for a in (ax[0,0],ax[1,0],ax[1,1]):a.grid(alpha=.25)
    fig.suptitle('N32 same-backend static contact oracle control; Stage 3 incomplete',fontsize=12);fig.tight_layout()
    fig.savefig(out/'contact-oracle.png',dpi=180);fig.savefig(out/'contact-oracle.svg');plt.close(fig)
    save(out/'plot-data.json',{'source':'saved states/reference/results only','state_modified':False,'H_all17':H,
      'new_candidate_or_reference_solve':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False})


if __name__=='__main__':main()
