"""Plot saved budget shares and unchanged H decision; no new physical data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_route_decision import ROOT,read,save


def main():
    out=ROOT/'runs/reconnect-t3-affine-review';r=read(out/'results.json');rows=r['checkpoints'][1:]
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2));x=np.arange(2)
    spatial=[100*q['state_indicator_fractions']['spatial_symbol_error'] for q in rows]
    time=[100*q['state_indicator_fractions']['midpoint_time_error_symbol'] for q in rows]
    observed=[100*q['state_indicator_fractions']['observed8_16'] for q in rows]
    other=[100-s-t-o for s,t,o in zip(spatial,time,observed)]
    bottom=np.zeros(2)
    for values,label,color in [(spatial,'spatial symbol term','tab:blue'),(time,'midpoint time term','tab:orange'),(observed,'observed 8/16 difference','tab:green'),(other,'numerical / input / evaluation','tab:gray')]:
        ax[0].bar(x,values,bottom=bottom,label=label,color=color,width=.55);bottom+=values
    for i,v in enumerate(spatial):ax[0].text(i,45,f'{v:.2f}% spatial',ha='center',color='white',fontweight='bold')
    ax[0].set_xticks(x,['t=0.025','t=0.05']);ax[0].set_ylim(0,116);ax[0].set_ylabel('% of original absolute state indicator')
    ax[0].set_title('Positive state-budget attribution (not H shares)');ax[0].legend(fontsize=7,loc='upper center')
    width=.19
    series=[('source_actual_H_gap','actual K16 H gap','tab:blue'),('saved_continuous_grid_H_gap','saved continuous-grid H gap','tab:green'),
      ('spatial_only_original_formula_diagnostic_lower','spatial-only H diagnostic','tab:gray'),('source_fixed_H_indicator','unchanged full H indicator','tab:orange')]
    for offset,(key,label,color) in zip((-1.5,-.5,.5,1.5),series):
        values=[q[key]['float'] if isinstance(q[key],dict) else q[key] for q in rows]
        ax[1].bar(x+offset*width,values,width,label=label,color=color)
    ax[1].set_xticks(x,['t=0.025: unresolved','t=0.05: resolved']);ax[1].set_ylabel('H difference / original-formula diagnostic')
    ax[1].set_title('Even removing time terms leaves t=0.025 unresolved');ax[1].legend(fontsize=6.5,loc='upper left')
    for panel in ax:panel.grid(axis='y',alpha=.2);panel.set_axisbelow(True)
    fig.suptitle('Read-only N32 review: no new simulation; Stage 3 incomplete',fontsize=12);fig.tight_layout()
    fig.savefig(out/'t3-affine-budget.png',dpi=180);fig.savefig(out/'t3-affine-budget.svg');plt.close(fig)
    save(out/'plot-data.json',{'source':'results.json','checkpoints':rows,'derived_from_saved_observables_only':True,
      'state_shares_not_H_shares':True,'fits_or_modified_acceptance':False,'new_states_or_scientific_runs':0,'fluid_orc_enabled':False})


if __name__=='__main__':main()
