"""Plot the saved normal-first budget; never alter the saved total."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'oracle-directional.npz') as z:
        x=z['material_label_centers'][:128,0];errors=z['fine_error_vectors'];total=z['fine_total_error']
    budgets=r['directional_runs']['fine']['budget'];g=budgets['all']
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    ax=axes[0,0];locations=np.arange(3)
    for offset,key,label in [(-.25,'normal_error_norm','Normal'),(0,'transverse_error_norm','Transverse'),(.25,'saved_total_error_norm','Saved total')]:
        bars=ax.bar(locations+offset,[100*budgets[k][key] for k in ['all','near_contact','far_from_contact']],width=.25,label=label)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=7)
    ax.set_xticks(locations,['All','Within 2h','Remaining']);ax.set_ylim(0,6.5)
    ax.set_ylabel('Error / full PDE rate norm [%]');ax.set_title('Directional norms do not add');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[0,1];ax.plot(x,errors[0,:128],label='Normal error');ax.plot(x,errors[1,:128],label='Transverse error');ax.plot(x,total[:128],'k--',label='Saved total error',lw=1.2)
    ax.axvline(.5,color='grey',ls=':',lw=.7);ax.axhline(0,color='grey',lw=.6)
    ax.set_xlabel('Material a [m]');ax.set_ylabel('Rate error [1/s]');ax.set_title('Opposite errors near contact, same row');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0];gram=np.asarray(g['gram_over_global_reference_norm_squared'])
    values=[gram[0,0],gram[1,1],2*gram[0,1],g['saved_total_error_norm']**2]
    bars=ax.bar(np.arange(4),values,color=['tab:blue','tab:orange','tab:red','grey'])
    ax.bar_label(bars,fmt='%.5f',padding=3,fontsize=8);ax.axhline(0,color='black',lw=.7)
    ax.set_xticks(np.arange(4),['Normal²','Transverse²','2 × cross','Saved total²']);ax.set_ylim(-.0046,.0040)
    ax.set_ylabel('Squared error / full PDE norm²');ax.set_title(f"Signed Gram budget: correlation {g['normal_transverse_correlation']:.3f}");ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];delta=r['component_quadrature_differences']
    values=[delta['rate_a']['all'],delta['rate_b']['all'],g['evaluation_residual_norm']]
    bars=ax.bar(np.arange(3),values,color=['tab:blue','tab:orange','grey']);ax.set_yscale('log')
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value*1.8,f'{value:.2e}',ha='center',fontsize=8)
    ax.set_xticks(np.arange(3),['Normal rule Δ','Transverse rule Δ','Sum residual']);ax.set_ylim(1e-17,1e-4)
    ax.set_ylabel('Norm / full PDE rate norm');ax.set_title('Numerical controls; differences are not bounds');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Unchanged oracle operator: normal-first residual budget',fontsize=12)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
