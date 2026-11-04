"""Saved harmonic versus extra-information oracle, common material cells."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);r=json.loads((source/'results.json').read_text())
    with np.load(source/'oracle-path.npz') as z:
        x=z['material_label_centers'][:128,0];ref=z['reference_rate'];harmonic=z['harmonic_fine_rate'];oracle=z['best_oracle_rate']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    ax=axes[0,0];locations=np.arange(3);keys=['all','near_contact','far_from_contact']
    a=ax.bar(locations-.18,[100*r['comparison'][k]['saved_harmonic_error'] for k in keys],width=.36,label='Saved harmonic')
    b=ax.bar(locations+.18,[100*r['comparison'][k]['oracle_path_error'] for k in keys],width=.36,label='Oracle path')
    ax.bar_label(a,fmt='%.3f',padding=3,fontsize=8);ax.bar_label(b,fmt='%.3f',padding=3,fontsize=8)
    ax.set_xticks(locations,['All','Within 2h','Remaining']);ax.set_ylim(0,17)
    ax.set_ylabel('Error / full PDE rate norm [%]');ax.set_title('Same-cell frozen-rate errors');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[0,1];ax.plot(x,ref[:128],label='PDE cell mean',lw=1.5);ax.plot(x,harmonic[:128],'--',label='Saved harmonic',lw=1.2);ax.plot(x,oracle[:128],label='Oracle path',lw=1.3)
    ax.set_xlabel('Material a [m]');ax.set_ylabel('Rate [1/s]');ax.set_title('Same original material row');ax.legend(fontsize=8);ax.grid(alpha=.2)
    errors=[harmonic-ref,oracle-ref];vmax=max(np.max(abs(e)) for e in errors)
    for ax,error,title in [(axes[1,0],errors[0],'Harmonic - PDE'),(axes[1,1],errors[1],'Oracle path - PDE')]:
        im=ax.imshow(error.reshape(128,128),origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=-vmax,vmax=vmax)
        ax.axvline(.5,color='k',ls=':',lw=.7);ax.set_xlabel('Material a [m]');ax.set_ylabel('Material b [m]');ax.set_title(title)
        fig.colorbar(im,ax=ax,label='Rate error [1/s]')
    fig.suptitle('Same h=0.04 contact: oracle full-density constitutive control',fontsize=11)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
