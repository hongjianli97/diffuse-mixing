"""Presentation-only rendering of audited column profiles; no field evaluation."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_route_decision import read, save, digest, ROOT


def main(directory):
    out=Path(directory).resolve();source=out/'exact-error-structure.json'
    rows=read(source);assert len(rows)==6
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True,sharey=True,constrained_layout=True)
    for ax,kind in zip(axes,('kernel','ledger')):
        ax.axvspan(.25,.75,color='#dce4ef',alpha=.5,label='Compression region')
        ax.axhline(0,color='#505050',linewidth=.7)
        for row in rows:
            if row['source']==kind:
                ax.plot([float(F(c['a_exact'])) for c in row['columns']],
                    [float(F(c['relative_column_mean_exact'])) for c in row['columns']],
                    color={64:'#244b81',128:'#167b6b',256:'#c2691d'}[row['N']],
                    linewidth=1.5,label=f"N={row['N']}")
        ax.set_title(kind);ax.set_xlabel('Material coordinate a');ax.set_xlim(0,1)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower right')
    axes[0].set_ylabel('Signed relative density error\n(mass-weighted column mean)',fontsize=10)
    fig.suptitle('Saved snapshot only: h / material spacing = 2',fontsize=12)
    files=[]
    for extension in ('png','svg'):
        dest=out/f'signed-error-profiles-readable.{extension}';assert not dest.exists()
        fig.savefig(dest,dpi=160);files.append(dest)
    plt.close(fig)
    save(out/'figure-provenance.json',{'input':str(source.relative_to(ROOT)),'input_sha256':digest(source),
        'renderer_sha256':digest(Path(__file__)),
        'files':{str(f.relative_to(ROOT)):digest(f) for f in files},
        'original_plot_preserved':True,'reason':'shorten axis label to avoid clipping; same audited column data',
        'new_fields_or_numerical_runs':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);main(p.parse_args().directory)
