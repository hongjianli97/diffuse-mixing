"""Fixed-frame minimax envelope, signed defects and unclosed boundary data."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);result=json.loads((source/'results.json').read_text());proof=json.loads((source/'budget-certificates.json').read_text())
    z=np.load(source/'moment-budget.npz');fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    lower=proof['lower']['exact_lower_bound']['float'];upper=proof['upper']['exact_upper_bound']['float']
    epsilon=result['budget_solver']['epsilon_from_unmodified_conductances'];ax=axes[0,0]
    ax.bar([0,1],np.array([lower,upper])*100,color=['C3','C0'],width=.55)
    ax.set_xticks([0,1],['Exact separating lower bound','Certified feasible upper bound']);ax.set_ylabel('Worst component defect / (2D) [%]')
    ax.set_title('Local lower bound is attained by global exchange')
    ax.text(.08,.86,f'Both ~ {lower*100:.8f}%\nExact bracket width < 2.70e-13',transform=ax.transAxes,fontsize=10)
    ax.set_ylim(0,2.3);ax.grid(axis='y',alpha=.2)
    ax=axes[0,1];delta=z['second_component_defect_relative_to_2D'].reshape(-1,3)*100
    im=ax.imshow(delta,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-epsilon*100,vmax=epsilon*100,interpolation='nearest')
    ax.set_xticks([0,1,2],['xx','xy','yy']);ax.set_ylabel('Fixed interior-cell order (60 cells)');ax.set_title('Signed defects: no physical target was changed')
    fig.colorbar(im,ax=ax,label='Component defect / (2D) [%]',shrink=.9)
    ax=axes[1,0];D=json.loads((source/'config.json').read_text())['physical_parameters']['D'];eigen=z['second_defect_eigenvalues'][:,0]/(2*D)*100
    ax.plot(range(len(eigen)),eigen,'o',ms=3);ax.axhline(0,color='black',lw=.8)
    ax.set_xlabel('Fixed interior-cell order');ax.set_ylabel('Minimum defect eigenvalue / (2D) [%]')
    ax.set_title('Residual tensor need not add PSD diffusion');ax.grid(alpha=.2)
    ax=axes[1,1];A=csr_matrix((z['A_data'],z['A_indices'],z['A_indptr']),shape=tuple(z['A_shape']));free=np.diff(A.tocsc().indptr)==0;g=z['raw_conductances']
    counts=[np.count_nonzero((g>0)&~free),np.count_nonzero((g==0)&~free),np.count_nonzero(free)]
    ax.bar(range(3),counts,color=['C0','.6','C1']);ax.set_xticks(range(3),['Moment-involved\npositive pairs','Moment-involved\nzero pairs','Unidentified\nboundary-only pairs'])
    for i,v in enumerate(counts):ax.annotate(str(v),(i,v),xytext=(0,5),textcoords='offset points',ha='center')
    ax.set_ylabel('Number of shared pair variables');ax.set_title('A budget witness is not a closed physical method');ax.set_ylim(0,max(counts)*1.15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('One fixed-grid error envelope: exact drift, explicit second defects, fluid ORC disabled',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
