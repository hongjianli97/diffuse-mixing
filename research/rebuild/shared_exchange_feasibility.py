"""One unknown conductance per undirected pair; no post-hoc symmetrization.

Only bulk linear/quadratic mass-average compatibility is assembled. Symmetric
nonnegative graph conductances imply algebraic conservation, not physical
boundary or finite-volume face consistency. Infeasible systems yield no graph.
"""
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix


def fixed_logical_pairs(columns,rows,column_reach=6,row_reach=1):
    c=np.asarray(columns);r=np.asarray(rows)
    if c.ndim!=1 or c.shape!=r.shape or len(c)<2 or not np.isfinite(c).all() or not np.isfinite(r).all():
        raise ValueError('matching finite logical cell columns and rows required')
    if np.any(c!=np.floor(c)) or np.any(r!=np.floor(r)) or len(set(zip(c.tolist(),r.tolist())))!=len(c):
        raise ValueError('unique integer logical cells required')
    if column_reach<0 or row_reach<0 or int(column_reach)!=column_reach or int(row_reach)!=row_reach:
        raise ValueError('nonnegative integer reaches required')
    i,j=np.triu_indices(len(c),1);keep=(abs(c[i]-c[j])<=column_reach)&(abs(r[i]-r[j])<=row_reach)
    return np.column_stack((i[keep],j[keep]))


def pair_features(first,other,columns,rows,means,variances,hy=.25,shear=.6):
    i=np.asarray(first,dtype=int);j=np.asarray(other,dtype=int);columns=np.asarray(columns);rows=np.asarray(rows)
    dy=(rows[j]-rows[i])*hy;dx=means[columns[j]]-means[columns[i]]+shear*dy
    return np.stack((dx,dy,dx*dx+variances[columns[j]]-variances[columns[i]],dx*dy,dy*dy),axis=-1)


def assemble_shared_moments(pairs,columns,rows,means,variances,mass,interior_cells,targets,hy=.25,shear=.6):
    m=np.asarray(mass,dtype=float);pairs=np.asarray(pairs,dtype=int);inside=np.asarray(interior_cells,dtype=int);target=np.asarray(targets,dtype=float)
    n=len(m)
    if len(columns)!=n or len(rows)!=n or not np.isfinite(hy) or hy<=0 or not np.isfinite(shear):
        raise ValueError('matching logical geometry and finite positive transverse spacing required')
    if m.ndim!=1 or np.any(m<=0) or not np.isfinite(m).all() or pairs.ndim!=2 or pairs.shape[1]!=2:
        raise ValueError('positive masses and undirected pairs required')
    if np.any(pairs<0) or np.any(pairs>=n) or np.any(pairs[:,0]>=pairs[:,1]) or len(set(map(tuple,pairs.tolist())))!=len(pairs):
        raise ValueError('unique ordered undirected pairs required')
    if target.shape!=(len(inside),5) or not np.isfinite(target).all() or len(set(inside.tolist()))!=len(inside) or np.any(inside<0) or np.any(inside>=n):
        raise ValueError('unique interior cells with matching five physical targets required')
    lookup={int(cell):k for k,cell in enumerate(inside)};rr=[];cc=[];vv=[]
    for direction in (0,1):
        i=pairs[:,direction];j=pairs[:,1-direction];features=pair_features(i,j,columns,rows,means,variances,hy,shear)
        for e,cell in enumerate(i):
            if int(cell) not in lookup:continue
            row=lookup[int(cell)]*5
            for f,v in enumerate(features[e]):
                if v!=0:rr.append(row+f);cc.append(e);vv.append(v)
    A=coo_matrix((vv,(rr,cc)),shape=(len(inside)*5,len(pairs))).tocsr()
    b=(m[inside,None]*target).ravel()
    return A,b


def solve_shared_moments(A,b,length_unit=.5):
    A=csr_matrix(A);b=np.asarray(b,dtype=float)
    if b.ndim!=1 or len(b)==0 or A.shape[0]!=len(b) or len(b)%5 or not np.isfinite(A.data).all() or not np.isfinite(b).all() or length_unit<=0 or not np.isfinite(length_unit):
        raise ValueError('finite five-moment blocks and positive unit required')
    units=np.tile([length_unit,length_unit,length_unit**2,length_unit**2,length_unit**2],len(b)//5)
    result=linprog(np.zeros(A.shape[1]),A_eq=A.multiply((1/units)[:,None]),b_eq=b/units,bounds=(0,None),method='highs')
    report={'solver':'scipy.optimize.linprog(highs), zero objective and default tolerances','status':int(result.status),
        'message':result.message,'solver_reports_feasible':bool(result.success),'equations':A.shape[0],'shared_pair_variables':A.shape[1],
        'physical_acceptance':False,'fluid_orc_enabled':False,'physical_boundary_or_face_consistency_checked':False}
    g=None
    if result.success:
        g=result.x.copy();report['minimum_raw_conductance']=float(g.min());report['maximum_raw_equation_residual_after_unit_conversion']=float(np.max(abs((A@g-b)/units)))
    return report,g


def exact_sparse_separator(A,b,dual_entries):
    """Independently check rational y^T A >= 0 and y^T b < 0."""
    A=csr_matrix(A).tocsc();b=np.asarray(b);dual={int(k):Fraction(v) for k,v in dual_entries.items()}
    if any(k<0 or k>=A.shape[0] for k in dual):raise ValueError('dual row outside system')
    slack=[]
    for j in range(A.shape[1]):
        value=Fraction(0)
        for p in range(A.indptr[j],A.indptr[j+1]):
            row=int(A.indices[p])
            if row in dual:value+=dual[row]*Fraction.from_float(float(A.data[p]))
        slack.append(value)
    gap=sum((v*Fraction.from_float(float(b[k])) for k,v in dual.items()),Fraction(0))
    def record(v):return {'numerator':str(v.numerator),'denominator':str(v.denominator),'float':float(v)}
    return {'verified':bool(min(slack)>=0 and gap<0),'minimum_exact_pair_slack':record(min(slack)),
        'exact_target_dot_dual':record(gap),'sparse_dual':{str(k):record(v) for k,v in dual.items()},
        'columns_checked':A.shape[1],'scope':'exact stored-data infeasibility of this coupled moment system, not a continuum interval or universal convergence proof'}


def conservative_graph(pairs,conductance,mass):
    """Diagnostic construction only; requires an actual nonnegative witness."""
    if conductance is None:raise ValueError('infeasible system has no conductance vector; no zero-graph fallback')
    pairs=np.asarray(pairs,dtype=int);g=np.asarray(conductance,dtype=float);m=np.asarray(mass,dtype=float)
    if m.ndim!=1 or g.shape!=(len(pairs),) or not np.isfinite(g).all() or np.any(g<0) or not np.isfinite(m).all() or np.any(m<=0):
        raise ValueError('finite nonnegative shared conductances and positive masses required')
    if pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0) or np.any(pairs>=len(m)) or np.any(pairs[:,0]>=pairs[:,1]):
        raise ValueError('ordered pairs inside the fixed mass array required')
    L=np.zeros((len(m),len(m)))
    for (i,j),v in zip(pairs,g):L[i,i]+=v;L[j,j]+=v;L[i,j]-=v;L[j,i]-=v
    return L,-L/m[:,None]
