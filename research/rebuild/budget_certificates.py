"""Exact stored-data lower bound and posterior feasible upper bound.

No physical weight is repaired. An approximate right inverse is certified by
exact rational arithmetic to prove existence of a nearby feasible vector.
Only a proof bound is returned, never corrected conductances for evolution.
"""
from fractions import Fraction
import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix


def F(x):return Fraction.from_float(float(x))


def record(v):return {'numerator':str(v.numerator),'denominator':str(v.denominator),'float':float(v)}


def separator_budget_lower(A,b,second_rows,second_scale,dual):
    A=csr_matrix(A).tocsc();b=np.asarray(b);y={int(k):Fraction(v) for k,v in dual.items()}
    scale={int(i):F(v) for i,v in zip(second_rows,second_scale)}
    if any(i<0 or i>=A.shape[0] for i in y):raise ValueError('dual row outside physical moment system')
    slack=[]
    for j in range(A.shape[1]):
        value=Fraction(0)
        for p in range(A.indptr[j],A.indptr[j+1]):
            i=int(A.indices[p])
            if i in y:value+=y[i]*F(A.data[p])
        slack.append(value)
    gap=sum((v*F(b[i]) for i,v in y.items()),Fraction(0))
    denominator=sum((abs(v)*scale[i] for i,v in y.items() if i in scale),Fraction(0))
    if min(slack)<0 or gap>=0 or denominator<=0:
        return {'verified':False,'reason':'input is not a positive budget-lower-bound separator'}
    return {'verified':True,'exact_lower_bound':record(-gap/denominator),'exact_separator_gap':record(gap),
        'exact_budget_denominator':record(denominator),'minimum_exact_column_slack':record(min(slack)),
        'columns_checked':A.shape[1],'scope':'all nonnegative shared vectors satisfying the stored exact drift equations'}


def posterior_budget_upper(problem,conductance):
    E=csr_matrix(problem['E']);e=np.asarray(problem['e']);S=csr_matrix(problem['S']);s=np.asarray(problem['s']);c=np.asarray(problem['scale'])
    g=np.asarray(conductance,dtype=float)
    if g.shape!=(E.shape[1],) or not np.isfinite(g).all() or np.any(g<0):
        return {'verified':False,'reason':'a finite unmodified nonnegative numerical witness is required'},{}
    raw_residual=[]
    for i in range(E.shape[0]):
        value=F(e[i])-sum((F(E.data[p])*F(g[E.indices[p]]) for p in range(E.indptr[i],E.indptr[i+1])),Fraction(0))
        raw_residual.append(value)
    if all(v==0 for v in raw_residual):
        bounds=[]
        for i in range(S.shape[0]):
            value=-F(s[i])+sum((F(S.data[p])*F(g[S.indices[p]]) for p in range(S.indptr[i],S.indptr[i+1])),Fraction(0))
            bounds.append(abs(value)/F(c[i]))
        return {'verified':True,'exact_upper_bound':record(max(bounds)),'drift_basis_size':0,
            'correction_bound_kg_per_s':record(Fraction(0)),'original_conductances_modified':False,
            'proof':'raw nonnegative vector already satisfies every stored drift equality exactly; direct rational second-moment bound'},{}
    nnz=np.diff(E.indptr);keep=np.flatnonzero(nnz!=0);removed=np.flatnonzero(nnz==0)
    if np.any(e[removed]!=0):return {'verified':False,'reason':'zero drift row has a nonzero target'},{}
    E=E[keep];e=e[keep];n=len(e);active=np.flatnonzero(g>0)
    if n==0 or len(active)<n:return {'verified':False,'reason':'insufficient positive columns for a full drift correction basis'},{}
    # This is a deterministic numerical proof-basis selection, not a graph or
    # physical parameter search. No selected weight is changed by this routine.
    weighted=E[:,active].toarray()*g[active][None,:]
    _,_,pivots=qr(weighted,mode='economic',pivoting=True)
    selected=active[pivots[:n]];B=E[:,selected].tocsc()
    try:R=np.linalg.inv(B.toarray())
    except np.linalg.LinAlgError:return {'verified':False,'reason':'proposed drift proof basis is singular'},{}
    if not np.isfinite(R).all():return {'verified':False,'reason':'nonfinite numerical right inverse'},{}
    rational_R=[[F(v) for v in row] for row in R]
    # Certify ||I-RB||_inf < 1 without any floating comparison tolerance.
    row_sums=[Fraction(0) for _ in range(n)]
    for j in range(n):
        entries=[(int(B.indices[p]),F(B.data[p])) for p in range(B.indptr[j],B.indptr[j+1])]
        for i in range(n):
            value=sum((rational_R[i][k]*v for k,v in entries),Fraction(0))
            row_sums[i]+=abs(Fraction(int(i==j))-value)
    q=max(row_sums)
    if q>=1:return {'verified':False,'reason':'right inverse contraction not certified','exact_inverse_residual_norm':record(q)},{}
    residual=[raw_residual[i] for i in keep]
    applied=[sum((v*r for v,r in zip(row,residual)),Fraction(0)) for row in rational_R]
    correction=max(map(abs,applied))/(1-q);minimum=min(F(g[j]) for j in selected)
    if correction>=minimum:
        return {'verified':False,'reason':'exact drift correction bound does not preserve basis positivity',
            'correction_bound_kg_per_s':record(correction),'minimum_basis_weight_kg_per_s':record(minimum)},{}
    chosen=set(selected.tolist());bounds=[]
    for i in range(S.shape[0]):
        value=-F(s[i]);sensitivity=Fraction(0)
        for p in range(S.indptr[i],S.indptr[i+1]):
            j=int(S.indices[p]);a=F(S.data[p]);value+=a*F(g[j])
            if j in chosen:sensitivity+=abs(a)
        bounds.append((abs(value)+sensitivity*correction)/F(c[i]))
    upper=max(bounds)
    report={'verified':True,'exact_upper_bound':record(upper),'drift_basis_size':n,
        'exact_inverse_residual_norm':record(q),'correction_bound_kg_per_s':record(correction),
        'minimum_basis_weight_kg_per_s':record(minimum),'original_conductances_modified':False,
        'proof':'||I-RB||inf=q<1 and ||delta||inf<=||R(e-Eg)||inf/(1-q); positivity and all second bounds follow',
        'scope':'existence of a nonnegative vector near the raw witness satisfying exact stored drift and the certified second-moment budget; no evolution'}
    return report,{'proof_basis_columns':selected,'proof_kept_drift_rows':keep,'proof_approximate_right_inverse':R}
