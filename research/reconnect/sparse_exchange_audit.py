"""Shared-edge graph assembly and raw generator audit without dense matrices."""
import numpy as np
from scipy.sparse import coo_matrix,issparse,diags
from trajectory_chain import ALGEBRA_TOL,edge_array


def sparse_laplacian(n,pairs,conductances):
    p=edge_array(n,pairs);g=np.asarray(conductances,dtype=float)
    if g.shape!=(len(p),) or not np.isfinite(g).all() or np.any(g<0):raise ValueError('one finite raw nonnegative conductance per unique shared edge required')
    i,j=p.T
    return coo_matrix((np.r_[g,g,-g,-g],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(n,n)).tocsr()


def sparse_max_abs(matrix):
    return float(np.max(abs(matrix.data))) if matrix.nnz else 0.


def audit_laplacian(mass,L):
    m=np.asarray(mass,dtype=float)
    if not issparse(L) or m.ndim!=1 or len(m)<2 or np.any(m<=0) or not np.isfinite(m).all() or L.shape!=(len(m),len(m)):
        raise ValueError('finite positive masses and matching sparse Laplacian required')
    A=L.tocsr(copy=True)
    if not A.has_canonical_format or not np.isfinite(A.data).all():raise ValueError('canonical finite raw sparse matrix required')
    Q=(-diags(1/m)@A).tocsr()
    if not np.isfinite(Q.data).all():raise ValueError('generator arithmetic not finite')
    weighted=Q.multiply(m[:,None]).tocsr();entries=Q.tocoo();mask=entries.row!=entries.col
    values=entries.data[mask];missing=mask.sum()<len(m)*(len(m)-1)
    minimum=float(values.min()) if len(values) else 0.
    if missing:minimum=min(0.,minimum)
    row=float(np.max(abs(np.asarray(Q.sum(axis=1)).ravel())))
    conservation=float(np.max(abs(np.asarray(m@Q).ravel())))
    symmetry=sparse_max_abs(weighted-weighted.T);scale=max(1.,sparse_max_abs(Q));wscale=max(1.,sparse_max_abs(weighted))
    return {'constant_residual_per_s':row,'mass_residual_kg_per_s':conservation,'weighted_symmetry_residual_kg_per_s':symmetry,
        'minimum_raw_offdiagonal_per_s':minimum,
        'graph_algebra_eligible':bool(row<=ALGEBRA_TOL*scale and conservation<=ALGEBRA_TOL*wscale and symmetry<=ALGEBRA_TOL*wscale and minimum>=-ALGEBRA_TOL*scale),
        'linear_algebra_tolerance':ALGEBRA_TOL,'physical_acceptance_inferred':False,'clipping_or_renormalization':False,
        'sparse_storage_only':True,'nodes':len(m),'stored_L_entries':A.nnz}


def restrict_equal_nested_squares(values,fine_mass,coarse_mass):
    """Diagnostic mass projection only; exact 2x2 logical nesting required.

    Uses declared parent masses as denominators. Refuses inconsistent child
    masses instead of renormalizing them. No geometry is sent to the backend.
    """
    v=np.asarray(values,dtype=float);mf=np.asarray(fine_mass,dtype=float);mc=np.asarray(coarse_mass,dtype=float)
    n=int(np.sqrt(len(mc)));nf=2*n
    if n*n!=len(mc) or v.shape!=(nf*nf,) or mf.shape!=v.shape or np.any(mf<=0) or np.any(mc<=0) or not all(np.isfinite(a).all() for a in [v,mf,mc]):
        raise ValueError('finite fixed 2x2 nested-square data and positive masses required')
    child_mass=mf.reshape(n,2,n,2).sum(axis=(1,3)).ravel()
    if not np.array_equal(child_mass,mc):raise ValueError('child masses differ from saved parent masses; no repair')
    return (mf*v).reshape(n,2,n,2).sum(axis=(1,3)).ravel()/mc
