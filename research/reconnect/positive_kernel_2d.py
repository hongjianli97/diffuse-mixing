"""Periodic 2D positive exchange from point samples and oracle density.

Constant scalar h and D; minimum-image support shorter than half every period.
Weights m/(rho*unit thickness) are quadrature areas, not asserted material cells.
No material faces, F/C, fitted moments, clipping or row normalization.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags
from scipy.sparse.csgraph import connected_components

C2=9/(4*np.pi)


def kernel_value(radius,h):
    r=np.asarray(radius,dtype=float)
    if not np.isscalar(h) or not np.isfinite(h) or h<=0 or not np.isfinite(r).all() or np.any(r<0):raise ValueError('finite nonnegative radius and positive scalar h required')
    q=r/h;out=np.zeros_like(q);inside=q<2;z=q[inside]
    out[inside]=C2/h**2*(1-z/2)**6*(1+3*z+35*z*z/12)
    return out


def slope_over_radius(radius,h):
    r=np.asarray(radius,dtype=float)
    if not np.isscalar(h) or not np.isfinite(h) or h<=0 or not np.isfinite(r).all() or np.any(r<0):raise ValueError('finite nonnegative radius and positive scalar h required')
    q=r/h;out=np.zeros_like(q);inside=q<2;z=q[inside]
    out[inside]=(7*C2/3)/h**4*(2+5*z)*(1-z/2)**5
    return out


def build_periodic_exchange(data,frame,D,h,periods,thickness=1.):
    """One snapshot from validated TrajectorySamples; no scalar probe input."""
    box=np.asarray(periods,dtype=float)
    if box.shape!=(2,) or not np.isfinite(box).all() or np.any(box<=0):raise ValueError('two finite positive periodic lengths required')
    if not np.isscalar(D) or not np.isfinite(D) or D<0:raise ValueError('nonnegative finite spatially constant D required')
    if not np.isscalar(h) or not np.isfinite(h) or h<=0 or 2*h>=box.min()/2:raise ValueError('positive scalar h with 2h below half every period required')
    if thickness!=1.:raise ValueError('this research adapter fixes physical thickness to one metre')
    if not isinstance(frame,(int,np.integer)) or not 0<=frame<len(data.time):raise ValueError('valid frozen sample index required')
    if data.positions.shape[2]!=2 or data.rho is None:raise ValueError('2D point samples and declared positive oracle rho required')
    x=data.positions[frame];rho=data.rho[frame];m=data.mass
    if np.any(x<0) or np.any(x>=box):raise ValueError('wrapped input coordinates in [0,period) required; no implicit repair')
    pairs=cKDTree(x,boxsize=box).query_pairs(2*h,output_type='ndarray')
    if len(pairs):pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
    i,j=pairs.T;dx=x[j]-x[i];dx-=box*np.rint(dx/box);r=np.linalg.norm(dx,axis=1)
    area=m/(rho*thickness)
    if not np.isfinite(area).all() or np.any(area<=0):raise ValueError('finite positive oracle quadrature area required')
    if D==0:g=np.zeros(len(pairs))
    else:
        gamma=rho*D;H=2*gamma[i]*gamma[j]/(gamma[i]+gamma[j])
        g=2*area[i]*area[j]*thickness*H*slope_over_radius(r,h)
    if not np.isfinite(g).all() or np.any(g<0):raise RuntimeError('invalid raw conductance; no repair')
    n=len(m);G=coo_matrix((np.r_[g,g],(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()
    L=(diags(np.asarray(G.sum(axis=1)).ravel())-G).tocsr()
    arrays={'pairs':pairs,'conductances':g,'minimum_image_displacement':dx,'distance':r,'quadrature_area':area,'point_rho':rho.copy()}
    diagnostic={'nodes':n,'geometric_support_pairs':len(pairs),'positive_exchange_pairs':int(np.count_nonzero(g)),
        'exchange_components':int(connected_components(G,directed=False,return_labels=False)),
        'kernel_normalization_2d':C2,'h_m':float(h),'D_m2_per_s':float(D),'unit_thickness_m':thickness,
        'minimum_raw_conductance_kg_per_s':float(g.min()) if len(g) else None,
        'quadrature_area_interpretation':'m/(oracle point rho * unit thickness); not identified true cell area',
        'material_faces_or_F_C_used':False,'scalar_probe_used_to_build_edges':False,
        'clipping_fitting_or_normalization':False,'physical_acceptance':False,'fluid_orc_enabled':False}
    return L,arrays,diagnostic
