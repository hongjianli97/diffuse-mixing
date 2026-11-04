"""New 1D Brookshaw/Wendland-C4-shaped positive material exchange candidate.

Explicit 1D normalization, true material volumes, symmetric pair widths,
Neumann even-image closure. Not recovered historical 2D ltnmix code.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

C1=27/32


def kernel_value(r,h):
    r=np.asarray(r,dtype=float);h=np.asarray(h,dtype=float)
    if not np.isfinite(r).all():raise ValueError('finite kernel coordinates required')
    if np.any(h<=0) or not np.isfinite(h).all():raise ValueError('positive finite h required')
    q=np.abs(r)/h;v=np.zeros(np.broadcast_shapes(q.shape,h.shape));inside=q<2
    qb=np.broadcast_to(q,v.shape);hb=np.broadcast_to(h,v.shape)
    z=qb[inside]
    v[inside]=C1/hb[inside]*(1-z/2)**6*(1+3*z+35*z*z/12)
    return v


def slope_over_radius(r,h):
    """-W'_h(r)/r with its analytic finite r=0 value; zero off support."""
    r=np.asarray(r,dtype=float);h=np.asarray(h,dtype=float)
    if not np.isfinite(r).all():raise ValueError('finite kernel coordinates required')
    if np.any(h<=0) or not np.isfinite(h).all():raise ValueError('positive finite h required')
    q=np.abs(r)/h;value=np.zeros(np.broadcast_shapes(q.shape,h.shape));inside=q<2
    qb=np.broadcast_to(q,value.shape);hb=np.broadcast_to(h,value.shape);z=qb[inside]
    value[inside]=(63/32)/hb[inside]**3*(2+5*z)*(1-z/2)**5
    return value


def build_exchange(positions,mass,volumes,rho,D,h,bounds):
    x,m,V,rho,h=[np.asarray(a,dtype=float) for a in (positions,mass,volumes,rho,h)]
    if x.ndim!=1 or len(x)<2 or any(a.shape!=x.shape for a in (m,V,rho,h)):
        raise ValueError('compatible 1D node arrays required')
    if any(not np.isfinite(a).all() for a in (x,m,V,rho,h)) or np.any(m<=0) or np.any(V<=0) or np.any(rho<=0) or np.any(h<=0):
        raise ValueError('finite positions and positive mass/volume/rho/h required')
    if len(np.unique(x))!=len(x):raise ValueError('duplicate node positions')
    a,b=map(float,bounds)
    if not np.isscalar(D):raise ValueError('this candidate requires spatially constant scalar D')
    if not np.isfinite([a,b,D]).all() or b<=a or D<0 or np.any(x<=a) or np.any(x>=b):
        raise ValueError('valid interior nodes, bounds and nonnegative D required')
    if 2*h.max()>=b-a:raise ValueError('single-image closure requires support smaller than domain length')
    if not np.allclose(m,rho*V,rtol=1e-10,atol=0):raise ValueError('supplied oracle rho and actual volume disagree with fixed masses')
    hp=(h[:,None]+h[None,:])/2;dx=x[None,:]-x[:,None]
    Bdirect=slope_over_radius(dx,hp)
    Bleft=slope_over_radius(x[:,None]+x[None,:]-2*a,hp)
    Bright=slope_over_radius(2*b-x[:,None]-x[None,:],hp)
    gamma=rho*D
    if D==0:H=np.zeros((len(x),len(x)))
    else:H=2*gamma[:,None]*gamma[None,:]/(gamma[:,None]+gamma[None,:])
    G=2*V[:,None]*V[None,:]*H*(Bdirect+Bleft+Bright)
    if not np.isfinite(G).all() or np.min(G)<0:raise RuntimeError('nonfinite or negative kernel conductance')
    # Construct each unordered shared edge once, rather than keeping two
    # independently rounded evaluations. No negative-value clipping.
    G=np.triu(G,1);G=G+G.T
    L=np.diag(G.sum(axis=1))-G
    components=connected_components(csr_matrix(G>0),directed=False,return_labels=False)
    offdiag=~np.eye(len(x),dtype=bool)
    left_exchange=np.sum((2*V[:,None]*V[None,:]*H*Bleft)[offdiag])/2
    right_exchange=np.sum((2*V[:,None]*V[None,:]*H*Bright)[offdiag])/2
    direct_g=2*V[:,None]*V[None,:]*H*Bdirect
    # Same-material, no-boundary diagnostic only. No nodes/edges are removed.
    pure=((x-a)>2*h.max())&((b-x)>2*h.max())&(np.abs(x)>2*h.max())
    first=(direct_g*dx).sum(axis=1)/m
    second=(direct_g*dx*dx).sum(axis=1)/(2*m)
    diag={'nodes':len(x),'edges':int(np.count_nonzero(np.triu(G,1))), 'components':int(components),
          'min_g':float(G.min()),'max_g':float(G.max()),'symmetry_residual':float(np.max(abs(G-G.T))),
          'max_relative_m_vs_rho_volume':float(np.max(abs(m-rho*V)/m)),
          'reflected_exchange_sum_left':float(left_exchange),'reflected_exchange_sum_right':float(right_exchange),
          'moment_diagnostic_nodes':int(pure.sum()),
          'interior_first_moment_max_abs':float(np.max(abs(first[pure]))) if pure.any() else None,
          'interior_diffusivity_ratio_min':float(np.min(second[pure]/D)) if pure.any() and D>0 else None,
          'interior_diffusivity_ratio_max':float(np.max(second[pure]/D)) if pure.any() and D>0 else None,
          'cross_contact_edges':int(np.count_nonzero(G[np.ix_(x<0,x>0)])),
          'clipping_or_row_normalization':False}
    arrays={'G':G,'L':L,'positions':x,'mass':m,'volume':V,'rho':rho,'h':h,
            'moment_first':first,'moment_diffusivity':second,'moment_interior_mask':pure}
    return arrays,diag
