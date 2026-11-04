"""Frozen material half-cells for the declared transverse-uniform benchmark.

This is an explicit physical partition with inherited mass, not a repair of
the original cell volumes or a generic density-free equal-mass subdivision.
The parent cells, state and geometry are never mutated.
"""
import numpy as np
from scipy.special import roots_legendre
from material_polygon_history import signed_area
from becker_polygon_benchmark import density_velocity


def split_transverse_parallelograms(quads,mass,c0,centroids,stream_variance,
                                   *,transverse_mass_uniform=False,transverse_initial_state_uniform=False,parent_ids=None):
    p=np.asarray(quads,dtype=float);m=np.asarray(mass,dtype=float);c=np.asarray(c0,dtype=float)
    b=np.asarray(centroids,dtype=float);v=np.asarray(stream_variance,dtype=float);n=len(m)
    if transverse_mass_uniform is not True:
        raise ValueError('equal half-mass requires an explicit transverse-uniform material density assumption')
    if transverse_initial_state_uniform is not True:
        raise ValueError('parent c0 alone does not determine child averages; explicit transverse-uniform initial scalar required')
    if p.shape!=(n,4,2) or c.shape!=(n,) or b.shape!=(n,2) or v.shape!=(n,) or n==0:
        raise ValueError('matching quad, positive mass, state, centroid and stream variance arrays required')
    if not all(np.isfinite(a).all() for a in (p,m,c,b,v)) or np.any(m<=0) or np.any(v<0) or np.any(c<0) or np.any(c>1):
        raise ValueError('finite physical arrays, positive masses and admissible original c0 required')
    ids=np.array([f'p{i}' for i in range(n)]) if parent_ids is None else np.asarray(parent_ids)
    if ids.shape!=(n,) or len(set(ids.tolist()))!=n:raise ValueError('unique parent IDs required')
    e1=p[:,1]-p[:,0];w=p[:,3]-p[:,0];size=np.maximum(np.linalg.norm(e1,axis=1),np.linalg.norm(w,axis=1))
    closure=p[:,2]-p[:,1]-p[:,3]+p[:,0]
    if np.any(size<=0) or np.any(np.max(abs(closure),axis=1)>64*np.finfo(float).eps*size):
        raise ValueError('declared parallelogram closure fails machine-roundoff check; no geometry projection')
    edge=np.roll(p,-1,axis=1)-p;nextedge=np.roll(edge,-1,axis=1)
    turns=edge[...,0]*nextedge[...,1]-edge[...,1]*nextedge[...,0]
    if np.any(turns<=0) or np.any(signed_area(p)<=0):raise ValueError('strict positive convex CCW parallelograms required')
    midleft=(p[:,0]+p[:,3])/2;midright=(p[:,1]+p[:,2])/2
    lower=np.stack((p[:,0],p[:,1],midright,midleft),axis=1)
    upper=np.stack((midleft,midright,p[:,2],p[:,3]),axis=1)
    child_quads=np.stack((lower,upper),axis=1).reshape(-1,4,2)
    offsets=np.stack((-w/4,w/4),axis=1);child_centroids=(b[:,None,:]+offsets).reshape(-1,2)
    ce=np.roll(child_quads,-1,axis=1)-child_quads;relative=child_centroids[:,None,:]-child_quads
    if np.any(ce[...,0]*relative[...,1]-ce[...,1]*relative[...,0]<=0):raise ValueError('mass centroid outside open child; no centroid repair')
    stream=e1/np.linalg.norm(e1,axis=1)[:,None]
    stream_part=v[:,None,None]*stream[:,:,None]*stream[:,None,:]
    transverse=w[:,:,None]*w[:,None,:]
    parent_covariance=stream_part+transverse/12
    child_covariance=np.repeat(stream_part+transverse/48,2,axis=0)
    return {'quads':child_quads,'mass':np.repeat(m/2,2),'c0':np.repeat(c,2),'centroids':child_centroids,
        'covariance':child_covariance,'parent_covariance':parent_covariance,
        'parent_index':np.repeat(np.arange(n),2),'half_slot':np.tile([0,1],n),
        'child_ids':np.array([f'{i}:eta-half-{s}' for i in ids for s in (0,1)]),
        'centroid_offsets':offsets.reshape(-1,2),'stream_unit_vector':stream,
        'parent_parallelogram_closure':closure}


def logical_child_layout(nx=32,parent_rows=2):
    if nx<1 or parent_rows<1 or int(nx)!=nx or int(parent_rows)!=parent_rows:raise ValueError('positive logical dimensions required')
    parent=np.repeat(np.arange(nx*parent_rows),2);slot=np.tile([0,1],nx*parent_rows)
    column=parent%nx;row=2*(parent//nx)+slot
    boundary=(column==0)|(column==nx-1)|(row==0)|(row==2*parent_rows-1)
    return column,row,boundary


def neighbour_child_indices(column_row_labels,selected_child,nx=32,parent_rows=2):
    column,row,boundary=logical_child_layout(nx,parent_rows);labels=np.asarray(column_row_labels,dtype=int)
    if not 0<=selected_child<len(row) or labels.ndim!=2 or labels.shape[1]!=2:raise ValueError('valid selected child and neighbour column/relative-row labels required')
    absolute_row=row[selected_child]+labels[:,1];col=labels[:,0]
    if np.any(col<0) or np.any(col>=nx) or np.any(absolute_row<0) or np.any(absolute_row>=2*parent_rows):
        raise ValueError('neighbour outside the declared refined physical partition; no virtual fallback')
    indices=2*((absolute_row//2)*nx+col)+absolute_row%2
    if np.any(indices==selected_child) or len(set(indices.tolist()))!=len(indices):raise ValueError('self/duplicate neighbours rejected')
    return indices


def polynomial_averages(centroids,covariance,origin,axes):
    """Linear and centered quadratic true mass averages in orthonormal axes."""
    d=(np.asarray(centroids)-origin)@axes;S=np.einsum('ai,cab,bj->cij',axes,np.asarray(covariance),axes)
    H=d[:,:,None]*d[:,None,:]+S
    return np.stack((d[:,0],d[:,1],H[:,0,0],H[:,0,1],H[:,1,1]),axis=1)


def direct_mass_polynomials(flow,quads,fixed_mass,t,origin,axes,T=.1,order=32):
    """Independent physical density quadrature, dividing by fixed physical m."""
    p=np.asarray(quads);z,w=roots_legendre(order);u=(z+1)/2;weights=(w[:,None]*w[None,:]/4)
    q=p[:,0,None,None,:]+u[None,:,None,None]*(p[:,1]-p[:,0])[:,None,None,:]+u[None,None,:,None]*(p[:,3]-p[:,0])[:,None,None,:]
    rho,_=density_velocity(flow,q[None,...],np.array([t]),T);area=signed_area(p)
    measure=rho[0]*weights[None,:,:]*area[:,None,None]
    actual_mass=measure.sum(axis=(1,2));local=(q-origin)@axes
    fields=np.stack((local[...,0],local[...,1],local[...,0]**2,local[...,0]*local[...,1],local[...,1]**2),axis=-1)
    means=np.einsum('cab,cabf->cf',measure,fields)/np.asarray(fixed_mass)[:,None]
    return actual_mass,means
