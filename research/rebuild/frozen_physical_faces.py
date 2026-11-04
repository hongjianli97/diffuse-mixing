"""Frozen real-face reference from explicit logical child identities.

No fabricated history, coordinate rounding, geometric snapping, virtual node
or graph-to-face flux allocation. The connected cell incidence has a cycle
space, so a cell divergence alone does not identify a unique face flux.
"""
import numpy as np
from scipy.special import roots_legendre
from becker_polygon_benchmark import density_velocity
from material_polygon_history import signed_area


def frozen_faces(quads,columns,rows):
    p=np.asarray(quads);columns=np.asarray(columns);rows=np.asarray(rows);n=len(p)
    if p.shape!=(n,4,2) or columns.shape!=(n,) or rows.shape!=(n,) or not np.isfinite(p).all():raise ValueError('matching finite frozen quads and logical identities required')
    if np.any(columns<0) or np.any(rows<0) or np.any(columns!=np.floor(columns)) or np.any(rows!=np.floor(rows)):
        raise ValueError('nonnegative integer logical identities required')
    columns=columns.astype(int);rows=rows.astype(int);nx=int(columns.max())+1;ny=int(rows.max())+1
    if len(set(zip(columns.tolist(),rows.tolist())))!=n or n!=nx*ny:raise ValueError('complete unique logical rectangle required')
    cv=np.column_stack((rows*(nx+1)+columns,rows*(nx+1)+columns+1,(rows+1)*(nx+1)+columns+1,(rows+1)*(nx+1)+columns))
    positions=np.empty(((nx+1)*(ny+1),2));seen=np.zeros(len(positions),dtype=bool)
    for q,ids in zip(p,cv):
        for point,key in zip(q,ids):
            if seen[key] and not np.array_equal(positions[key],point):raise ValueError('one logical shared vertex has inconsistent coordinates; no averaging')
            positions[key]=point;seen[key]=True
    edge=np.roll(p,-1,axis=1)-p;turn=edge[...,0]*np.roll(edge,-1,axis=1)[...,1]-edge[...,1]*np.roll(edge,-1,axis=1)[...,0]
    if np.any(signed_area(p)<=0) or np.any(turn<=0):raise ValueError('strict positive CCW convex cells required')
    pairs=sorted({tuple(sorted((int(a),int(b)))) for ids in cv for a,b in zip(ids,np.roll(ids,-1))})
    ev=np.array(pairs);lookup={pair:i for i,pair in enumerate(pairs)};C=np.zeros((n,len(pairs)),dtype=int)
    for i,ids in enumerate(cv):
        for a,b in zip(ids,np.roll(ids,-1)):C[i,lookup[tuple(sorted((int(a),int(b))))]]=1 if a<b else -1
    counts=np.count_nonzero(C,axis=0);boundary=counts==1
    if np.any((counts!=1)&(counts!=2)) or np.any(C[:,~boundary].sum(axis=0)!=0):raise ValueError('invalid shared face orientation')
    reached={0};pending=[0]
    while pending:
        i=pending.pop();neighbours=np.flatnonzero(np.any(C[:,np.flatnonzero(C[i])]!=0,axis=1))
        for j in neighbours:
            if int(j) not in reached:reached.add(int(j));pending.append(int(j))
    if len(reached)!=n:raise ValueError('connected physical cell complex required')
    return {'positions':positions,'cell_vertices':cv,'edge_vertices':ev,'incidence':C,'boundary_mask':boundary,
        'interior_flux_cycle_dimension':int(np.count_nonzero(~boundary)-(n-1))}


def physical_scalar_face_flux(topology,probe,flow,D,order=64):
    if not np.isfinite(D) or D<0:raise ValueError('finite nonnegative D required')
    p=topology['positions'];a,b=topology['edge_vertices'].T;v=p[b]-p[a];z,w=roots_legendre(order);s=(z+1)/2
    points=p[a,None,:]+s[None,:,None]*v[:,None,:];_,gradient,*_=probe.world(points)
    rho,_=density_velocity(flow,points[None,...],np.array([probe.time]),probe.T)
    q=-D*np.sum(rho[0]*(gradient[...,0]*v[:,None,1]-gradient[...,1]*v[:,None,0])*(w/2),axis=1)
    return q
