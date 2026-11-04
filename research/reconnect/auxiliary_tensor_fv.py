"""Positive auxiliary-face FV for a complete periodic tensor point set.

Geometry is inferred from unchanged points, not supplied true material cells.
The builder never receives masses, scalar states or PDE/reference responses.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np


@dataclass(frozen=True)
class AuxiliaryFaces:
    ids: tuple
    shape: tuple
    periods: tuple
    thickness: F
    D: F
    grid_indices: tuple
    x_gaps: tuple
    y_gaps: tuple
    x_widths: tuple
    y_widths: tuple
    edges: tuple
    axes: tuple
    distances: tuple
    lengths: tuple
    conductances: tuple


def _positive(value,label,allow_zero=False):
    if isinstance(value,(bool,np.bool_)) or not math.isfinite(float(value)):
        raise ValueError('finite '+label+' required')
    q=F(float(value))
    if q<0 or not allow_zero and q==0:raise ValueError('positive '+label+' required')
    return q


def build_auxiliary_faces(ids,positions,rho,D,periods=(1,1),thickness=1):
    ident=np.asarray(ids);x=np.asarray(positions,dtype=float);density=np.asarray(rho,dtype=float)
    if ident.ndim!=1 or ident.dtype.kind not in 'iu' or len(set(map(int,ident)))!=len(ident):
        raise ValueError('unique integer identities required')
    n=len(ident)
    if x.shape!=(n,2) or density.shape!=(n,) or not np.isfinite(x).all():
        raise ValueError('complete finite point positions and density required')
    if not np.isfinite(density).all() or np.any(density<=0):raise ValueError('all point densities must be positive')
    if len(periods)!=2:raise ValueError('two periodic lengths required')
    box=tuple(_positive(v,'periodic length') for v in periods)
    diffusion=_positive(D,'D',True);b=_positive(thickness,'thickness')
    if any(not 0<=F(float(v))<box[k] for row in x for k,v in enumerate(row)):
        raise ValueError('original positions must be in the stated periodic box; no wrapping repair')
    # Internal geometric ordering does not modify input rows or scalar labels.
    xs=sorted(set(map(float,x[:,0])));ys=sorted(set(map(float,x[:,1])))
    nx,ny=len(xs),len(ys)
    if min(nx,ny)<3 or nx*ny!=n:raise ValueError('complete tensor point set with at least three points per axis required')
    xi={v:k for k,v in enumerate(xs)};yi={v:k for k,v in enumerate(ys)};lookup={}
    for i,(a,c) in enumerate(x):
        key=(yi[float(c)],xi[float(a)])
        if key in lookup:raise ValueError('duplicate point or missing tensor point')
        lookup[key]=i
    if len(lookup)!=n:raise ValueError('incomplete point product')
    gaps=[];widths=[]
    for values,L in zip([xs,ys],box):
        q=tuple(map(F,values));h=tuple((q[(k+1)%len(q)]-q[k])%L for k in range(len(q)))
        if min(h)<=0 or max(h)>=L/2 or sum(h,F(0))!=L:
            raise ValueError('unique short periodic faces required by this limited adapter')
        gaps.append(h);widths.append(tuple((h[k-1]+h[k])/2 for k in range(len(h))))
    r=tuple(F(float(v)) for v in density);faces=[]
    for row in range(ny):
        for col in range(nx):
            i=lookup[(row,col)]
            for j,axis,d,length in [(lookup[(row,(col+1)%nx)],0,gaps[0][col],widths[1][row]),
                (lookup[((row+1)%ny,col)],1,gaps[1][row],widths[0][col])]:
                gi=r[i]*diffusion;gj=r[j]*diffusion
                g=F(0) if diffusion==0 else b*length*2*gi*gj/(d*(gi+gj))
                first,second=(i,j) if ident[i]<ident[j] else (j,i)
                faces.append((first,second,axis,d,length,g))
    faces.sort(key=lambda v:(int(ident[v[0]]),int(ident[v[1]])))
    if len({(a,c) for a,c,*_ in faces})!=2*n:raise ValueError('duplicate shared face')
    return AuxiliaryFaces(tuple(map(int,ident)),(ny,nx),box,b,diffusion,
        tuple(tuple(lookup[(row,col)] for col in range(nx)) for row in range(ny)),
        gaps[0],gaps[1],widths[0],widths[1],tuple((v[0],v[1]) for v in faces),
        tuple(v[2] for v in faces),tuple(v[3] for v in faces),tuple(v[4] for v in faces),tuple(v[5] for v in faces))


def _state(graph,mass,cbar):
    n=len(graph.ids);m=np.asarray(mass,dtype=float);u=np.asarray(cbar,dtype=float)
    if m.shape!=(n,) or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('unchanged positive masses required')
    if u.shape!=(n,) or not np.isfinite(u).all():raise ValueError('finite full material-mean state required')
    return m,u


def exact_action(graph,mass,cbar):
    m,u=_state(graph,mass,cbar);mm=tuple(F(float(v)) for v in m);uu=tuple(F(float(v)) for v in u)
    flux=[F(0)]*len(m)
    for (i,j),g in zip(graph.edges,graph.conductances):
        f=g*(uu[j]-uu[i]);flux[i]+=f;flux[j]-=f
    return tuple(v/a for v,a in zip(flux,mm))


def independent_float_action(graph,mass,cbar,rho):
    """Different directional assembly and direct half-resistance evaluation."""
    m,u=_state(graph,mass,cbar);r=np.asarray(rho,dtype=float)
    if r.shape!=m.shape or not np.isfinite(r).all() or np.any(r<=0):raise ValueError('positive density required')
    grid=np.array(graph.grid_indices,dtype=int);state=u[grid];density=r[grid];capacities=m[grid]
    dx=np.array(list(map(float,graph.x_gaps)))[None,:]
    dy=np.array(list(map(float,graph.y_gaps)))[:,None]
    wx=np.array(list(map(float,graph.x_widths)))[None,:]
    wy=np.array(list(map(float,graph.y_widths)))[:,None]
    if graph.D==0:return np.zeros_like(u)
    gamma=density*float(graph.D)
    gx=float(graph.thickness)*wy/(dx/(2*gamma)+dx/(2*np.roll(gamma,-1,axis=1)))
    gy=float(graph.thickness)*wx/(dy/(2*gamma)+dy/(2*np.roll(gamma,-1,axis=0)))
    fx=gx*(np.roll(state,-1,axis=1)-state);fy=gy*(np.roll(state,-1,axis=0)-state)
    rate=(fx-np.roll(fx,1,axis=1)+fy-np.roll(fy,1,axis=0))/capacities
    result=np.empty_like(u);result[grid]=rate
    if not np.isfinite(result).all():raise ArithmeticError('nonfinite floating control; no repair')
    return result


def coefficient_packet(graph):
    return {'ids':list(graph.ids),'shape':list(graph.shape),'periods_exact':list(map(str,graph.periods)),
        'D_exact':str(graph.D),'thickness_exact':str(graph.thickness),'edges':[list(v) for v in graph.edges],
        'edge_ids':[[graph.ids[i],graph.ids[j]] for i,j in graph.edges],'axes':list(graph.axes),
        'face_lengths_exact':list(map(str,graph.lengths)),'center_distances_exact':list(map(str,graph.distances)),
        'conductances_exact':list(map(str,graph.conductances)),
        'auxiliary_geometry_not_true_material_cells':True,'builder_received_mass_scalar_or_reference':False}
