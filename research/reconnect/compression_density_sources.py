"""Moving density adapters from saved positions/mass; no scalar or oracle fields."""
from fractions import Fraction as F
import math
import numpy as np
from scipy.spatial import cKDTree
from positive_kernel_2d import kernel_value
from t1_density_sources import identities,points,positive,signed_ring_area


def periodic_winding(ids,times,positions,expected_times):
    ident=identities(ids);t=np.asarray(times,dtype=float);x=np.asarray(positions,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or not np.array_equal(t,np.asarray(expected_times,dtype=float)):
        raise ValueError('complete declared time sequence required')
    if t[0]!=0 or np.any(np.diff(t)<=0) or np.max(np.diff(t))>=.5:
        raise ValueError('increasing times from zero with given speed bound below half period required')
    if x.shape!=(len(t),len(ident),2):raise ValueError('complete matching position frames required')
    for frame in x:points(ident,frame,[1,1])
    winding=np.zeros(x.shape,dtype=np.int64)
    for k in range(1,len(t)):
        for i in range(len(ident)):
            for axis in (0,1):
                d=F(float(x[k,i,axis]))-F(float(x[k-1,i,axis]))
                if abs(d)==F(1,2):raise ValueError('exact half-period ambiguity')
                jump=math.floor(d+F(1,2));short=d-jump
                if abs(short)>=F(1,2):raise ValueError('ambiguous periodic displacement')
                winding[k,i,axis]=int(winding[k-1,i,axis])-jump
    return winding


def anchor_rings(ids,labels,edges):
    ids,x,_=points(ids,labels,[1,1]);rings=[[] for _ in ids];seen=set()
    for edge in edges:
        i,j=map(int,edge)
        if not 0<=i<j<len(ids) or (i,j) in seen:raise ValueError('unique saved shared edges required')
        seen.add((i,j));shift=[]
        for axis in (0,1):
            d=F(float(x[j,axis]))-F(float(x[i,axis]))
            if abs(d)==F(1,2):raise ValueError('ambiguous initial neighbor image')
            shift.append(-math.floor(d+F(1,2)))
        rings[i].append([j,*shift]);rings[j].append([i,*[-a for a in shift]])
    for i,ring in enumerate(rings):
        if len(ring)!=4 or len({v[0] for v in ring})!=4:raise ValueError('four unique saved neighbors required')
        ring.sort(key=lambda v:math.atan2(float(F(float(x[v[0],1]))+v[2]-F(float(x[i,1]))),float(F(float(x[v[0],0]))+v[1]-F(float(x[i,0])))))
    areas=ring_areas(x,np.zeros(x.shape,dtype=np.int64),rings)
    return rings,areas


def ring_areas(wrapped,winding,rings):
    x=np.asarray(wrapped,dtype=float);w=np.asarray(winding)
    if x.shape!=(len(rings),2) or w.shape!=x.shape or w.dtype.kind not in 'iu' or not np.isfinite(x).all() or np.any(x<0) or np.any(x>=1):
        raise ValueError('unchanged wrapped points and integer winding required')
    lifted=[(F(float(p[0]))+int(q[0]),F(float(p[1]))+int(q[1])) for p,q in zip(x,w)]
    area=[]
    for i,ring in enumerate(rings):
        if len(ring)!=4 or len({v[0] for v in ring})!=4:raise ValueError('fixed four-neighbor ring required')
        vertices=[]
        for j,sx,sy in ring:
            if not all(isinstance(a,(int,np.integer)) for a in (j,sx,sy)) or not 0<=j<len(x) or j==i:raise ValueError('valid fixed ring identities/images required')
            vertices.append((lifted[j][0]+sx-lifted[i][0],lifted[j][1]+sy-lifted[i][1]))
        area.append(signed_ring_area(vertices))
    return area


def ledger_density(wrapped,winding,rho0,rings,anchor_areas):
    r0=positive(rho0,len(rings),'given initial point density');areas=ring_areas(wrapped,winding,rings)
    if len(anchor_areas)!=len(rings) or any(F(a)<=0 for a in anchor_areas):raise ValueError('positive fixed initial areas required')
    rho=np.array([float(F(float(r))*F(a0)/a) for r,a0,a in zip(r0,anchor_areas,areas)])
    return positive(rho,len(rings),'raw ledger density').copy(),areas


def kernel_density(ids,wrapped,mass_ids,mass,h):
    ids,x,box=points(ids,wrapped,[1,1]);m=positive(mass,len(ids),'original material mass')
    if not np.array_equal(ids,identities(mass_ids)):raise ValueError('mass identity/order mismatch')
    if not np.isscalar(h) or not np.isfinite(h) or h<=0 or 2*h>=.5:raise ValueError('fixed positive h with short support required')
    tree=cKDTree(x,boxsize=box);neighbors=tree.query_ball_point(x,2*h,eps=0,workers=1)
    rho=np.empty(len(ids));counts=[]
    for i,js in enumerate(neighbors):
        js=sorted(js,key=lambda j:int(ids[j]));delta=x[js]-x[i];delta-=np.rint(delta)
        r=np.sqrt(np.sum(delta*delta,axis=1));inside=r<2*h
        if i not in js:raise ArithmeticError('kernel self term missing')
        rho[i]=sum(float(v) for v in m[js]*kernel_value(r,h));counts.append(int(np.count_nonzero(inside)))
    return positive(rho,len(ids),'raw kernel density').copy(),{'support_count_min':min(counts),'support_count_max':max(counts),'self_retained':True,'h':float(h),'normalization':False}
