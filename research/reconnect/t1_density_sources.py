"""Fixed-input density adapters; no concentration, reference, fitting or repair."""
from fractions import Fraction as F
import copy
import math
import numpy as np
from positive_kernel_2d import kernel_value


def identities(ids):
    a = np.asarray(ids)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or len(a) < 2 or len(set(a.tolist())) != len(a):
        raise ValueError('unique integer identities required')
    return a


def positive(values, n, label):
    a = np.asarray(values, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('finite positive matching ' + label + ' required')
    return a


def points(ids, positions, periods, wrapped=True):
    ids = identities(ids); x = np.asarray(positions, dtype=float); box = np.asarray(periods, dtype=float)
    if box.shape != (2,) or not np.isfinite(box).all() or np.any(box <= 0):
        raise ValueError('two positive finite periods required')
    if x.shape != (len(ids), 2) or not np.isfinite(x).all():
        raise ValueError('matching finite positions required')
    if wrapped and (np.any(x < 0) or np.any(x >= box)):
        raise ValueError('already wrapped positions required; no implicit repair')
    if len(set(map(tuple, x.tolist()))) != len(ids):
        raise ValueError('duplicate positions')
    return ids, x, box


def kernel_density(ids, positions, mass_ids, mass, h, periods):
    ids, x, box = points(ids, positions, periods)
    if not np.array_equal(identities(mass_ids), ids):
        raise ValueError('mass identity/order mismatch')
    m = positive(mass, len(ids), 'material masses')
    if not np.isscalar(h) or not np.isfinite(h) or h <= 0 or 2*h >= box.min()/2:
        raise ValueError('positive fixed h with support below half every period required')
    order = np.argsort(ids); out = np.empty(len(ids))
    for i, p in enumerate(x):
        delta = x[order]-p
        delta -= box*np.rint(delta/box)
        radius = np.sqrt(np.sum(delta*delta, axis=1))
        # Increasing identity order, self retained. No density or kernel-sum divisor.
        terms = m[order]*kernel_value(radius, h)
        out[i] = sum(float(v) for v in terms)
    return positive(out, len(ids), 'raw kernel densities').copy()


def signed_ring_area(vertices):
    """Exact oriented polygon area; reject intersections rather than taking abs."""
    v = np.asarray(vertices, dtype=object)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3 or not all(math.isfinite(float(z)) for z in v.flat):
        raise ValueError('finite polygon vertices required')
    q = [tuple(z if isinstance(z,F) else F(float(z)) for z in p) for p in v]
    if len(set(q)) != len(q):
        raise ValueError('repeated ring vertex')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a, b, c):
        return cross(a,b,c)==0 and all(min(a[k],b[k])<=c[k]<=max(a[k],b[k]) for k in (0,1))
    n=len(q)
    for i in range(n):
        a,b=q[i],q[(i+1)%n]
        if cross(q[(i-1)%n],a,b)==0:
            raise ValueError('degenerate consecutive ring vertices')
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1): continue
            c,d=q[j],q[(j+1)%n]
            if (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b))):
                raise ValueError('self-intersecting ring')
    area=sum((a[0]*b[1]-a[1]*b[0] for a,b in zip(q,q[1:]+q[:1])),F(0))/2
    if area<=0: raise ValueError('nonpositive oriented ring area')
    return area


def density_from_ring(rho0, initial, current):
    if not np.isscalar(rho0) or not np.isfinite(rho0) or rho0<=0:
        raise ValueError('positive finite initial density required')
    a0,a=signed_ring_area(initial),signed_ring_area(current)
    rho=float(F(float(rho0))*a0/a)
    if not math.isfinite(rho) or rho<=0: raise ValueError('unrepresentable positive density')
    return rho,a0,a


def fixed_rings(ids, positions, packet):
    ids,x,box=points(ids,positions,list(map(float,map(F,packet['periods_exact']))))
    if ids.tolist()!=packet['ids']: raise ValueError('geometry identity/order mismatch')
    rings=[[] for _ in ids]
    for (i,j),shift in zip(packet['edges'],packet['image_shifts']):
        rings[i].append([j,*shift]);rings[j].append([i,*[-s for s in shift]])
    for i,ring in enumerate(rings):
        if len(ring)!=4 or len({v[0] for v in ring})!=4: raise ValueError('fixed four-neighbor T1 ring required')
        ring.sort(key=lambda v:math.atan2(*(x[v[0]]+np.array(v[1:])*box-x[i])[::-1]))
        signed_ring_area([x[j]+np.array([sx,sy])*box-x[i] for j,sx,sy in ring])
    return rings


def ledger_density(ids, initial, current, rho0, rings, periods):
    ids,x0,box=points(ids,initial,periods)
    _,x,_=points(ids,current,periods,wrapped=False)
    rho0=positive(rho0,len(ids),'given anchor densities')
    if len(rings)!=len(ids): raise ValueError('one fixed ring per identity required')
    rho=[];initial_areas=[];current_areas=[]
    for i,ring in enumerate(rings):
        if len(ring)!=4 or len({v[0] for v in ring})!=4: raise ValueError('unique fixed four-neighbor ring required')
        for j,sx,sy in ring:
            if not all(isinstance(v,(int,np.integer)) for v in (j,sx,sy)) or not 0<=j<len(ids) or j==i:
                raise ValueError('valid fixed neighbor/image identities required')
        initial=[tuple(F(float(x0[j,k]))+shift*F(float(box[k]))-F(float(x0[i,k])) for k,shift in enumerate((sx,sy))) for j,sx,sy in ring]
        current=[tuple(F(float(x[j,k]))+shift*F(float(box[k]))-F(float(x[i,k])) for k,shift in enumerate((sx,sy))) for j,sx,sy in ring]
        r,a0,a=density_from_ring(rho0[i],initial,current)
        rho.append(r);initial_areas.append(str(a0));current_areas.append(str(a))
    return np.array(rho),{'initial_areas_exact':initial_areas,'current_areas_exact':current_areas,'rings':rings}


def reweight(packet, ids, rho, mass):
    ids=identities(ids);r=positive(rho,len(ids),'source densities');positive(mass,len(ids),'unchanged material masses')
    if ids.tolist()!=packet['ids']: raise ValueError('geometry identity/order mismatch')
    D=F(packet['D_exact']);thickness=F(packet['thickness_exact'])
    if D<0 or thickness!=1: raise ValueError('nonnegative D and unit thickness required')
    if len(packet['edges'])!=len(packet['face_over_distance_exact']): raise ValueError('matching saved faces required')
    result=copy.deepcopy(packet);conductances=[]
    for (i,j),ratio in zip(packet['edges'],packet['face_over_distance_exact']):
        if not 0<=i<j<len(ids) or F(ratio)<=0: raise ValueError('positive saved face ratio and ordered edge required')
        a,b=F(float(r[i]))*D,F(float(r[j]))*D
        conductances.append(str(F(0) if D==0 else thickness*F(ratio)*2*a*b/(a+b)))
    result['conductances_exact']=conductances
    return result
