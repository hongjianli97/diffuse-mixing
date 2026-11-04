"""Exact cached exchange for a fully validated dyadic Cartesian point array.

Integral-only reuse follows the sealed product-grid proof. Representative
ownership receipts are never transplanted to other edges. No c0 or reference
enters construction. The public entry is restricted to the original N128 gate.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np

from periodic_density_path import _fraction, _identity, _pair
from endpoint_density_path import localized_inverse_density_path
from reconstructed_path_rows import _Geometry
from positive_kernel_2d import slope_over_radius


def encode_path(p):
    return {'start_id':p.start_id,'end_id':p.end_id,'displacement':list(map(str,p.displacement)),
        'mean_inverse_density_exact':str(p.mean_inverse_density),
        'intervals':[{'start':str(v.start),'stop':str(v.stop),'source_id':v.source_id,'density':str(v.density)} for v in p.intervals],
        'breakpoint_owners':[[str(t),i] for t,i in p.breakpoint_owners]}


@dataclass(frozen=True)
class CachedExchange:
    N: int
    ids: tuple
    mass: tuple
    rho: tuple
    q: F
    D: F
    h: F
    offsets: tuple
    signatures: tuple
    path_map: tuple
    coefficient_map: tuple
    coefficient_keys: tuple
    radius_squared_numerators: tuple
    kernel_values: tuple
    path_means: tuple
    representative_records: tuple
    conductances: tuple


def _build_product_cache(positions,ids,mass,rho,D,h,periods,thickness=1,progress=None):
    """Private small-product helper for implementation checks, not flow scans."""
    box=_pair(periods,'periods');b=_fraction(thickness,'thickness')
    diffusion=_fraction(D,'D');width=_fraction(h,'h')
    if box!=(F(1),F(1)) or b!=1 or diffusion<0 or width<=0 or 4*width>=1:
        raise ValueError('unit torus/thickness, D>=0 and 0<h<1/4 required')
    points=tuple(_pair(p,'position') for p in positions)
    identities=tuple(_identity(v) for v in ids)
    masses=tuple(_fraction(v,'mass') for v in mass)
    density=tuple(_fraction(v,'density') for v in rho)
    N=math.isqrt(len(points))
    if N<4 or N*N!=len(points) or N&(N-1):
        raise ValueError('complete dyadic square point array required')
    if len(identities)!=N*N or len(masses)!=N*N or len(density)!=N*N or len(set(identities))!=N*N:
        raise ValueError('matching global inputs with unique integer IDs required')
    if any(v<=0 for v in masses+density):raise ValueError('positive global mass and point density required')
    expected=tuple((F(2*c+1,2*N),F(2*r+1,2*N)) for r in range(N) for c in range(N))
    if points!=expected:raise ValueError('exact complete wrapped Cartesian point array required; no repair')
    columns=density[:N]
    if any(v!=columns[i%N] for i,v in enumerate(density)):
        raise ValueError('all given densities must repeat exactly transversely')
    q=F(1,N*N)
    if any(m/r!=q for m,r in zip(masses,density)):
        raise ValueError('constant given m/rho pattern required; no mass or density adjustment')
    offsets=tuple((dx,dy) for dx in range(N//2) for dy in range(-N//2+1,N//2)
                  if (dx>0 or dx==0 and dy>0) and F(dx*dx+dy*dy,N*N)<=(2*width)**2)
    if not offsets:raise ValueError('nonempty geometric support required')
    signatures=tuple(sorted({(dx,tuple(columns[(c+j)%N] for j in range(dx+1)))
                             for c in range(N) for dx,dy in offsets}))
    sig_lookup={v:i for i,v in enumerate(signatures)}
    path_map=tuple(tuple(sig_lookup[(dx,tuple(columns[(c+j)%N] for j in range(dx+1)))]
                        for dx,dy in offsets) for c in range(N))
    radius_numbers=tuple(sorted({dx*dx+dy*dy for dx,dy in offsets}))
    radial_lookup={v:i for i,v in enumerate(radius_numbers)}
    keys=tuple(sorted({(path_map[c][k],radial_lookup[dx*dx+dy*dy])
                       for c in range(N) for k,(dx,dy) in enumerate(offsets)}))
    key_lookup={v:i for i,v in enumerate(keys)}
    coefficient_map=tuple(tuple(key_lookup[(path_map[c][k],radial_lookup[dx*dx+dy*dy])]
                               for k,(dx,dy) in enumerate(offsets)) for c in range(N))
    geometry=_Geometry(points,box)
    means=[];receipts=[]
    for sid,(dx,pattern) in enumerate(signatures):
        c=next(c for c in range(N) if tuple(columns[(c+j)%N] for j in range(dx+1))==pattern)
        j=(c+dx)%N if dx else N+c
        if diffusion==0:
            mean=None;record={'class_id':sid,'D_zero_path_bypassed':True}
        else:
            radius2=F(dx*dx if dx else 1,N*N)
            keep=geometry.inside_ball(c,radius2)
            localized=localized_inverse_density_path([points[k] for k in keep],[identities[k] for k in keep],
                [density[k] for k in keep],box,identities[c],identities[j])
            mean=localized.path.mean_inverse_density
            record={'class_id':sid,'normal_offset':dx,'given_density_sequence':list(map(str,pattern)),
                'representative_input_indices':[c,j],'retained_ids':list(localized.selection.retained_ids),
                'path':encode_path(localized.path),'ownership_applies_only_to_representative':True}
        means.append(mean);receipts.append(record)
        if progress is not None:
            progress('representative_paths',sid+1,len(signatures),record)
    B=tuple(float(slope_over_radius(math.sqrt(float(F(r2,N*N))),float(width))) for r2 in radius_numbers)
    if any(not math.isfinite(v) or v<0 for v in B):raise ArithmeticError('invalid raw B_h; no clipping')
    g=tuple(F(0) if diffusion==0 else 2*q*q*diffusion/means[sid]*F(B[rid]) for sid,rid in keys)
    if any(v<0 for v in g):raise ArithmeticError('invalid raw conductance; no clipping')
    return CachedExchange(N,identities,masses,density,q,diffusion,width,offsets,signatures,path_map,
        coefficient_map,keys,radius_numbers,B,tuple(means),tuple(receipts),g)


def build_original_global_cache(positions,ids,mass,rho,D,h,periods,thickness=1,progress=None):
    """Only the sealed original N128 geometry/density/D/h gate is admitted."""
    if len(positions)!=128**2 or _fraction(D,'D')!=F(.02) or _fraction(h,'h')!=F(.04):
        raise ValueError('this research entry fixes original N128, D=.02 and h=.04')
    given=tuple(_fraction(v,'density') for v in rho)
    pattern=(F(1),)*64+(F(4),)*64
    if len(given)!=128**2 or any(v!=pattern[i%128] for i,v in enumerate(given)):
        raise ValueError('original complete given density pattern required before any path evaluation')
    cache=_build_product_cache(positions,ids,mass,rho,D,h,periods,thickness,progress)
    if cache.rho[:128]!=(F(1),)*64+(F(4),)*64:
        raise ValueError('original given column density required')
    if (len(cache.path_means),len(cache.kernel_values),len(cache.conductances),len(cache.offsets))!=(132,45,964,166):
        raise ArithmeticError('fixed request classification changed')
    return cache


def coefficient_for_indices(cache,i,j):
    N=cache.N
    ri,ci=divmod(i,N);rj,cj=divmod(j,N)
    dx,dy=cj-ci,rj-ri
    if dx>N//2:dx-=N
    elif dx<-N//2:dx+=N
    if dy>N//2:dy-=N
    elif dy<-N//2:dy+=N
    if dx<0 or dx==0 and dy<0:
        i,j=j,i;dx,dy=-dx,-dy;ci=i%N
    k=cache.offsets.index((dx,dy))
    return cache.conductances[cache.coefficient_map[ci][k]]


@dataclass(frozen=True)
class ExactAction:
    rates: tuple
    flux_numerators: tuple
    row_L1: tuple
    Q: int
    S: int
    edge_count: int
    energy_pair_exact: F
    energy_row_exact: F


def apply_cached_exchange(cache,cbar_by_id,conductances=None,progress=None):
    """Stream every shared edge once using unchanged exact rational values."""
    if set(cbar_by_id)!=set(cache.ids):raise ValueError('all original IDs required for state application')
    state=tuple(_fraction(cbar_by_id[i],'cbar') for i in cache.ids)
    g=cache.conductances if conductances is None else tuple(conductances)
    if len(g)!=len(cache.conductances) or any(not isinstance(v,F) or v<0 for v in g):
        raise ValueError('matching raw nonnegative exact coefficient table required')
    Q=math.lcm(*(v.denominator for v in g));S=math.lcm(*(v.denominator for v in state))
    G=tuple(v.numerator*(Q//v.denominator) for v in g)
    C=tuple(v.numerator*(S//v.denominator) for v in state)
    N=cache.N;acc=[0]*(N*N);row_sum=[0]*(N*N);energy=0;edges=0
    for k,(dx,dy) in enumerate(cache.offsets):
        column_G=tuple(G[cache.coefficient_map[c][k]] for c in range(N))
        for r in range(N):
            base=r*N;neighbor_base=((r+dy)%N)*N
            for c in range(N):
                i=base+c;j=neighbor_base+(c+dx)%N;weight=column_G[c]
                delta=C[j]-C[i];flux=weight*delta
                acc[i]+=flux;acc[j]-=flux;row_sum[i]+=weight;row_sum[j]+=weight
                energy+=weight*delta*delta;edges+=1
        if progress is not None and ((k+1)%20==0 or k+1==len(cache.offsets)):
            progress('edge_action',k+1,len(cache.offsets))
    if sum(acc)!=0:raise ArithmeticError('shared integer flux inventory failed')
    row_energy=sum(v*n for v,n in zip(C,acc))
    if row_energy+energy!=0:raise ArithmeticError('exact shared energy identity failed')
    rates=tuple(F(v,Q*S)/m for v,m in zip(acc,cache.mass))
    row_L1=tuple(F(2*v,Q)/m for v,m in zip(row_sum,cache.mass))
    return ExactAction(rates,tuple(acc),row_L1,Q,S,edges,F(energy,Q*S*S),F(row_energy,Q*S*S))


def floating_action(cache,cbar):
    """Independent vectorized floating shared flux reduction, no repair."""
    N=cache.N;state=np.asarray(cbar,dtype=float).reshape(N,N)
    if state.shape!=(N,N) or not np.isfinite(state).all():raise ValueError('finite full original state required')
    flux_sum=np.zeros_like(state)
    for k,(dx,dy) in enumerate(cache.offsets):
        neighbor=np.roll(state,shift=(-dy,-dx),axis=(0,1))
        g=np.array([float(cache.conductances[cache.coefficient_map[c][k]]) for c in range(N)])
        flux=g[None,:]*(neighbor-state)
        flux_sum+=flux
        flux_sum-=np.roll(flux,shift=(dy,dx),axis=(0,1))
    return flux_sum.ravel()/np.array(list(map(float,cache.mass)))
