"""Reference-only P_rho L_h P_rho aggregate on known material squares.

This uses the original analytic density and material geometry as extra oracle
information. It neither modifies nor constructs a candidate exchange backend.
All tensor weights and signed aggregate contributions are retained unadjusted.
"""
from fractions import Fraction as F
import math
import numpy as np

from contact_continuous import gauss
from oracle_path_reference import path_gamma
from positive_kernel_2d import slope_over_radius


def cell_support_offsets(N,h):
    if isinstance(N,bool) or not isinstance(N,(int,np.integer)) or N<4 or N&(N-1):
        raise ValueError('dyadic N>=4 required for this known-square reference')
    if not np.isscalar(h) or not np.isfinite(h) or not 0<h<.125:
        raise ValueError('0<h<1/8 required for separated periodic contacts')
    cutoff=(2*F(float(h))*N)**2;limit=cutoff.numerator//cutoff.denominator
    reach=math.isqrt(limit)+1
    if reach>=N//2:raise ValueError('unique local periodic cell lifts required')
    return tuple((dx,dy) for dx in range(-reach,reach+1) for dy in range(-reach,reach+1)
        if (dx or dy) and max(abs(dx)-1,0)**2+max(abs(dy)-1,0)**2<=limit)


def tensor_cell_rule(N,order):
    if isinstance(order,bool) or not isinstance(order,(int,np.integer)) or order<2:
        raise ValueError('integer tensor Gauss order>=2 required')
    z,w=gauss(int(order))
    if np.any(w<=0):raise ArithmeticError('raw positive Gauss weights required; no repair')
    x,y=np.meshgrid(z,z,indexing='xy')
    local=np.column_stack([x.ravel(),y.ravel()])/(2*N)
    weights=(w[:,None]*w[None,:]).ravel()/(4*N*N)
    return {'nodes_1D':z.copy(),'weights_1D':w.copy(),'local_node_offsets':local,
            'weights_2D':weights,'order':int(order)}


def validate_material_observation(N,ids,mass,cbar,centers,D,h,target_ids):
    offsets=cell_support_offsets(N,h)
    if not np.isscalar(D) or not np.isfinite(D) or D<0:raise ValueError('finite D>=0 required')
    identities=np.asarray(ids);m=np.asarray(mass,dtype=float);u=np.asarray(cbar,dtype=float);x=np.asarray(centers,dtype=float)
    if identities.shape!=(N*N,) or identities.dtype.kind not in 'iu' or len(set(map(int,identities)))!=N*N:
        raise ValueError('unique integer original IDs required')
    if m.shape!=(N*N,) or u.shape!=(N*N,) or x.shape!=(N*N,2):raise ValueError('matching complete original observation required')
    if not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(u).all():
        raise ValueError('all original masses positive and all material means finite required')
    expected=np.array([((c+.5)/N,(r+.5)/N) for r in range(N) for c in range(N)])
    if not np.array_equal(x,expected):raise ValueError('known original aligned material-square centers required; no inference or repair')
    rho=np.where(x[:,0]<.5,1.,4.)
    if not np.array_equal(m,rho/N**2):raise ValueError('original mass inconsistent with this extra-information material reference')
    targets=tuple(target_ids)
    if not targets or len(set(targets))!=len(targets) or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in targets):
        raise ValueError('unique integer target IDs required')
    lookup={int(v):i for i,v in enumerate(identities)}
    if any(int(v) not in lookup for v in targets):raise ValueError('all requested material IDs must be present')
    return identities,m,u,x,tuple(lookup[int(v)] for v in targets),offsets


def integrate_ideal_material_rows(N,ids,mass,cbar,centers,D,h,target_ids,order,target_block=8,progress=None):
    """Four-dimensional tensor integration of exactly the original cbar field.

    Every touching cell is included, even if its center is outside point
    support. The target/neighbor cell measures use ordinary area; target rho
    cancels between mass observation and the continuous generator's 1/rho.
    """
    identities,m,u,x,targets,offsets=validate_material_observation(N,ids,mass,cbar,centers,D,h,target_ids)
    if isinstance(target_block,bool) or not isinstance(target_block,(int,np.integer)) or target_block<1:
        raise ValueError('positive integer computational block size required')
    rule=tensor_cell_rule(N,order);local=rule['local_node_offsets'];weights=rule['weights_2D']
    records=[]
    for target in targets:
        r,c=divmod(target,N)
        neighbor_indices=np.array([((r+dy)%N)*N+(c+dx)%N for dx,dy in offsets],dtype=np.int64)
        lifted=x[target]+np.array(offsets,dtype=float)/N
        neighbor=(lifted[:,None,:]+local[None,:,:]).reshape(-1,2)
        neighbor_weight=np.tile(weights,len(offsets))
        difference=np.repeat(u[neighbor_indices]-u[target],len(weights))
        points=x[target]+local
        point_inner=np.zeros(len(weights));cell_mass_rate=np.zeros(len(offsets));evaluations=0
        min_K=None;max_K=None;negative_raw_B=0
        if D!=0:
            for start in range(0,len(weights),target_block):
                stop=min(start+target_block,len(weights));target_points=points[start:stop]
                da=neighbor[None,:,0]-target_points[:,None,0]
                db=neighbor[None,:,1]-target_points[:,None,1]
                B=slope_over_radius(np.hypot(da,db),h)
                K=path_gamma(target_points[:,None,0],da,D)
                if np.any(K<=0) or not np.isfinite(K).all() or np.any(B<0):
                    raise ArithmeticError('invalid raw reference integrand; no repair')
                min_K=float(K.min()) if min_K is None else min(min_K,float(K.min()))
                max_K=float(K.max()) if max_K is None else max(max_K,float(K.max()))
                negative_raw_B+=int(np.count_nonzero(B<0))
                integrand=K*B*neighbor_weight[None,:]*difference[None,:]
                point_inner[start:stop]=np.sum(integrand,axis=1)
                by_cell=np.sum(integrand.reshape(stop-start,len(offsets),len(weights)),axis=2)
                cell_mass_rate+=2*np.sum(weights[start:stop,None]*by_cell,axis=0)
                evaluations+=(stop-start)*len(neighbor)
                if progress is not None:progress(int(identities[target]),int(order),stop,len(weights))
        numerator=2*float(np.dot(weights,point_inner))
        cell_sum=math.fsum(map(float,cell_mass_rate))
        rho=1. if x[target,0]<.5 else 4.
        records.append({'target_id':int(identities[target]),'target_index':int(target),'rate':numerator/m[target],
            'raw_mass_rate_numerator':numerator,'mass_rate_from_cell_sum_control':cell_sum,
            'mass_rate_reduction_difference':numerator-cell_sum,
            'original_mass':float(m[target]),'original_cbar':float(u[target]),
            'neighbor_indices':neighbor_indices,'neighbor_ids':identities[neighbor_indices].copy(),
            'neighbor_offsets':np.array(offsets,dtype=np.int64),'neighbor_lifted_centers':lifted,
            'target_nodes':points,'target_weights':weights.copy(),'target_node_inner_integrals':point_inner,
            'neighbor_signed_mass_rate_contributions':cell_mass_rate,
            'quadrature_area':float(math.fsum(map(float,weights))),
            'rho_times_quadrature_area_minus_original_mass':rho*math.fsum(map(float,weights))-float(m[target]),
            'kernel_node_pairs_evaluated':evaluations,'minimum_raw_path_gamma':min_K,'maximum_raw_path_gamma':max_K,
            'negative_raw_B_count':negative_raw_B,'extra_geometry_and_density_reference_only':True})
    return rule,records
