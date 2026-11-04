"""Pure prescribed compression points and retained local controls of saved kernels."""
import math
import numpy as np
from compression_probe import map_x
from voronoi_time_action import metrics


def sample_points(labels,ids,rho0,times):
    a=np.asarray(labels,dtype=float);ident=np.asarray(ids);r=np.asarray(rho0,dtype=float);t=np.asarray(times,dtype=float)
    n=len(ident)
    if ident.ndim!=1 or ident.dtype.kind not in 'iu' or n<2 or len(np.unique(ident))!=n:
        raise ValueError('complete unique integer identities required')
    if a.shape!=(n,2) or r.shape!=(n,) or not np.isfinite(a).all() or not np.isfinite(r).all() or np.any(r<=0):
        raise ValueError('complete original labels and positive given point density required')
    if np.any(a<0) or np.any(a>=1) or t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or t[0]<0 or t[-1]>.05:
        raise ValueError('original periodic labels and increasing times in[0,.05] required')
    alpha=.25*t/.05;cosine=np.cos(2*np.pi*a[:,0])
    positions=np.empty((len(t),n,2))
    positions[:,:,0]=np.mod(map_x(a[None,:,0],alpha[:,None],t[:,None]),1.)
    positions[:,:,1]=np.mod(a[None,:,1]-.1*t[:,None],1.)
    rho=r[None,:]/(1+alpha[:,None]*cosine[None,:])
    return {'ids':ident.copy(),'time':t.copy(),'positions':positions,'rho':rho}


def kernel_local_controls(saved,old,mass,c0,checkpoint):
    """Sum both original local action checks, without treating them as full sequences."""
    if checkpoint not in (0,1,2):raise ValueError('original three checkpoints required')
    m=np.asarray(mass,dtype=float);initial=np.asarray(c0,dtype=float)
    sigma=math.sqrt(metrics(m,initial,initial)['variance']);norm=lambda v:float(np.sqrt(m@(v*v)))
    absolute=Hsum=0.;details=[]
    for K in (4,8):
        states=saved[f'k{K}_states'];rows=old['cases'][str(K)]['steps']
        if states.shape!=(K+1,len(m)) or len(rows)!=K:raise ValueError('complete original paths and local control rows required')
        for field,key in [('direct_endpoints','direct_vs_sampled_relative_difference'),('two_half_endpoints','two_half_vs_sampled_relative_difference')]:
            controls=saved[f'k{K}_{field}']
            if controls.shape!=(K,len(m)) or not np.isfinite(controls).all():raise ValueError('complete finite original local endpoints required')
            for step in range(K*checkpoint//2):
                delta=norm(controls[step]-states[step+1]);reported=rows[step][key]
                if not math.isclose(delta/sigma,reported,rel_tol=64*np.finfo(float).eps,abs_tol=1e-30):
                    raise ArithmeticError('original local numerical control does not match saved arrays')
                hd=abs(metrics(m,controls[step],initial)['H']-metrics(m,states[step+1],initial)['H'])
                absolute+=delta;Hsum+=hd
                details.append({'K':K,'step':step,'form':field,'mass_norm_difference':delta,'local_H_difference':hd})
    return {'absolute_sum':absolute,'relative_sum':absolute/sigma,'local_H_absolute_sum':Hsum,'details':details,
      'original_local_checks_are_independent_complete_paths':False,'local_discrepancies_are_certified_total_error_bounds':False}
