"""Given periodic area-preserving shear; derivatives/geometry are reference-only."""
import numpy as np
from scipy.special import roots_legendre
from trajectory_chain import TrajectorySamples,SI_UNITS
from smooth_density_probe import validate


def shear_map(a,b,s,time):
    return a+s/(2*np.pi)*np.sin(2*np.pi*b)+.2*time,b-.1*time


def make_shear_samples(labels,ids,mass,c0,rho0,times,shear_end=.25,end_time=.05):
    a=np.asarray(labels,dtype=float);times=np.asarray(times,dtype=float);rho0=np.asarray(rho0,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or rho0.shape!=(len(a),) or not np.isfinite(a).all() or not np.isfinite(shear_end) or not np.isfinite(end_time) or end_time<=0:
        raise ValueError('finite prescribed shear, matching labels/density and positive duration required')
    if not np.isfinite(times).all() or np.any(times<0) or np.any(times>end_time):raise ValueError('no unexplained time extrapolation')
    s=shear_end*times/end_time
    x,y=shear_map(a[None,:,0],a[None,:,1],s[:,None],times[:,None])
    positions=np.stack([x,np.broadcast_to(y,x.shape)],axis=-1)%1.
    rho=np.broadcast_to(rho0,(len(times),len(a))).copy()
    data=TrajectorySamples(times,positions,ids,mass,c0,SI_UNITS,rho=rho)
    arrays={'shear_amplitude':s,'reference_eta':s[:,None]*np.cos(2*np.pi*a[None,:,1])}
    audit={'analytic_map_determinant':1.,'material_density_time_derivative':0.,'analytic_velocity_divergence':0.,
        'density_source':'given rho0 along each fixed identity; analytic determinant is one',
        'map_derivatives_or_true_geometry_are_candidate_inputs':False}
    return data,arrays,audit


def scalar_derivatives(a,b):
    k=2*np.pi;ca=np.cos(k*a);sa=np.sin(k*a);cb=np.cos(k*b);sb=np.sin(k*b)
    return {'c':.5+.2*ca*cb,'c_a':-.2*k*sa*cb,'c_b':-.2*k*ca*sb,
        'c_aa':-.2*k*k*ca*cb,'c_bb':-.2*k*k*ca*cb,'c_ab':.2*k*k*sa*sb}


def shear_point_fields(a,b,epsilon,s,D):
    k=2*np.pi;v=scalar_derivatives(a,b);rho0=1+epsilon*np.cos(k*a);rho0_a=-epsilon*k*np.sin(k*a)
    eta=s*np.cos(k*b);eta_b=-s*k*np.sin(k*b)
    parts={'base_laplacian':D*(v['c_aa']+v['c_bb']),
        'metric_correction':D*eta**2*v['c_aa'],
        'mixed_derivative':-2*D*eta*v['c_ab'],
        'shear_gradient':-D*eta_b*v['c_a'],
        'density_drift':D*(rho0_a/rho0)*((1+eta**2)*v['c_a']-eta*v['c_b'])}
    parts['rate']=sum(parts.values())
    return rho0,v['c'],parts


def material_flux(a,b,epsilon,s,D):
    """D*rho0 times the transformed scalar gradient, no second derivatives."""
    k=2*np.pi;rho0=1+epsilon*np.cos(k*a);eta=s*np.cos(k*b)
    c_a=-.2*k*np.sin(k*a)*np.cos(k*b);c_b=-.2*k*np.cos(k*a)*np.sin(k*b)
    return D*rho0*((1+eta**2)*c_a-eta*c_b),D*rho0*(c_b-eta*c_a)


def shear_gauss_reference(N,epsilon,s,D,labels,mass,order):
    validate(N,epsilon,D)
    if not np.isfinite(s) or not isinstance(order,(int,np.integer)) or order<2:raise ValueError('finite shear and integer Gauss order>=2 required')
    a=np.asarray(labels);m=np.asarray(mass)
    if a.shape!=(N*N,2) or m.shape!=(N*N,) or not np.isfinite(a).all() or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('matching original cells and positive saved masses required')
    z,w=roots_legendre(order);offset=z/(2*N);weights=w[:,None]*w[None,:]/(4*N*N)
    keys=['mass','state_numerator','base_laplacian','metric_correction','mixed_derivative','shear_gradient','density_drift','rate']
    result={key:[] for key in keys}
    for start in range(0,len(a),128):
        centers=a[start:start+128];x=centers[:,0,None,None]+offset[None,:,None];y=centers[:,1,None,None]+offset[None,None,:]
        rho,c,parts=shear_point_fields(x,y,epsilon,s,D);measure=rho*weights[None,:,:]
        result['mass'].append(np.sum(measure*np.ones_like(c),axis=(1,2)))
        result['state_numerator'].append(np.sum(measure*c,axis=(1,2)))
        for key,value in parts.items():result[key].append(np.sum(measure*value,axis=(1,2)))
    result={key:np.concatenate(parts) for key,parts in result.items()}
    for key in keys[2:]:result[key]/=m
    result['state']=result['state_numerator']/m
    return result


def shear_face_reference(N,epsilon,s,D,labels,mass,order):
    """Independent integrated material divergence from the four cell faces."""
    validate(N,epsilon,D)
    if not np.isfinite(s) or not isinstance(order,(int,np.integer)) or order<2:raise ValueError('finite shear and integer Gauss order>=2 required')
    a=np.asarray(labels);m=np.asarray(mass)
    if a.shape!=(N*N,2) or m.shape!=(N*N,) or np.any(m<=0) or not np.isfinite(a).all() or not np.isfinite(m).all():raise ValueError('matching cells and raw positive mass required')
    z,w=roots_legendre(order);offset=z/(2*N);weight=w/(2*N);half=1/(2*N)
    x=a[:,0,None];y=a[:,1,None]
    right=material_flux(x+half,y+offset,epsilon,s,D)[0];left=material_flux(x-half,y+offset,epsilon,s,D)[0]
    top=material_flux(x+offset,y+half,epsilon,s,D)[1];bottom=material_flux(x+offset,y-half,epsilon,s,D)[1]
    a_rate=np.sum((right-left)*weight,axis=1)/m;b_rate=np.sum((top-bottom)*weight,axis=1)/m
    return {'a_face_rate':a_rate,'b_face_rate':b_rate,'rate':a_rate+b_rate}


def shear_geometry(N,labels,s,time):
    """Exact unwrapped current area and area centroid; never used by kernel."""
    labels=np.asarray(labels);x,y=shear_map(labels[:,0],labels[:,1],s,time)
    centroid_x=labels[:,0]+s/(2*np.pi)*np.sinc(1/N)*np.sin(2*np.pi*labels[:,1])+.2*time
    return {'true_cell_area':np.full(len(labels),1/(N*N)),
        'unwrapped_node_x':x,'unwrapped_node_y':y,'true_area_centroid_x':centroid_x,
        'node_minus_area_centroid_x':x-centroid_x}
