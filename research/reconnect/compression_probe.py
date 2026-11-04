"""Given smooth compressible map; geometry/derivatives are reference-only.

Candidate adapter receives only TrajectorySamples, never J or material faces.
All integrals keep the supplied original masses and mass-average scalar.
"""
import numpy as np
from scipy.special import roots_legendre
from trajectory_chain import TrajectorySamples,SI_UNITS
from smooth_density_probe import validate


def map_x(a,alpha,time):
    return a+alpha/(2*np.pi)*np.sin(2*np.pi*a)+.2*time


def make_compression_samples(labels,ids,mass,c0,rho0_point,times,alpha_end=.25,end_time=.05):
    a=np.asarray(labels,dtype=float);times=np.asarray(times,dtype=float);rho0=np.asarray(rho0_point,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or rho0.shape!=(len(a),) or not np.isfinite(alpha_end) or abs(alpha_end)>=1 or not np.isfinite(end_time) or end_time<=0:
        raise ValueError('finite prescribed map with |alpha_end|<1 and matching reference labels required')
    if np.any(times<0) or np.any(times>end_time):raise ValueError('sample times must stay in explicitly defined motion interval')
    alpha=alpha_end*times/end_time;cosine=np.cos(2*np.pi*a[:,0]);J=1+alpha[:,None]*cosine[None,:]
    positions=np.empty((len(times),len(a),2));positions[:,:,0]=np.mod(map_x(a[None,:,0],alpha[:,None],times[:,None]),1.)
    positions[:,:,1]=np.mod(a[None,:,1]-.1*times[:,None],1.)
    rho=rho0[None,:]/J
    data=TrajectorySamples(times,positions,ids,mass,c0,SI_UNITS,rho=rho)
    Jt=alpha_end/end_time*cosine[None,:];divergence=Jt/J;rho_t=-rho0[None,:]*Jt/(J*J)
    diagnostics={'minimum_map_J':float(J.min()),'maximum_map_J':float(J.max()),
        'rho_J_vs_given_rho0_max_difference':float(np.max(abs(rho*J-rho0[None,:]))),
        'material_continuity_max_residual':float(np.max(abs(rho_t+rho*divergence))),
        'map_J_or_velocity_gradient_is_candidate_input':False,'density_is_evaluated_by_rho0_over_J_not_overwritten':True}
    return data,{'alpha':alpha,'J':J,'material_rho_time_derivative':rho_t,'velocity_divergence':divergence},diagnostics


def material_point_fields(a,b,epsilon,alpha,D):
    k=2*np.pi;cosine=np.cos(k*a);sine=np.sin(k*a);cy=np.cos(k*b)
    rho0=1+epsilon*cosine;J=1+alpha*cosine;rho=rho0/J
    rho0a=-epsilon*k*sine;Ja=-alpha*k*sine
    c=.5+.2*cosine*cy;ca=-.2*k*sine*cy;caa=-.2*k*k*cosine*cy;cbb=caa
    physical_laplacian=D*(caa/J**2-ca*Ja/J**3+cbb)
    physical_drift=D*(rho0a/rho0-Ja/J)*ca/J**2
    return rho0,J,rho,c,physical_laplacian,physical_drift,physical_laplacian+physical_drift


def analytic_material_rate(N,epsilon,alpha,D,mass,c0,labels):
    validate(N,epsilon,D)
    if not np.isfinite(alpha) or abs(alpha)>=1:raise ValueError('orientation-preserving smooth map required')
    a=np.asarray(labels);m=np.asarray(mass);state=np.asarray(c0);k=2*np.pi;half=1/(2*N)
    if a.shape!=(N*N,2) or m.shape!=(N*N,) or state.shape!=m.shape or np.any(m<=0):raise ValueError('matching original material means and masses required')
    def material_diffusive_gradient_flux(x):
        return -D*.2*k*(1+epsilon*np.cos(k*x))/(1+alpha*np.cos(k*x))**2*np.sin(k*x)
    left=a[:,0]-half;right=a[:,0]+half
    longitudinal=(material_diffusive_gradient_flux(right)-material_diffusive_gradient_flux(left))/N*np.sinc(1/N)*np.cos(k*a[:,1])/m
    transverse=-D*k*k*(state-.5)
    return longitudinal+transverse,{'longitudinal_rate':longitudinal,'transverse_rate':transverse}


def compression_geometry(N,labels,alpha,time):
    a=np.asarray(labels);half=1/(2*N);left=map_x(a[:,0]-half,alpha,time);right=map_x(a[:,0]+half,alpha,time)
    width=right-left;area=width/N;closed_area=(1+alpha*np.sinc(1/N)*np.cos(2*np.pi*a[:,0]))/(N*N)
    node=map_x(a[:,0],alpha,time);center=(left+right)/2
    return {'true_cell_area_from_unwrapped_faces':area,'closed_form_cell_area':closed_area,
        'unwrapped_left_faces':left,'unwrapped_right_faces':right,'true_geometric_center_x':center,
        'unwrapped_node_x':node,'node_minus_geometric_center_x':node-center,'true_cell_width':width}


def compression_gauss_reference(N,epsilon,alpha,D,labels,saved_mass,order):
    validate(N,epsilon,D)
    if not isinstance(order,(int,np.integer)) or order<2:raise ValueError('integer quadrature order>=2 required')
    a=np.asarray(labels);m=np.asarray(saved_mass);z,w=roots_legendre(order);offset=z/(2*N);weights=w[:,None]*w[None,:]/(4*N*N)
    values={key:[] for key in ['mass','state_numerator','rate_numerator','laplacian_numerator','drift_numerator','area']}
    for start in range(0,len(a),128):
        centers=a[start:start+128];x=centers[:,0,None,None]+offset[None,:,None];y=centers[:,1,None,None]+offset[None,None,:]
        rho0,J,rho,c,lap,drift,total=material_point_fields(x,y,epsilon,alpha,D)
        measure=rho0*weights[None,:,:]
        for key,value in [('mass',np.ones_like(c)),('state_numerator',c),('rate_numerator',total),('laplacian_numerator',lap),('drift_numerator',drift)]:
            values[key].append(np.sum(measure*value,axis=(1,2)))
        values['area'].append(np.sum(J*weights[None,:,:]*np.ones_like(c),axis=(1,2)))
    result={key:np.concatenate(parts) for key,parts in values.items()}
    for dest,source in [('state','state_numerator'),('rate','rate_numerator'),('laplacian_rate','laplacian_numerator'),('drift_rate','drift_numerator')]:result[dest]=result[source]/m
    return result
