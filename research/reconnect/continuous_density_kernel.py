"""Continuous comparators for the saved smooth-density snapshot, no graph build.

rho(x)=1+epsilon*cos(2*pi*x), c=.5+.2*cos(2*pi*x)*cos(2*pi*y).
Use exact y-mode separation and fixed analytic extensions of saved cell inputs.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import roots_legendre
from positive_kernel_2d import slope_over_radius
from smooth_density_probe import validate,analytic_reference


@dataclass(frozen=True)
class PolarRule:
    radial: int
    angular: int
    cell_x: int

    def __post_init__(self):
        if any(not isinstance(v,(int,np.integer)) or v<2 for v in [self.radial,self.angular,self.cell_x]) or self.angular<4 or self.angular%2:
            raise ValueError('positive integer radial/cell orders and even angular order>=4 required')


def input_extension(x,N,epsilon):
    """rho_point, rho_average, saved-state y amplitude, raw weight ratio."""
    k=2*np.pi;s=np.sinc(1/N);s2=np.sinc(2/N);cx=np.cos(k*x)
    rho=1+epsilon*cx;average=1+epsilon*s*cx
    cos2_average=(1+s2*np.cos(2*k*x))/2
    amplitude=.2*s*(s*cx+epsilon*cos2_average)/average
    return rho,average,amplitude,average/rho


def x_coefficient(x,N,epsilon,D,h,rule,closure='physical'):
    """Coefficient F(x) in r(x,y)=F(x)*cos(2*pi*y).

    physical: true c and unit volume quadrature density.
    saved_inputs: exact analytic cell-average extension and rho_bar/rho_point.
    No fits, correction of the candidate or reinterpretation as a new solver.
    """
    validate(N,epsilon,D);x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(h) or not 0<h<.25:raise ValueError('finite x vector and periodic support with 0<h<.25 required')
    if closure not in ['physical','saved_inputs']:raise ValueError('unknown continuous comparator')
    if D==0:return np.zeros_like(x)
    z,w=roots_legendre(rule.radial);r=h*(z+1);radial_weight=h*w*r*slope_over_radius(r,h)
    theta=2*np.pi*np.arange(rule.angular)/rule.angular
    u=r[:,None]*np.cos(theta)[None,:];v=r[:,None]*np.sin(theta)[None,:]
    mode_y=np.cos(2*np.pi*v);angular_weight=2*np.pi/rule.angular
    answer=np.empty_like(x)
    for start in range(0,len(x),32):
        center=x[start:start+32,None,None];other=center+u[None,:,:]
        rho0=1+epsilon*np.cos(2*np.pi*center);rho1=1+epsilon*np.cos(2*np.pi*other)
        harmonic=2*D*rho0*rho1/(rho0+rho1)
        if closure=='physical':
            f0=.2*np.cos(2*np.pi*center);f1=.2*np.cos(2*np.pi*other);weight1=1.
        else:
            _,_,f0,_=input_extension(center,N,epsilon);_,_,f1,weight1=input_extension(other,N,epsilon)
        integrand=weight1*harmonic*(f1*mode_y[None,:,:]-f0)
        radial_integral=np.sum(integrand,axis=2)*angular_weight
        answer[start:start+len(center)]=2*np.sum(radial_integral*radial_weight[None,:],axis=1)/rho0[:,0,0]
    return answer


def reference_rates(N,epsilon,D,h,rule):
    """True rho-weighted physical-cell average and saved-input center action."""
    validate(N,epsilon,D);axis=(np.arange(N)+.5)/N;z,w=roots_legendre(rule.cell_x)
    subpoints=axis[:,None]+z[None,:]/(2*N)
    point_physical=x_coefficient(subpoints.ravel(),N,epsilon,D,h,rule,'physical').reshape(N,rule.cell_x)
    rho_sub=1+epsilon*np.cos(2*np.pi*subpoints)
    rho_bar=1+epsilon*np.sinc(1/N)*np.cos(2*np.pi*axis)
    physical_x=np.sinc(1/N)*np.sum(point_physical*rho_sub*(w/2)[None,:],axis=1)/rho_bar
    input_x=x_coefficient(axis,N,epsilon,D,h,rule,'saved_inputs')
    transverse=np.cos(2*np.pi*axis)
    return {'physical_rate':np.outer(transverse,physical_x).ravel(),
        'input_closure_rate':np.outer(transverse,input_x).ravel(),
        'physical_x_coefficient':physical_x,'input_x_coefficient':input_x}


def split_budget(mass,local_rate,discrete_rate,physical_rate,input_rate):
    m=np.asarray(mass);local=np.asarray(local_rate);discrete=np.asarray(discrete_rate)
    physical=np.asarray(physical_rate);inputs=np.asarray(input_rate)
    if m.ndim!=1 or any(a.shape!=m.shape for a in [local,discrete,physical,inputs]) or any(not np.isfinite(a).all() for a in [m,local,discrete,physical,inputs]) or np.any(m<=0):
        raise ValueError('matching finite rate vectors and positive masses required')
    norm2=float(m@(local*local))
    if norm2<=0:raise ValueError('nonzero local-PDE rate norm required for relative budget')
    components=np.array([physical-local,inputs-physical,discrete-inputs]);total=discrete-local
    gram=(components*m[None,:])@components.T/norm2
    total2=float(m@(total*total));projections=(components@(m*total))/total2 if total2>0 else None
    summary={'total_relative_error':float(np.sqrt(total2/norm2)),
        'finite_width_relative_norm':float(np.sqrt(gram[0,0])),
        'averaged_input_point_density_closure_relative_norm':float(np.sqrt(gram[1,1])),
        'node_quadrature_relative_norm':float(np.sqrt(gram[2,2])),
        'normalized_Gram':gram.tolist(),
        'signed_projection_onto_total_error':None if projections is None else projections.tolist(),
        'signed_vector_reconstruction_max_residual_per_s':float(np.max(abs(total-components.sum(axis=0)))),
        'norms_are_not_assumed_additive':True,'split_is_a_fixed_comparator_chain_not_unique_causal_fractions':True}
    return components,summary
