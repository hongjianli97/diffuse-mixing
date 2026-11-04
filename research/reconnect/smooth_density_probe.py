"""One periodic smooth-density manufactured snapshot and independent references.

Material squares and analytic formulas belong only to input/reference production.
The unchanged candidate receives point samples, fixed mass and oracle point rho.
"""
import numpy as np
from scipy.special import roots_legendre
from trajectory_chain import TrajectorySamples,SI_UNITS
from kernel2d_study import fixture


def validate(N,epsilon,D):
    if not isinstance(N,(int,np.integer)) or N<2 or not np.isfinite(epsilon) or abs(epsilon)>=1 or not np.isfinite(D) or D<0:
        raise ValueError('integer N>=2, |epsilon|<1 and nonnegative finite D required')


def point_fields(x,y,epsilon=.25,D=.02):
    k=2*np.pi;cx=np.cos(k*x);cy=np.cos(k*y);sx=np.sin(k*x)
    rho=1+epsilon*cx;c=.5+.2*cx*cy
    lap=-2*D*k*k*(c-.5)
    drift=D*.2*epsilon*k*k*sx*sx*cy/rho
    return rho,c,lap,drift,lap+drift


def analytic_reference(N,epsilon=.25,D=.02):
    validate(N,epsilon,D)
    axis=(np.arange(N)+.5)/N;labels=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    k=2*np.pi;s=np.sinc(1/N);s2=np.sinc(2/N)
    C=s*np.cos(k*labels[:,0]);Y=s*np.cos(k*labels[:,1]);C2=(1+s2*np.cos(2*k*labels[:,0]))/2
    rho_average=1+epsilon*C;mass=rho_average/(N*N)
    state=.5+.2*(C+epsilon*C2)*Y/rho_average
    laplacian=-2*D*k*k*.2*(C+epsilon*C2)*Y/rho_average
    drift=D*.2*epsilon*k*k*(1-C2)*Y/rho_average
    # Direct divergence integral, independent of adding the saved split rates.
    total=D*.2*k*k*(-2*C+epsilon*(1-3*C2))*Y/rho_average
    return {'labels':labels,'mass':mass,'state':state,'rho_average':rho_average,
        'rho_point':1+epsilon*np.cos(k*labels[:,0]),'laplacian_rate':laplacian,'drift_rate':drift,'rate':total,
        'true_area':np.full(N*N,1/(N*N))}


def make_samples(N=64,epsilon=.25,D=.02):
    ref=analytic_reference(N,epsilon,D);old,labels,_=fixture(N)
    data=TrajectorySamples(old.time,old.positions,old.ids,ref['mass'],ref['state'],SI_UNITS,
        rho=np.broadcast_to(ref['rho_point'],(len(old.time),len(old.mass))).copy())
    return data,ref


def gauss_reference(N,epsilon,D,labels,order):
    """Direct tensor-product rho, rho*c and rho*PDE integrals, in chunks."""
    validate(N,epsilon,D)
    if not isinstance(order,(int,np.integer)) or order<2:raise ValueError('integer quadrature order>=2 required')
    a=np.asarray(labels,dtype=float)
    if a.shape!=(N*N,2) or not np.isfinite(a).all():raise ValueError('matching finite material reference centers required')
    z,w=roots_legendre(order);offset=z/(2*N);weights=w[:,None]*w[None,:]/(4*N*N)
    result={key:[] for key in ['mass','component_mass','laplacian_numerator','drift_numerator','rate_numerator']}
    for start in range(0,len(a),128):
        centers=a[start:start+128];x=centers[:,0,None,None]+offset[None,:,None];y=centers[:,1,None,None]+offset[None,None,:]
        rho,c,lap,drift,total=point_fields(x,y,epsilon,D)
        measure=rho*weights[None,:,:]
        for key,value in [('mass',np.ones_like(c)),('component_mass',c),('laplacian_numerator',lap),('drift_numerator',drift),('rate_numerator',total)]:
            result[key].append(np.sum(measure*value,axis=(1,2)))
    result={key:np.concatenate(parts) for key,parts in result.items()}
    # These are independently integrated means, not inputs substituted for m/c0.
    result['state']=result['component_mass']/result['mass']
    for key in ['laplacian','drift','rate']:result[key+'_rate' if key!='rate' else 'rate']=result[key+'_numerator']/result['mass']
    return result
