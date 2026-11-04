"""1D mass/true-face density and a clearly separate midpoint-dual proxy.

No oracle density, gradient, F/C fit, clipping or mass normalization enters
these geometry and cellwise-resistance functions.
"""
import numpy as np


def positive_widths(faces):
    faces=np.asarray(faces,dtype=float)
    if faces.ndim not in (1,2) or faces.shape[-1]<3 or not np.isfinite(faces).all():
        raise ValueError('finite 1D faces or face histories required')
    widths=np.diff(faces,axis=-1)
    if np.any(widths<=0):raise ValueError('face identity/order lost or nonpositive material volume')
    return widths


def density_from_mass_faces(mass,faces):
    m=np.asarray(mass,dtype=float);V=positive_widths(faces)
    if m.ndim!=1 or len(m)!=V.shape[-1] or not np.isfinite(m).all() or np.any(m<=0):
        raise ValueError('fixed positive material masses required')
    return m/V


def midpoint_dual_faces(points,boundaries):
    x=np.asarray(points,dtype=float);b=np.asarray(boundaries,dtype=float)
    if x.ndim not in (1,2) or x.shape[-1]<2 or b.shape!=x.shape[:-1]+(2,):
        raise ValueError('compatible tracer and two boundary trajectories required')
    if not np.isfinite(x).all() or not np.isfinite(b).all() or np.any(np.diff(x,axis=-1)<=0):
        raise ValueError('finite distinct ordered tracer identities required')
    if np.any(b[...,0]>=x[...,0]) or np.any(b[...,1]<=x[...,-1]):
        raise ValueError('boundary trajectories must enclose all tracers')
    faces=np.concatenate((b[...,:1],(x[...,:-1]+x[...,1:])/2,b[...,1:]),axis=-1)
    positive_widths(faces)
    return faces


def cellwise_conductance(faces,density,D):
    V=positive_widths(faces);rho=np.asarray(density,dtype=float)
    if rho.shape!=V.shape or not np.isfinite(rho).all() or np.any(rho<=0):
        raise ValueError('positive cellwise density with matching geometry required')
    if not np.isscalar(D) or not np.isfinite(D) or D<0:raise ValueError('finite nonnegative scalar D required')
    if D==0:return np.zeros(V.shape[:-1]+(V.shape[-1]-1,))
    half=V/(2*rho*D)
    return 1/(half[...,:-1]+half[...,1:])
