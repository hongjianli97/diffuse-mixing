"""Separable ordinary FV on a static, aligned 1:4 Cartesian contact.

The normal ring alone is assembled. All transverse FFT coefficients of the
supplied mass means are retained. No clipping, zero-eigenvalue reset, mean
replacement, probability normalization, or concentration extrapolation.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt
import numpy as np


MATERIAL_PRIOR='static Cartesian material cells, supplied points at centers, constant density per cell'


def fraction_sqrt(value):
    a,b=isqrt(value.numerator),isqrt(value.denominator)
    if a*a!=value.numerator or b*b!=value.denominator:
        raise ValueError('this limited exact adapter requires rational neighboring density-product roots')
    return F(a,b)


@dataclass(frozen=True)
class FVModes:
    shape: tuple
    periods: tuple
    D_exact: F
    area_exact: F
    rho: np.ndarray
    sqrt_rho: np.ndarray
    normal_matrix: np.ndarray
    normal_eigenvalues: np.ndarray
    normal_eigenvectors: np.ndarray
    transverse_eigenvalues: np.ndarray
    normal_g_exact: tuple
    normal_matrix_rounding_row_bound_exact: F
    eigen_residual_frobenius: float
    orthogonality_residual_frobenius: float


def build_modes(rho,mass,D,periods=(1,1),*,model_prior,thickness=1):
    if model_prior!=MATERIAL_PRIOR:raise ValueError('explicit static material geometry prior required')
    density=np.array(rho,dtype=float,copy=True);m=np.array(mass,dtype=float,copy=True)
    if density.ndim!=2 or density.shape!=m.shape or min(density.shape)<4:
        raise ValueError('matching complete grids with at least four cells per axis required')
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(m)) or np.any(density<=0) or np.any(m<=0):
        raise ValueError('finite positive density and mass required')
    if not np.array_equal(density,np.broadcast_to(density[0],density.shape)):
        raise ValueError('this adapter requires transverse-independent stationary density')
    box=tuple(F(float(x)) for x in periods);diffusion=F(float(D));b=F(float(thickness))
    if len(box)!=2 or min(box)<=0 or diffusion<=0 or b<=0:raise ValueError('positive D and geometry required')
    ny,nx=density.shape;dx,dy=box[0]/nx,box[1]/ny;area=b*dx*dy
    if any(F(float(a))!=F(float(r))*area for a,r in zip(m.flat,density.flat)):
        raise ValueError('original mass must exactly equal rho times material volume; no repair')
    r=tuple(F(float(x)) for x in density[0]);g=[]
    for i in range(nx):
        g.append(b*dy/(dx/(2*r[i]*diffusion)+dx/(2*r[(i+1)%nx]*diffusion)))
    exact=[[F(0)]*nx for _ in range(nx)]
    for i in range(nx):
        j=(i+1)%nx
        exact[i][i]+=g[i]/(r[i]*area);exact[j][j]+=g[i]/(r[j]*area)
        value=-g[i]/(area*fraction_sqrt(r[i]*r[j]))
        exact[i][j]+=value;exact[j][i]+=value
    B=np.asarray([[float(value) for value in row] for row in exact])
    rounding=max(sum((abs(F(float(v))-a) for v,a in zip(row,original)),F(0)) for row,original in zip(B,exact))
    eigenvalues,Q=np.linalg.eigh(B) # Raw eigenvalues, including any roundoff-negative zero mode.
    k=np.fft.fftfreq(ny)*ny
    transverse=4*float(diffusion)/float(dy)**2*np.sin(np.pi*k/ny)**2
    residual=float(np.linalg.norm(B@Q-Q*eigenvalues,ord='fro'))
    orthogonal=float(np.linalg.norm(Q.T@Q-np.eye(nx),ord='fro'))
    return FVModes((ny,nx),box,diffusion,area,density,np.sqrt(density[0]),B,eigenvalues,Q,
        transverse,tuple(g),rounding,residual,orthogonal)


def coefficients(modes,material_means):
    state=np.asarray(material_means,dtype=float)
    if state.shape!=modes.shape or not np.all(np.isfinite(state)):
        raise ValueError('finite original material means on the entire grid required')
    return np.fft.fft(state*modes.sqrt_rho[None,:],axis=0,norm='ortho')@modes.normal_eigenvectors


def reconstruct(modes,coeff):
    value=np.fft.ifft(coeff@modes.normal_eigenvectors.T,axis=0,norm='ortho')/modes.sqrt_rho[None,:]
    return value.real.copy(),value.imag.copy()


def propagate(modes,coeff,T,steps=None):
    if not np.isfinite(T) or T<0:raise ValueError('finite nonnegative time required')
    lam=modes.transverse_eigenvalues[:,None]+modes.normal_eigenvalues[None,:]
    if steps is None:factor=np.exp(-T*lam)
    else:
        if isinstance(steps,bool) or not isinstance(steps,(int,np.integer)) or steps<=0:
            raise ValueError('positive integer implicit Euler step count required')
        if np.any(1+(T/steps)*lam<=0):raise ValueError('implicit Euler factors are undefined')
        factor=np.exp(-steps*np.log1p((T/steps)*lam))
    return reconstruct(modes,coeff*factor)


def direct_structured_action(modes,material_means):
    """Apply the normal ring and transverse eigenvalues without normal eigh."""
    z=np.fft.fft(np.asarray(material_means)*modes.sqrt_rho[None,:],axis=0,norm='ortho')
    value=np.fft.ifft(-(z@modes.normal_matrix+modes.transverse_eigenvalues[:,None]*z),
        axis=0,norm='ortho')/modes.sqrt_rho[None,:]
    return value.real.copy(),value.imag.copy()
