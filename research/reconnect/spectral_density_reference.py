"""Independent periodic PDE reference; no point-kernel quantities are used.

c=.5+u(x,t)*cos(2*pi*y), rho=1+epsilon*cos(2*pi*x).
v=sqrt(rho)*u solves v_t=-H v with a raw symmetric Fourier-collocation H.
"""
import numpy as np
from scipy.linalg import eigh


def periodic_second_derivative(M):
    if not isinstance(M,(int,np.integer)) or M<4 or M%2:raise ValueError('even periodic spectral size M>=4 required')
    i=np.arange(M);distance=abs(i[:,None]-i[None,:]);distance=np.minimum(distance,M-distance)
    nonzero=distance>0;D2=np.empty((M,M));d=distance[nonzero]
    D2[nonzero]=-2*np.pi**2*(-1.)**d/np.sin(np.pi*d/M)**2
    np.fill_diagonal(D2,-np.pi**2*(M*M+2)/3)
    return D2


def spectral_system(M,epsilon,D):
    if not np.isfinite(epsilon) or abs(epsilon)>=1 or not np.isfinite(D) or D<0:raise ValueError('positive smooth rho and nonnegative finite D required')
    D2=periodic_second_derivative(M);x=np.arange(M)/M;k=2*np.pi
    rho=1+epsilon*np.cos(k*x);rho_x=-epsilon*k*np.sin(k*x);rho_xx=-epsilon*k*k*np.cos(k*x)
    potential=rho_xx/(2*rho)-rho_x**2/(4*rho**2)
    H=D*(-D2+np.diag(potential+k*k));root=np.sqrt(rho);v0=root*.2*np.cos(k*x)
    eigenvalues,U=eigh(H,check_finite=True)
    return {'x':x,'rho':rho,'root_rho':root,'D2':D2,'potential':potential,'H':H,'v0':v0,'eigenvalues':eigenvalues,'eigenvectors':U}


def project_component_amplitude(q_values,N,saved_mass):
    """Integrate q=rho*u in x via Fourier sinc, y cosine exactly; keep m."""
    q=np.asarray(q_values,dtype=float);m=np.asarray(saved_mass,dtype=float);M=len(q)
    if q.ndim!=1 or not np.isfinite(q).all() or m.shape!=(N*N,) or np.any(m<=0) or not np.isfinite(m).all():raise ValueError('finite reference values and saved positive masses required')
    rows=m.reshape(N,N)
    if not np.array_equal(rows,np.broadcast_to(rows[0],rows.shape)):raise ValueError('this analytic reference requires rho depending only on x')
    frequencies=np.fft.fftfreq(M,d=1/M);coefficients=np.fft.fft(q)/M
    axis=(np.arange(N)+.5)/N;phase=np.exp(2j*np.pi*axis[:,None]*frequencies[None,:])
    q_mean_complex=phase@(coefficients*np.sinc(frequencies/N))
    rho_mean=rows[0]*N*N;amplitude=q_mean_complex.real/rho_mean
    state=.5+np.outer(np.sinc(1/N)*np.cos(2*np.pi*axis),amplitude).ravel()
    return state,amplitude,{'maximum_imaginary_cell_integral':float(np.max(abs(q_mean_complex.imag))),
        'nyquist_coefficient_abs':float(abs(coefficients[M//2])),'x_weighted_mode_inventory':float(coefficients[0].real)}


def reference_path(M,N,epsilon,D,times,saved_mass):
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(times<0) or np.any(np.diff(times)<=0):raise ValueError('increasing nonnegative reference times required')
    system=spectral_system(M,epsilon,D);vals=system['eigenvalues'];U=system['eigenvectors'];coeff=U.T@system['v0']
    states=[];amplitudes=[];controls=[]
    for t in times:
        v=U@(np.exp(-t*vals)*coeff);q=system['root_rho']*v
        state,amp,audit=project_component_amplitude(q,N,saved_mass)
        exact_inventory=.1*epsilon*np.exp(-D*(2*np.pi)**2*t)
        audit.update({'time_s':float(t),'exact_x_weighted_mode_inventory':float(exact_inventory),
            'x_mode_inventory_absolute_difference':abs(audit['x_weighted_mode_inventory']-exact_inventory)})
        states.append(state);amplitudes.append(amp);controls.append(audit)
    diagnostics={'reference_spectral_nodes':M,'raw_H_symmetry_max_difference':float(np.max(abs(system['H']-system['H'].T))),
        'minimum_raw_reference_eigenvalue_per_s':float(vals.min()),
        'raw_constant_x_mode_generator_residual':float(np.max(abs(system['H']@system['root_rho']-D*(2*np.pi)**2*system['root_rho']))),
        'density_drift_included':True,'posthoc_symmetrization_or_eigenvalue_clipping':False,'projection_controls':controls}
    return np.array(states),np.array(amplitudes),diagnostics
