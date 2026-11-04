"""Independent full 2D material PDE: odd Fourier collocation + DOP853.

No point kernel, graph weights, state clipping, spectral filtering or mass
correction. Point reference states are integrated into saved material means.
"""
import numpy as np
from scipy.fft import fft2,ifft2
from scipy.integrate import DOP853


class ShearReferenceOperator:
    def __init__(self,M,epsilon=.25,D=.02,shear_end=.25,T=.05):
        if not isinstance(M,(int,np.integer)) or M<5 or M%2!=1 or not np.isfinite(epsilon) or abs(epsilon)>=1 or not np.isfinite(D) or D<0 or not np.isfinite(shear_end) or not np.isfinite(T) or T<=0:
            raise ValueError('odd M>=5, positive rho, nonnegative D and finite prescribed motion required')
        self.M=M;self.epsilon=epsilon;self.D=D;self.shear_end=shear_end;self.T=T
        self.axis=np.arange(M)/M;self.a,self.b=np.meshgrid(self.axis,self.axis,indexing='xy')
        self.rho=1+epsilon*np.cos(2*np.pi*self.a)
        k=2*np.pi*np.fft.fftfreq(M,d=1/M);self.ka=k[None,:];self.kb=k[:,None]
        self.cosine_b=np.cos(2*np.pi*self.b);self.maximum_derivative_imaginary_residual=0.

    def real_inverse(self,values):
        z=ifft2(values,workers=1)
        self.maximum_derivative_imaginary_residual=max(self.maximum_derivative_imaginary_residual,float(np.max(abs(z.imag))))
        return z.real

    def gradient(self,c):
        c=np.asarray(c,dtype=float)
        if c.shape!=(self.M,self.M) or not np.isfinite(c).all():raise ValueError('matching finite reference field required')
        hat=fft2(c,workers=1)
        return self.real_inverse(1j*self.ka*hat),self.real_inverse(1j*self.kb*hat)

    def rhs(self,t,flat):
        c=np.asarray(flat,dtype=float).reshape(self.M,self.M)
        if self.D==0:return np.zeros_like(c).ravel()
        ca,cb=self.gradient(c);eta=self.shear_end*(t/self.T)*self.cosine_b
        qa=self.D*self.rho*((1+eta*eta)*ca-eta*cb);qb=self.D*self.rho*(cb-eta*ca)
        divergence=self.real_inverse(1j*self.ka*fft2(qa,workers=1)+1j*self.kb*fft2(qb,workers=1))
        return (divergence/self.rho).ravel()

    def dissipation(self,t,c):
        ca,cb=self.gradient(c);eta=self.shear_end*(t/self.T)*self.cosine_b
        return float(self.D*np.mean(self.rho*(ca*ca+(cb-eta*ca)**2)))

    def initial_scalar(self):return .5+.2*np.cos(2*np.pi*self.a)*np.cos(2*np.pi*self.b)


def project_mass_average(c,rho,N,saved_mass):
    """Exact cell integrals of the represented rho*c Fourier series; keep m."""
    c=np.asarray(c,dtype=float);rho=np.asarray(rho,dtype=float);m=np.asarray(saved_mass,dtype=float)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or c.shape[0]%2!=1 or rho.shape!=c.shape or not isinstance(N,(int,np.integer)) or N<2 or m.shape!=(N*N,) or np.any(m<=0) or np.any(rho<=0) or not all(np.isfinite(x).all() for x in [c,rho,m]):
        raise ValueError('finite odd square reference field, positive rho and original cell masses required')
    M=len(c);frequency=np.fft.fftfreq(M,d=1/M);axis=(np.arange(N)+.5)/N
    phase=np.exp(2j*np.pi*axis[:,None]*frequency[None,:])*np.sinc(frequency/N)[None,:]
    coefficients=fft2(rho*c,workers=1)/(M*M)
    complex_means=(phase@coefficients@phase.T)/(N*N*m.reshape(N,N))
    return complex_means.real.ravel(),complex_means.imag.ravel()


def reference_path(M,max_step,times,epsilon=.25,D=.02,shear_end=.25,T=.05,rtol=1e-10,atol=1e-12,initial=None):
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<2 or times[0]!=0. or times[-1]!=T or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or not np.isfinite(max_step) or max_step<=0 or not np.isfinite(rtol) or rtol<=0 or not np.isfinite(atol) or atol<=0:
        raise ValueError('increasing observation times including both endpoints and positive numerical controls required')
    operator=ShearReferenceOperator(M,epsilon,D,shear_end,T)
    c0=operator.initial_scalar() if initial is None else np.asarray(initial,dtype=float).copy()
    if c0.shape!=(M,M) or not np.isfinite(c0).all():raise ValueError('matching finite initial reference field required')
    solver=DOP853(operator.rhs,0.,c0.ravel(),T,max_step=max_step,rtol=rtol,atol=atol)
    accepted_times=[0.];accepted=[c0.copy()];snapshots=[c0.copy()];next_index=1
    while solver.status=='running':
        message=solver.step()
        if solver.status=='failed':raise RuntimeError('unmodified reference integration failed: '+str(message))
        accepted_times.append(float(solver.t));accepted.append(solver.y.reshape(M,M).copy())
        dense=None
        while next_index<len(times) and times[next_index]<=solver.t:
            t=float(times[next_index])
            if t==solver.t:state=solver.y
            else:
                if dense is None:dense=solver.dense_output()
                state=dense(t)
            snapshots.append(state.reshape(M,M).copy());next_index+=1
    if next_index!=len(times):raise RuntimeError('reference did not cover the requested observations')
    accepted=np.array(accepted);accepted_times=np.array(accepted_times);snapshots=np.array(snapshots)
    inventories=np.mean(operator.rho[None,:,:]*accepted,axis=(1,2));means=inventories/np.mean(operator.rho)
    energies=.5*np.mean(operator.rho[None,:,:]*(accepted-means[:,None,None])**2,axis=(1,2))
    extrema=np.stack([accepted.min(axis=(1,2)),accepted.max(axis=(1,2))],axis=1)
    rates=[];chi=[];mass_rate=[];energy_identity=[]
    for t,c in zip(times,snapshots):
        rate=operator.rhs(float(t),c.ravel()).reshape(M,M);dissipation=operator.dissipation(float(t),c)
        rates.append(rate);chi.append(dissipation);mass_rate.append(float(np.mean(operator.rho*rate)))
        energy_identity.append(float(np.mean(operator.rho*c*rate)+dissipation))
    arrays={'axis':operator.axis,'rho0_grid':operator.rho,'accepted_time':accepted_times,'accepted_point_states':accepted,
        'snapshot_time':times,'point_snapshots':snapshots,'snapshot_rates':np.array(rates),
        'weighted_inventory':inventories,'point_half_variance_energy':energies,'accepted_extrema':extrema,
        'snapshot_chi':np.array(chi),'snapshot_mass_rate':np.array(mass_rate),'snapshot_energy_identity_residual':np.array(energy_identity)}
    audit={'M_per_direction':M,'maximum_step_s':float(max_step),'rtol':rtol,'atol':atol,'accepted_steps':len(accepted_times)-1,
        'RHS_evaluations':solver.nfev,'maximum_actual_step_s':float(np.max(np.diff(accepted_times))),
        'maximum_inventory_change':float(np.max(abs(inventories-inventories[0]))),
        'maximum_half_variance_energy_increase':float(np.max(np.diff(energies))),
        'raw_accepted_point_extrema':[float(extrema[:,0].min()),float(extrema[:,1].max())],
        'maximum_derivative_imaginary_residual':operator.maximum_derivative_imaginary_residual,
        'maximum_snapshot_mass_rate_residual':float(np.max(np.abs(mass_rate))),
        'maximum_snapshot_energy_identity_residual':float(np.max(np.abs(energy_identity))),
        'matrix_free_reference_only':True,'clipping_filtering_or_mass_state_normalization':False,'fluid_orc_enabled':False}
    return arrays,audit
