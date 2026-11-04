"""Independent finite-interval translating contact and exact scalar eigenmode.

Prescribed inviscid contact background, passive scalar only. No thermal or
multicomponent feedback. Perfect scalar transmission: [c]=[rho D c_x]=0.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from experiment import step

METHODS=('ordinary_volume_center','ordinary_mass_center','mass_average_resistance')


@dataclass(frozen=True)
class Contact:
    rho_left: float=2.
    rho_right: float=1.
    D: float=.02
    half_length: float=1.
    speed: float=.3
    amplitude: float=.2

    def __post_init__(self):
        vals=(self.rho_left,self.rho_right,self.D,self.half_length,self.speed,self.amplitude)
        if not np.isfinite(vals).all():raise ValueError('finite parameters required')
        if min(self.rho_left,self.rho_right,self.half_length)<=0 or self.D<0 or self.amplitude<=0:
            raise ValueError('positive density/length/amplitude, nonnegative D required')
        if self.amplitude*max(1.,self.rho_left/self.rho_right)>.5:
            raise ValueError('specified amplitude would leave mass fraction bounds')

    @property
    def k(self):return np.pi/(2*self.half_length)
    @property
    def total_mass(self):return (self.rho_left+self.rho_right)*self.half_length
    def decay(self,t):return np.exp(-self.D*self.k**2*t)

    def mass_coordinate(self,a):
        a=np.asarray(a,dtype=float)
        return self.rho_left*(np.minimum(a,0)+self.half_length)+self.rho_right*np.maximum(a,0)

    def inverse_mass(self,mu):
        mu=np.asarray(mu,dtype=float);split=self.rho_left*self.half_length
        return np.where(mu<=split,mu/self.rho_left-self.half_length,(mu-split)/self.rho_right)

    def mass(self,faces):return np.diff(self.mass_coordinate(faces))

    def scalar(self,a,t):
        a=np.asarray(a,dtype=float)
        mode=np.where(a<=0,np.cos(self.k*(a+self.half_length)),
                      -self.rho_left/self.rho_right*np.cos(self.k*(self.half_length-a)))
        return .5+self.amplitude*self.decay(t)*mode

    def flux(self,a,t):
        a=np.asarray(a,dtype=float)
        phase=np.where(a<=0,a+self.half_length,self.half_length-a)
        return self.rho_left*self.D*self.amplitude*self.k*self.decay(t)*np.sin(self.k*phase)

    def cell_average(self,faces,t):
        a=np.asarray(faces[:-1]);b=np.asarray(faces[1:]);m=self.mass(faces)
        left=np.sin(self.k*(np.minimum(b,0)+self.half_length))-np.sin(self.k*(np.minimum(a,0)+self.half_length))
        right=np.sin(self.k*(self.half_length-np.maximum(b,0)))-np.sin(self.k*(self.half_length-np.maximum(a,0)))
        return .5+self.amplitude*self.rho_left/self.k*self.decay(t)*(left+right)/m

    def resistance_potential(self,a):
        if self.D==0:raise ValueError('resistance potential undefined at D=0')
        a=np.asarray(a,dtype=float)
        return np.where(a<=0,a/(self.rho_left*self.D),a/(self.rho_right*self.D))

    def conductance(self,faces,method):
        faces=np.asarray(faces,dtype=float)
        if faces.ndim!=1 or len(faces)<3 or not np.isfinite(faces).all() or np.any(np.diff(faces)<=0):
            raise ValueError('increasing finite material faces required')
        if method not in METHODS:raise ValueError('unknown contact conductance method')
        if self.D==0:return np.zeros(len(faces)-2)
        if method=='ordinary_volume_center':
            psi=self.resistance_potential((faces[:-1]+faces[1:])/2)
        elif method=='ordinary_mass_center':
            mu=self.mass_coordinate(faces);psi=self.resistance_potential(self.inverse_mass((mu[:-1]+mu[1:])/2))
        else:
            # rho*psi=a/D in both phases when psi(0)=0: exact integral,
            # including any cell containing the density jump.
            psi=(faces[1:]**2-faces[:-1]**2)/(2*self.D*self.mass(faces))
        R=np.diff(psi)
        if np.any(R<=0) or not np.isfinite(R).all():raise RuntimeError('invalid contact resistance')
        return 1/R


def mesh(cells,placement,L=1.):
    if not isinstance(cells,int) or cells<4 or cells%2:raise ValueError('even cells>=4 required')
    faces=np.linspace(-L,L,cells+1)
    if placement=='cut':
        # Move only the contact-aligned face. The contact at zero is now
        # inside a material cell, 3/4 of its width from the left edge.
        faces[cells//2]=(2*L/cells)/3
    elif placement!='aligned':raise ValueError('unknown contact placement')
    return faces


def laplacian(g):
    n=len(g)+1;L=np.zeros((n,n));idx=np.arange(n-1)
    L[idx,idx]+=g;L[idx+1,idx+1]+=g;L[idx,idx+1]-=g;L[idx+1,idx]-=g
    return L


def semidiscrete(c0,m,g,T):
    """Dense matrix exponential of the candidate, not the physical reference."""
    L=laplacian(g);P=expm(-T*L/m[:,None]);c=P@c0
    diag={'min_P':float(P.min()),'max_row_sum_error':float(np.max(abs(P.sum(axis=1)-1))),
          'max_mass_invariant_error':float(np.max(abs(m@P-m))/np.max(m))}
    if min(c)<min(c0)-1e-10 or max(c)>max(c0)+1e-10 or diag['min_P']< -1e-10:
        raise RuntimeError('semidiscrete bounds/sign failure')
    if max(diag['max_row_sum_error'],diag['max_mass_invariant_error'])>1e-10:
        raise RuntimeError('semidiscrete probability/mass failure')
    return c,P,diag


def backward_euler(c0,m,g,T,steps):
    c=c0.copy();mean=m@c0/m.sum();dt=T/steps;mass0=m@c0
    mass=residual=balance=0.;lo=float(c.min());hi=float(c.max());L=laplacian(g)
    tiny=np.finfo(float).tiny;mass_scale=max(abs(float(mass0)),float(np.dot(m,abs(c0))),tiny)
    for _ in range(steps):
        nxt=step(c,m,g,dt)
        mass=max(mass,abs(float(m@(nxt-c0)))/mass_scale)
        residual=max(residual,float(np.linalg.norm(m*(nxt-c)+dt*(L@nxt))/max(float(np.linalg.norm(m*c)),tiny)))
        E=lambda x:.5*np.dot(m,(x-mean)**2)
        err=E(nxt)-E(c)+dt*np.dot(g,np.diff(nxt)**2)+.5*np.dot(m,(nxt-c)**2)
        balance=max(balance,abs(float(err))/max(float(np.dot(m,c*c)),tiny))
        lo=min(lo,float(nxt.min()));hi=max(hi,float(nxt.max()));c=nxt
    if max(mass,residual,balance)>1e-10 or lo<c0.min()-1e-10 or hi>c0.max()+1e-10:
        raise RuntimeError('contact BE algebra failure')
    return c,{'max_relative_mass_drift':mass,'max_relative_solve_residual':residual,
             'max_relative_energy_balance_residual':balance,'min_all_times':lo,'max_all_times':hi}
