"""New Becker Pr=3/4 reconstruction; SI units, unit transverse area."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class Becker:
    delta: float = .08
    mach: float = 2.
    gamma: float = 1.4
    rho1: float = 1.
    p1: float = 1.
    R: float = 1.

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in (self.delta,self.mach,self.gamma,self.rho1,self.p1,self.R)):
            raise ValueError('finite positive physical parameters required')
        if self.mach<=1 or self.gamma<=1:raise ValueError('supersonic Mach and gamma>1 required')

    @property
    def u1(self):return self.mach*np.sqrt(self.gamma*self.p1/self.rho1)
    @property
    def u2(self):return self.u1*((self.gamma-1)*self.mach**2+2)/((self.gamma+1)*self.mach**2)
    @property
    def j(self):return self.rho1*self.u1
    @property
    def B(self):return self.delta*self.j/((self.u1+self.u2)*np.log(9.))
    @property
    def eta(self):return .75*self.B*(self.gamma+1)/(2*self.gamma)*(self.u1-self.u2)
    @property
    def cp(self):return self.gamma*self.R/(self.gamma-1)
    @property
    def H(self):return self.cp*self.p1/(self.rho1*self.R)+self.u1**2/2

    def u(self,z):return self.u1*expit(-np.asarray(z))+self.u2*expit(np.asarray(z))

    def xi(self,z):
        z=np.asarray(z)
        return self.B/self.j*(self.u1*(np.log(2)-np.logaddexp(0,-z))
                                    +self.u2*(np.logaddexp(0,z)-np.log(2)))

    def z_at_xi(self,xi):
        xi=np.asarray(xi,dtype=float)
        if not np.isfinite(xi).all():raise ValueError('nonfinite coordinate')
        z=(xi*self.j/self.B-(self.u1-self.u2)*np.log(2))/np.where(xi<0,self.u1,self.u2)
        for _ in range(16):z=z-(self.xi(z)-xi)/(self.B/self.j*self.u(z))
        if np.max(abs(self.xi(z)-xi))>1e-12:raise RuntimeError('profile inversion failed')
        return z

    def mu(self,a):return self.B*(self.z_at_xi(1.)-self.z_at_xi(-np.asarray(a)))
    @property
    def total(self):return float(self.mu(1.))
    def z_mu(self,mu,t):return self.z_at_xi(1.)+(self.j*t-np.asarray(mu))/self.B
    def x_mu(self,mu,t):return self.u1*t-self.xi(self.z_mu(mu,t))
    def rho_mu(self,mu,t):return self.j/self.u(self.z_mu(mu,t))

    def fluxes(self,z):
        u=self.u(z);w=expit(z); A=(self.gamma+1)/(2*self.gamma)
        du=-self.j*A*(self.u1-self.u2)**2*w*expit(-z)/((4*self.eta/3)*u)
        temp=(self.H-u*u/2)/self.cp
        rho=self.j/u; p=rho*self.R*temp; tau=4*self.eta/3*du
        heat=(self.eta*self.cp/.75)*u*du/self.cp
        return rho*u, self.j*u+p-tau, self.j*self.H-tau*u+heat


def initial(edges):
    edges=np.asarray(edges);total=edges[-1]; widths=np.diff(edges)
    return .5+.25*np.cos(np.pi*(edges[:-1]+edges[1:])/(2*total))*np.sinc(widths/(2*total))
