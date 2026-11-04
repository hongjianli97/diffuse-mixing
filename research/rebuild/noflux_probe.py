"""One frozen bounded manufactured state, independent of the graph objective.

The scalar and its complete gradient vanish relative to background on the
outer material-coordinate rectangle. Reference operators are instantaneous
diffusion actions, not a claimed unforced time-dependent manufactured solution.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import roots_legendre,expit
from becker_polygon_benchmark import from_world,density_velocity
from material_polygon_history import signed_area


def bump(s):
    s=np.asarray(s)
    return 16*s*s*(1-s)*(1-s),32*s*(1-s)*(1-2*s),32*(1-6*s+6*s*s)


@dataclass(frozen=True)
class NoFluxProbe:
    xi_bounds: tuple
    eta_bounds: tuple=(-.5,.5)
    time: float=.05
    T: float=.1
    background: float=.5
    amplitude: float=.25

    def __post_init__(self):
        if len(self.xi_bounds)!=2 or len(self.eta_bounds)!=2 or not np.isfinite([*self.xi_bounds,*self.eta_bounds,self.time,self.T,self.background,self.amplitude]).all():
            raise ValueError('finite domain, time and scalar parameters required')
        if self.xi_bounds[1]<=self.xi_bounds[0] or self.eta_bounds[1]<=self.eta_bounds[0] or self.T<=0 or not 0<=self.time<=self.T:
            raise ValueError('positive domain widths, T and time within interval required')
        if self.amplitude<0 or self.background<0 or self.background+self.amplitude>1:
            raise ValueError('predefined scalar range must be within [0,1]; no clipping')

    @property
    def shear(self):return .6*np.sin(np.pi*self.time/self.T)

    @property
    def rotation(self):
        theta=np.pi*self.time/(3*self.T);c=np.cos(theta);s=np.sin(theta)
        return np.array([[c,-s],[s,c]])

    def canonical(self,xi,eta):
        L=self.xi_bounds[1]-self.xi_bounds[0];H=self.eta_bounds[1]-self.eta_bounds[0]
        f,fp,fpp=bump((xi-self.xi_bounds[0])/L);g,gp,gpp=bump((eta-self.eta_bounds[0])/H);a=self.amplitude
        return (self.background+a*f*g,a*fp*g/L,a*f*gp/H,
                a*fpp*g/(L*L),a*fp*gp/(L*H),a*f*gpp/(H*H))

    def world(self,points):
        q=from_world(np.asarray(points)[None,...],np.array([self.time]),self.T)[0]
        c,cx,cy,cxx,cxy,cyy=self.canonical(q[...,0],q[...,1]);k=self.shear
        grad=np.stack((cx,cy-k*cx),axis=-1)@self.rotation.T
        laplacian=(1+k*k)*cxx-2*k*cxy+cyy
        return c,grad,laplacian,q,cx,cy


def rho_log_stream_gradient(flow,xi,t):
    z=flow.z_at_xi(flow.u1*t-xi);u=flow.u(z);uz=(flow.u2-flow.u1)*expit(z)*expit(-z)
    return flow.j/u,flow.j*uz/(flow.B*u*u)


def point_diffusion_rhs(probe,flow,points,D):
    if not np.isfinite(D) or D<0:raise ValueError('finite nonnegative physical D required')
    value,gradient,laplacian,q,cx,cy=probe.world(points)
    rho,logprime=rho_log_stream_gradient(flow,q[...,0],probe.time);k=probe.shear
    rhs=D*(laplacian+logprime*((1+k*k)*cx-k*cy))
    return value,gradient,rho,rhs


def mass_coordinate_average(probe,flow,mass_edges,eta_edges,columns,rows,order=64):
    """Exact averaging measure dmu*deta; constants integrated analytically."""
    edges=np.asarray(mass_edges);y=np.asarray(eta_edges);z,w=roots_legendre(order)
    mu=(edges[:-1]+edges[1:])[:,None]/2+np.diff(edges)[:,None]*z/2
    xi=flow.x_mu(mu,probe.time);L=probe.xi_bounds[1]-probe.xi_bounds[0]
    stream=bump((xi-probe.xi_bounds[0])/L)[0]@(w/2)
    v,vw=roots_legendre(3);eta=(y[:-1]+y[1:])[:,None]/2+np.diff(y)[:,None]*v/2
    H=probe.eta_bounds[1]-probe.eta_bounds[0]
    transverse=bump((eta-probe.eta_bounds[0])/H)[0]@(vw/2)
    return probe.background+probe.amplitude*stream[np.asarray(columns,dtype=int)]*transverse[np.asarray(rows,dtype=int)]


def volume_reference(probe,flow,quads,fixed_mass,D,order=64):
    p=np.asarray(quads);m=np.asarray(fixed_mass);z,w=roots_legendre(order);u=(z+1)/2;weights=w[:,None]*w[None,:]/4
    points=p[:,0,None,None,:]+u[None,:,None,None]*(p[:,1]-p[:,0])[:,None,None,:]+u[None,None,:,None]*(p[:,3]-p[:,0])[:,None,None,:]
    c,gradient,rho,rhs=point_diffusion_rhs(probe,flow,points,D)
    measure=rho*weights[None,:,:]*signed_area(p)[:,None,None]
    actual_mass=measure.sum(axis=(1,2));average=np.sum(measure*c,axis=(1,2))/m;rate=np.sum(measure*rhs,axis=(1,2))/m
    return {'actual_mass':actual_mass,'average':average,'rate':rate,
        'sampled_point_minimum':float(c.min()),'sampled_point_maximum':float(c.max())}
