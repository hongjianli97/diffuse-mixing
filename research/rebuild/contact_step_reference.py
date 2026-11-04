"""Finite-interval exact transmission series for a scalar step at the contact.

Equal D and equal half-lengths, rho discontinuous, no-flux outer endpoints.
No candidate conductances, clipping, normalization or fitted coefficients.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import erfcx
from contact import Contact


@dataclass(frozen=True)
class StepReference:
    flow: Contact
    left: float=1.
    right: float=0.

    def __post_init__(self):
        if not np.isfinite([self.left,self.right]).all() or not 0<=self.left<=1 or not 0<=self.right<=1:
            raise ValueError('finite initial mass fractions in [0,1] required')

    @property
    def mean(self):
        f=self.flow
        return (f.rho_left*self.left+f.rho_right*self.right)/(f.rho_left+f.rho_right)

    def initial_average(self,faces):
        f=self.flow;a=np.asarray(faces[:-1]);b=np.asarray(faces[1:])
        ml=f.rho_left*(np.minimum(b,0)-np.minimum(a,0))
        mr=f.rho_right*(np.maximum(b,0)-np.maximum(a,0))
        return (ml*self.left+mr*self.right)/(ml+mr)

    def tail_bound(self,t,terms):
        """Uniform concentration-tail bound from a decreasing geometric majorant.

        For m>=K: (m+.5)^2 >= (K+.5)^2+(m-K)*(2K+1).
        This bounds the infinite omitted series, not just a K/2K difference.
        """
        f=self.flow
        if t<=0 or f.D<=0 or terms<1:raise ValueError('positive time,D and terms required')
        a=f.D*(np.pi/f.half_length)**2*t;z=terms+.5
        C=2*abs(self.left-self.right)*max(f.rho_left,f.rho_right)/(np.pi*(f.rho_left+f.rho_right))
        return float(C*np.exp(-a*z*z)/(z*(-np.expm1(-a*(2*terms+1)))))

    def flux_tail_bound(self,t,terms):
        f=self.flow
        if t<=0 or f.D<=0 or terms<1:raise ValueError('positive time,D and terms required')
        a=f.D*(np.pi/f.half_length)**2*t;z=terms+.5
        C=2*f.rho_left*f.rho_right*f.D*abs(self.left-self.right)/((f.rho_left+f.rho_right)*f.half_length)
        return float(C*np.exp(-a*z*z)/(-np.expm1(-a*(2*terms+1))))

    def terms_for(self,t,tolerance=1e-13):
        if not 0<tolerance<1:raise ValueError('tolerance must be in (0,1)')
        f=self.flow
        if t<=0 or f.D<=0:raise ValueError('series selection needs positive time and D')
        a=f.D*(np.pi/f.half_length)**2*t
        K=max(8,int(np.ceil(np.sqrt(np.log(1/tolerance)/a))))
        while self.tail_bound(t,K)>tolerance:K+=max(1,K//8)
        return K

    def modes(self,t,terms):
        if t<=0 or self.flow.D<=0:raise ValueError('positive time and D for modes')
        f=self.flow;n=np.arange(terms);k=(n+.5)*np.pi/f.half_length
        coeff=(self.left-self.right)*2*f.rho_right/(f.rho_left+f.rho_right)*(-1.)**n/(k*f.half_length)
        return k,coeff*np.exp(-f.D*k*k*t)

    def average(self,faces,t,terms=None):
        f=self.flow;faces=np.asarray(faces,dtype=float)
        if faces.ndim!=1 or len(faces)<2 or np.any(np.diff(faces)<=0) or not np.isfinite(faces).all():
            raise ValueError('finite increasing material faces required')
        if not np.isfinite(t) or t<0:raise ValueError('finite nonnegative time required')
        if t==0 or f.D==0:return self.initial_average(faces)
        K=self.terms_for(t) if terms is None else terms;k,decay=self.modes(t,K)
        a,b=faces[:-1],faces[1:];m=f.mass(faces)
        left=np.sin(k[:,None]*(np.minimum(b,0)+f.half_length))-np.sin(k[:,None]*(np.minimum(a,0)+f.half_length))
        right=np.sin(k[:,None]*(f.half_length-np.maximum(b,0)))-np.sin(k[:,None]*(f.half_length-np.maximum(a,0)))
        means=f.rho_left*(left+right)/(k[:,None]*m)
        return self.mean+decay@means

    def image_log_tail_bound(self,t,images):
        f=self.flow
        if t<=0 or f.D<=0 or images<1:raise ValueError('positive time,D and images required')
        a=f.half_length**2/(f.D*t)
        amplitude=max(abs(self.left-self.mean),abs(self.mean-self.right))
        if amplitude==0:return float('-inf')
        return float(np.log(2*amplitude)-a*images**2-np.log(-np.expm1(-2*a*images)))

    @staticmethod
    def representable_upper_exp(log_bound):
        if log_bound==float('-inf'):return 0.
        value=np.exp(log_bound)
        # Round a positive analytical upper bound upward if it underflows.
        # This changes only bound representation, never a solution value.
        return float(np.nextafter(value,np.inf))

    def image_tail_bound(self,t,images):
        return self.representable_upper_exp(self.image_log_tail_bound(t,images))

    def right_inventory(self,start,t,tolerance=1e-13):
        """Stable finite-domain erfc-image integral over [start,L].

        Independent of the eigenfunction cell-average implementation. No
        small positive inventory is formed by subtracting two O(1) masses.
        """
        f=self.flow;L=f.half_length
        if not 0<=start<=L or not np.isfinite(t) or t<0:raise ValueError('right-region start/time invalid')
        if t==0 or f.D==0:return float(f.rho_right*self.right*(L-start)),0.,0
        images=1
        while self.image_tail_bound(t,images)>tolerance:images+=1
        w=2*np.sqrt(f.D*t)
        def F(x):
            z=x/w
            return w*np.exp(-z*z)*(z*erfcx(z)-1/np.sqrt(np.pi))
        integral=0.
        for n in range(images):
            integral+=(-1.)**n*(F(2*(n+1)*L-start)-F(2*n*L+start))
        inventory=f.rho_right*(self.right*(L-start)+(self.mean-self.right)*integral)
        tail=0. if start==L else self.representable_upper_exp(np.log(f.rho_right*(L-start))+self.image_log_tail_bound(t,images))
        return float(inventory),float(tail),images

    def scalar(self,a,t,terms=None):
        f=self.flow;a=np.asarray(a,dtype=float)
        if t<0:raise ValueError('nonnegative time required')
        if t==0 or f.D==0:return np.where(a<0,self.left,np.where(a>0,self.right,self.mean))
        K=self.terms_for(t) if terms is None else terms;k,decay=self.modes(t,K)
        flat=a.ravel();phase=np.where(flat<=0,flat+f.half_length,f.half_length-flat)
        phi=np.cos(k[:,None]*phase)*np.where(flat<=0,1.,-f.rho_left/f.rho_right)
        result=self.mean+decay@phi
        # Exact interface trace for t>0, not a correction to the solution.
        result[flat==0]=self.mean
        return result.reshape(a.shape)

    def flux(self,a,t,terms=None):
        f=self.flow;a=np.asarray(a,dtype=float)
        if t<=0 or f.D<=0:raise ValueError('positive time and D required for step flux')
        K=self.terms_for(t) if terms is None else terms;k,decay=self.modes(t,K)
        flat=a.ravel();phase=np.where(flat<=0,flat+f.half_length,f.half_length-flat)
        return (f.rho_left*f.D*(decay*k)@np.sin(k[:,None]*phase)).reshape(a.shape)


def right_phase_inventory_bounds(c,m,left_mass,right_mass):
    """Information bounds from cell averages and physical 0<=c<=1.

    This max/min constructs admissible subcell inventories, never clips c.
    Bounds concern the represented state, not continuum error bounds.
    """
    total=m*c
    lower=np.maximum(0.,total-left_mass)
    upper=np.minimum(right_mass,total)
    return float(lower.sum()),float(upper.sum())
