"""Isolated nonuniform material-mean reference; no candidate-backend dependency."""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
from scipy.linalg import solve_banded


def readonly(values):
    a=np.asarray(values,dtype=float).copy();a.flags.writeable=False;return a


@dataclass(frozen=True)
class Mesh:
    parent_edges: np.ndarray
    counts: tuple
    edges: np.ndarray
    mass: np.ndarray
    centres: np.ndarray
    starts: np.ndarray

    @property
    def parent_mass(self):return np.diff(self.parent_edges)


def build_mesh(parent_edges,counts,level=0):
    p=np.asarray(parent_edges,dtype=float)
    if p.ndim!=1 or len(p)<2 or not np.isfinite(p).all() or np.any(np.diff(p)<=0):raise ValueError('strict finite parent edges required')
    if isinstance(level,(bool,np.bool_)) or not isinstance(level,(int,np.integer)) or level<0:raise ValueError('nonnegative integer level required')
    counts=tuple(counts)
    if len(counts)!=len(p)-1 or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<=0 for n in counts):raise ValueError('one positive integer count per parent required')
    n=tuple(int(v)*2**int(level) for v in counts);e=[]
    for a,b,N in zip(p[:-1],p[1:],n):
        a,b=F(float(a)),F(float(b));e.extend(float(a+(b-a)*j/N) for j in range(N))
    e.append(float(p[-1]));e=np.array(e);mass=np.diff(e);centres=e[:-1]+mass/2
    if not np.isfinite(e).all() or np.any(mass<=0) or np.any(np.diff(centres)<=0):raise ValueError('subdivision not representable without moving boundaries')
    starts=np.r_[0,np.cumsum(n)].astype(np.int64);starts.flags.writeable=False
    if not np.array_equal(e[starts],p):raise ArithmeticError('original parent boundary changed')
    return Mesh(readonly(p),n,readonly(e),readonly(mass),readonly(centres),starts)


def observe(mesh,values,parent_mass=None):
    u=np.asarray(values,dtype=float);m=mesh.parent_mass if parent_mass is None else np.asarray(parent_mass,dtype=float)
    if u.shape!=mesh.mass.shape or not np.isfinite(u).all():raise ValueError('complete finite child means required')
    if m.shape!=mesh.parent_mass.shape or not np.array_equal(m,mesh.parent_mass):raise ValueError('original parent masses must match stored parent edges; no reweighting')
    return np.add.reduceat(mesh.mass*u,mesh.starts[:-1])/m


def conductances(mesh,rho_mu,t,D):
    rho=np.asarray(rho_mu(mesh.edges[1:-1],t),dtype=float)
    if rho.ndim==0:rho=np.full(len(mesh.mass)-1,float(rho))
    if rho.shape!=(len(mesh.mass)-1,) or not np.isfinite(rho).all() or np.any(rho<=0):raise ValueError('positive finite density on every internal face required')
    g=rho**2*D/np.diff(mesh.centres)
    if not np.isfinite(g).all() or (D>0 and np.any(g<=0)):raise ArithmeticError('nonpositive/overflowed positive internal conductance; no repair')
    return g


class StepRestrictionError(ArithmeticError):
    def __init__(self,step,margin,diagnostics):
        super().__init__('CN sufficient positivity condition failed before solve');self.step=step;self.margin=margin.copy();self.diagnostics=diagnostics


class RawStateError(ArithmeticError):
    def __init__(self,step,raw_state,diagnostics):
        super().__init__('raw reference state failed original finite/mass/extrema guards');self.step=step;self.raw_state=raw_state.copy();self.diagnostics=diagnostics


def solve(mesh,c0,rho_mu,steps,T=.1,D=.02,positivity='require',history=False):
    if isinstance(steps,(bool,np.bool_)) or not isinstance(steps,(int,np.integer)) or steps<1:raise ValueError('positive integer steps required')
    if not math.isfinite(T) or T<=0 or not math.isfinite(D) or D<0:raise ValueError('positive duration and nonnegative finite D required')
    if positivity not in ('require','monitor'):raise ValueError('explicit require/monitor positivity policy required')
    initial=np.asarray(c0,dtype=float)
    if initial.shape!=mesh.mass.shape or not np.isfinite(initial).all() or initial.min()<0 or initial.max()>1:raise ValueError('finite original material mass fractions in[0,1] required')
    initial=initial.copy();c=initial.copy();m=mesh.mass;dt=T/steps;mass0=float(m@initial);lo=float(c.min());hi=float(c.max());band=np.zeros((3,len(m)))
    diag={'positivity_policy':positivity,'linear_solves':0,'coefficient_evaluations':0,'steps_completed':0,'steps_without_sufficient_positivity':0,
      'max_relative_mass_drift':0.,'min_all_times':lo,'max_all_times':hi,'initial_min':lo,'initial_max':hi,'states_clipped_or_normalized':False,
      'D0_exact_identity':D==0,'no_internal_face_identity':len(m)==1,'minimum_RHS_diagonal':None}
    states=[c.copy()] if history else [];actions=[]
    for k in range(steps):
        if D==0 or len(m)==1:
            diag['steps_completed']+=1
            if history:states.append(c.copy())
            continue
        t=(k+.5)*dt;g=conductances(mesh,rho_mu,t,D);diag['coefficient_evaluations']+=1
        half=.5*dt*g;rhsdiag=m.copy();rhsdiag[:-1]-=half;rhsdiag[1:]-=half
        minimum=float(rhsdiag.min());diag['minimum_RHS_diagonal']=minimum if diag['minimum_RHS_diagonal'] is None else min(diag['minimum_RHS_diagonal'],minimum)
        if minimum<0:
            diag['steps_without_sufficient_positivity']+=1
            if positivity=='require':raise StepRestrictionError(k,rhsdiag,diag.copy())
        diagonal=m.copy();diagonal[:-1]+=half;diagonal[1:]+=half
        rhs=m*c;jump=half*np.diff(c);rhs[:-1]+=jump;rhs[1:]-=jump
        band[1]=diagonal;band[0,1:]=-half;band[2,:-1]=-half
        c=solve_banded((1,1),band,rhs,check_finite=False);diag['linear_solves']+=1
        drift=abs(float(m@c)-mass0)/abs(mass0) if mass0 else abs(float(m@c))
        diag['max_relative_mass_drift']=max(diag['max_relative_mass_drift'],drift)
        diag['min_all_times']=min(diag['min_all_times'],float(c.min()));diag['max_all_times']=max(diag['max_all_times'],float(c.max()))
        if not np.isfinite(c).all() or drift>1e-9 or c.min()<lo-1e-10 or c.max()>hi+1e-10:raise RawStateError(k,c,diag.copy())
        diag['steps_completed']+=1
        if history:states.append(c.copy());actions.append({'step':k,'time':t,'g':g.copy(),'RHS_diagonal':rhsdiag.copy()})
    return c,diag,{'states':np.array(states),'actions':actions}
