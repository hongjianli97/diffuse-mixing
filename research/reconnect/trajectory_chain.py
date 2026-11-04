"""Small dense supplied-graph research chain; not recovered ltnmix 0.8.0.

No material faces, F/C, physical conductance inference or fluid ORC here.
Caller supplies one symmetric nonnegative conductance per unordered edge.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import shortest_path

ALGEBRA_TOL=1e-10  # Existing handoff's linear-algebra tolerance, not PDE accuracy.
SI_UNITS={'time':'s','positions':'m','mass':'kg','rho':'kg/m^3','g':'kg/s'}


@dataclass(frozen=True)
class TrajectorySamples:
    time: object
    positions: object
    ids: object
    mass: object
    c0: object
    units: object
    rho: object=None
    frame_ids: object=None

    def __post_init__(self):
        if dict(self.units)!=SI_UNITS:raise ValueError('explicit matching SI units required')
        t=np.array(self.time,dtype=float,copy=True);x=np.array(self.positions,dtype=float,copy=True)
        ids=np.array(self.ids,copy=True);m=np.array(self.mass,dtype=float,copy=True);c=np.array(self.c0,dtype=float,copy=True)
        if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('strictly increasing finite time samples required')
        if ids.ndim!=1 or len(ids)<2 or ids.dtype.kind not in 'iuU' or len(np.unique(ids))!=len(ids):raise ValueError('fixed unique integer/string material ids required')
        if ids.dtype.kind=='U' and np.any(ids==''):raise ValueError('empty material identity')
        n=len(ids)
        if x.ndim!=3 or x.shape[:2]!=(len(t),n) or x.shape[2] not in (1,2) or not np.isfinite(x).all():raise ValueError('finite 1D/2D position samples required')
        if any(len(np.unique(frame,axis=0))!=n for frame in x):raise ValueError('duplicate positions in a sample frame')
        if m.shape!=(n,) or not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(m.sum()):raise ValueError('fixed finite positive masses required')
        if c.shape!=(n,) or not np.isfinite(c).all() or np.any(c<0) or np.any(c>1):raise ValueError('initial cell mass fractions in [0,1] required')
        rho=None if self.rho is None else np.array(self.rho,dtype=float,copy=True)
        if rho is not None and (rho.shape!=(len(t),n) or not np.isfinite(rho).all() or np.any(rho<=0)):raise ValueError('optional positive sampled density required')
        frame_ids=None if self.frame_ids is None else np.array(self.frame_ids,copy=True)
        if frame_ids is not None and (frame_ids.shape!=(len(t),n) or not np.all(frame_ids==ids[None,:])):raise ValueError('lost, reordered or changed material identity')
        for name,value in [('time',t),('positions',x),('ids',ids),('mass',m),('c0',c),('rho',rho),('frame_ids',frame_ids)]:
            if value is not None:value.setflags(write=False)
            object.__setattr__(self,name,value)


def edge_array(n,pairs):
    p=np.asarray(pairs)
    if p.ndim!=2 or p.shape[1]!=2 or p.dtype.kind not in 'iu':raise ValueError('integer edge index array [E,2] required')
    if np.any(p<0) or np.any(p>=n) or np.any(p[:,0]>=p[:,1]) or len(np.unique(p,axis=0))!=len(p):raise ValueError('unique unordered i<j edges required')
    return p


def laplacian(n,pairs,g):
    p=edge_array(n,pairs);g=np.asarray(g,dtype=float)
    if g.shape!=(len(p),) or not np.isfinite(g).all() or np.any(g<0):raise ValueError('one raw finite nonnegative conductance per edge required; no clipping')
    L=np.zeros((n,n));i,j=p.T
    np.add.at(L,(i,i),g);np.add.at(L,(j,j),g);L[i,j]=-g;L[j,i]=-g
    return L


def operator_audit(mass,Q):
    m=np.asarray(mass,dtype=float);Q=np.asarray(Q,dtype=float)
    if m.ndim!=1 or np.any(m<=0) or not np.isfinite(m).all() or Q.shape!=(len(m),len(m)) or not np.isfinite(Q).all():raise ValueError('positive mass and finite square generator required')
    off=~np.eye(len(m),dtype=bool);weighted=m[:,None]*Q
    scale=max(1.,float(np.max(abs(Q))));wscale=max(1.,float(np.max(abs(weighted))))
    row=float(np.max(abs(Q.sum(axis=1))));conservation=float(np.max(abs(m@Q)))
    symmetry=float(np.max(abs(weighted-weighted.T)));minimum=float(Q[off].min())
    return {'constant_residual_per_s':row,'mass_residual_kg_per_s':conservation,'weighted_symmetry_residual_kg_per_s':symmetry,
        'minimum_raw_offdiagonal_per_s':minimum,'graph_algebra_eligible':bool(row<=ALGEBRA_TOL*scale and conservation<=ALGEBRA_TOL*wscale and symmetry<=ALGEBRA_TOL*wscale and minimum>=-ALGEBRA_TOL*scale),
        'linear_algebra_tolerance':ALGEBRA_TOL,'physical_acceptance_inferred':False,'clipping_or_renormalization':False}


def reference_distances(data,pairs):
    p=edge_array(len(data.mass),pairs);length=np.linalg.norm(data.positions[0,p[:,1]]-data.positions[0,p[:,0]],axis=1)
    a=np.full((len(data.mass),len(data.mass)),np.inf);np.fill_diagonal(a,0.)
    a[p[:,0],p[:,1]]=length;a[p[:,1],p[:,0]]=length
    # Disconnected components retain infinite distances; no invented bridges.
    return shortest_path(a,directed=False),length


def mixing_metrics(m,c,initial_energy):
    mean=float(c[0]+m@(c-c[0])/m.sum());E=.5*float(m@((c-mean)**2))
    return {'mass_mean':mean,'component_mass_kg':float(m@c),'energy_kg':E,'variance':2*E/float(m.sum()),
        'I':E/initial_energy if initial_energy>0 else None,'H':1-E/initial_energy if initial_energy>0 else None,
        'status':'nonconstant_initial_state' if initial_energy>0 else 'constant_initial_state'}


def propagate_supplied(data,pairs,conductances):
    """One piecewise-constant supplied graph per sample interval, ordered BE.

    This function verifies graph algebra only. Positions/rho are carried as
    provenance; it does not infer physical g from them or claim PDE validity.
    """
    n=len(data.mass);p=edge_array(n,pairs);g=np.asarray(conductances,dtype=float)
    if g.shape!=(len(data.time)-1,len(p)):raise ValueError('one supplied graph per sample interval required')
    Ls=np.array([laplacian(n,p,row) for row in g]);m=data.mass;M=np.diag(m);P=np.eye(n);c=data.c0.copy()
    E0=mixing_metrics(m,c,1.)['energy_kg'];metrics=[mixing_metrics(m,c,E0)];states=[c.copy()];operators=[];balances=[];audits=[]
    for dt,L in zip(np.diff(data.time),Ls):
        audit=operator_audit(m,-L/m[:,None])
        if not audit['graph_algebra_eligible']:raise ValueError('supplied graph algebra failed')
        A=np.linalg.solve(M+dt*L,M);new=A@c;P=A@P
        metric=mixing_metrics(m,new,E0);physical=dt*float(new@L@new);numerical=.5*float(m@((new-c)**2))
        residual=metric['energy_kg']-metrics[-1]['energy_kg']+physical+numerical
        balances.append({'dt_s':float(dt),'dt_chi_endpoint_kg':physical,'backward_euler_extra_dissipation_kg':numerical,'energy_identity_residual_kg':residual,
            'linear_solve_residual_kg':float(np.max(abs((M+dt*L)@new-M@c)))})
        if abs(residual)>ALGEBRA_TOL*max(1.,E0) or new.min()<data.c0.min()-ALGEBRA_TOL or new.max()>data.c0.max()+ALGEBRA_TOL:raise RuntimeError('graph propagation invariant failed')
        states.append(new.copy());operators.append(A);metrics.append(metric);audits.append(audit);c=new
    raw={'row_sum_residual':float(np.max(abs(P.sum(axis=1)-1))),'mass_invariance_residual_kg':float(np.max(abs(m@P-m))),
        'minimum_raw_P':float(P.min()),'P_c0_vs_direct_max_difference':float(np.max(abs(P@data.c0-c))),
        'fluid_orc_enabled':False,'physical_acceptance':False}
    if max(raw['row_sum_residual'],raw['mass_invariance_residual_kg']/max(1.,float(m.max())),raw['P_c0_vs_direct_max_difference'])>ALGEBRA_TOL or raw['minimum_raw_P']<-ALGEBRA_TOL:raise RuntimeError('raw propagation kernel invariant failed')
    return {'states':np.array(states),'P':P,'steps':np.array(operators),'laplacians':Ls,'metrics':metrics,'energy_balances':balances,'operator_audits':audits,'kernel_audit':raw,
        'forward_row_mass_kernel':P.T*m[None,:]/m[:,None],
        'forward_column_component_mass_map':m[:,None]*P/m[None,:]}
