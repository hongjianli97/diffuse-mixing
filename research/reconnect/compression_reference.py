"""Independent time-dependent material-PDE reference, not a point backend.

c=.5+.2*A(a,t)*cos(2*pi*b), B=exp(D*(2*pi)^2*t)*A.
Fixed material-column mass means obey M B_t=-L(t) B. Face coefficients
are analytic rho0/J^2, unrelated to the candidate kernel or point weights.
Midpoint Crank--Nicolson preserves the raw inventory and dissipates its
discrete quadratic energy. It is not claimed to be a positive propagator.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu


def material_columns(M,epsilon):
    if not isinstance(M,(int,np.integer)) or M<4 or not np.isfinite(epsilon) or abs(epsilon)>=1:
        raise ValueError('integer M>=4 and |epsilon|<1 required')
    a=(np.arange(M)+.5)/M;k=2*np.pi;s=np.sinc(1/M)
    C=s*np.cos(k*a);C2=(1+np.sinc(2/M)*np.cos(2*k*a))/2
    rho_average=1+epsilon*C
    return a,rho_average/M,(C+epsilon*C2)/rho_average


def material_laplacian(M,epsilon,alpha,D):
    if not np.isfinite(alpha) or abs(alpha)>=1 or not np.isfinite(D) or D<0:
        raise ValueError('orientation-preserving prescribed map and nonnegative D required')
    material_columns(M,epsilon)
    face=(np.arange(M)+1)/M;cosine=np.cos(2*np.pi*face)
    g=D*M*(1+epsilon*cosine)/(1+alpha*cosine)**2
    i=np.arange(M);j=(i+1)%M
    return coo_matrix((np.r_[g,g,-g,-g],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(M,M)).tocsc()


def project_to_saved_cells(B,fine_mass,time,D,N,saved_mass):
    """Raw component integrals over nested columns, divided by saved masses.

    Constant inventory is also integrated, retaining parent/child mass rounding;
    no ratio is silently set to one and no reference/candidate mass is repaired.
    """
    B=np.asarray(B,dtype=float);mf=np.asarray(fine_mass,dtype=float);m=np.asarray(saved_mass,dtype=float)
    if B.ndim!=1 or mf.shape!=B.shape or not isinstance(N,(int,np.integer)) or N<2 or len(B)%N or m.shape!=(N*N,) or np.any(mf<=0) or np.any(m<=0) or not all(np.isfinite(x).all() for x in [B,mf,m]):
        raise ValueError('finite nested columns and saved positive 2D masses required')
    if not np.isfinite(time) or time<0 or not np.isfinite(D) or D<0:raise ValueError('nonnegative finite time and D required')
    rows=m.reshape(N,N)
    if not np.array_equal(rows,np.broadcast_to(rows[0],rows.shape)):raise ValueError('reference requires density depending on material x only')
    child_mass=mf.reshape(N,-1).sum(axis=1);component=(mf*B).reshape(N,-1).sum(axis=1)
    y=(np.arange(N)+.5)/N;Y=np.sinc(1/N)*np.cos(2*np.pi*y)
    numerator=.5*np.broadcast_to(child_mass[None,:],(N,N))/N
    numerator=numerator+.2*np.exp(-D*(2*np.pi)**2*time)*np.outer(Y,component)/N
    return (numerator/rows).ravel(),child_mass


def reference_path(M,steps,N,saved_mass,epsilon=.25,D=.02,alpha_end=.25,T=.05):
    if not isinstance(steps,(int,np.integer)) or steps<2 or steps%2 or not np.isfinite(T) or T<=0:
        raise ValueError('positive duration and even number of at least two steps required')
    material_laplacian(M,epsilon,alpha_end,D)
    axis,mass,initial=material_columns(M,epsilon);B=initial.copy();Mdiag=diags(mass,format='csc')
    times=np.linspace(0,T,steps+1);dt=T/steps;states=[];snapshots=[];snapshot_times=[]
    inventory=[float(mass@B)];energy=[float(.5*mass@(B*B))];balances=[];minimum_rhs_diagonal=np.inf
    for step in range(steps+1):
        if step in [0,steps//2,steps]:
            state,child_mass=project_to_saved_cells(B,mass,times[step],D,N,saved_mass)
            states.append(state);snapshots.append(B.copy());snapshot_times.append(times[step])
        if step==steps:break
        alpha=alpha_end*(step+.5)/steps;L=material_laplacian(M,epsilon,alpha,D)
        rhs=Mdiag-.5*dt*L;minimum_rhs_diagonal=min(minimum_rhs_diagonal,float(rhs.diagonal().min()))
        following=splu(Mdiag+.5*dt*L).solve(rhs@B)
        average=(B+following)/2;new_energy=float(.5*mass@(following*following))
        balances.append(new_energy-energy[-1]+dt*float(average@(L@average)))
        B=following;inventory.append(float(mass@B));energy.append(new_energy)
    snapshots=np.array(snapshots);snapshot_times=np.array(snapshot_times)
    arrays={'axis':axis,'mass_1d':mass,'initial_B':initial,'step_time':times,'snapshot_time':snapshot_times,
        'B_snapshots':snapshots,'A_snapshots':snapshots*np.exp(-D*(2*np.pi)**2*snapshot_times[:,None]),
        'projected_states':np.array(states),'child_mass_1d':child_mass,'B_inventory':np.array(inventory),
        'B_energy':np.array(energy),'CN_energy_balance_residual':np.array(balances)}
    audit={'reference_columns':M,'time_steps':steps,'dt_s':dt,
        'B_weighted_inventory_max_change':float(np.max(abs(arrays['B_inventory']-inventory[0]))),
        'B_energy_max_increase':float(np.max(np.diff(energy))),
        'CN_energy_balance_max_abs_residual':float(np.max(np.abs(balances))),
        'minimum_raw_CN_rhs_diagonal_kg':minimum_rhs_diagonal,
        'saved_parent_vs_child_mass_max_relative_difference':float(np.max(abs(child_mass/(np.asarray(saved_mass)[:N]*N)-1))),
        'raw_projected_state_range':[float(arrays['projected_states'].min()),float(arrays['projected_states'].max())],
        'reference_is_a_positive_candidate_backend':False,'clipping_renormalization_or_fitting':False}
    return arrays,audit
