"""Pure given-trajectory sampling and generic chronological action; no PDE solve."""
from fractions import Fraction as F
import math
import numpy as np
from shear_probe import shear_map
from voronoi_time_action import exponential,uniformization,arithmetic_allowance,metrics,dissipation


def sample_points(labels,ids,rho0,times):
    a=np.asarray(labels,dtype=float);ident=np.asarray(ids);r=np.asarray(rho0,dtype=float);t=np.asarray(times,dtype=float)
    n=len(ident)
    if ident.ndim!=1 or ident.dtype.kind not in 'iu' or len(np.unique(ident))!=n or n<2:
        raise ValueError('complete unique integer identities required')
    if a.shape!=(n,2) or r.shape!=(n,) or not np.isfinite(a).all() or not np.isfinite(r).all() or np.any(r<=0):
        raise ValueError('original finite labels and positive point densities required')
    if np.any(a<0) or np.any(a>=1) or t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or t[0]<0 or t[-1]>.05:
        raise ValueError('original periodic labels and increasing times in[0,.05] required')
    s=.25*t/.05
    x,y=shear_map(a[None,:,0],a[None,:,1],s[:,None],t[:,None])
    positions=np.stack([x,np.broadcast_to(y,x.shape)],axis=-1)%1.
    return {'ids':ident.copy(),'time':t.copy(),'positions':positions,
            'rho':np.broadcast_to(r,(len(t),n)).copy()}


def ordered_path(mass,c0,load_action,intervals=16,persist=lambda *args:None,guard=lambda *args:None):
    if not isinstance(intervals,int) or intervals<2 or intervals%2:
        raise ValueError('positive even interval count required')
    m=np.asarray(mass,dtype=float);c0=np.asarray(c0,dtype=float)
    sigma=math.sqrt(metrics(m,c0,c0)['variance']);dt=F(1,20*intervals)
    if not c0.min()>=0 or not c0.max()<=1:raise ValueError('original initial mass fraction must be in[0,1]')
    state=c0.copy();control=c0.copy();states=[state.copy()];controls=[control.copy()];halves=[];rows=[]
    machine_sum=tail_sum=0.;M=float(m.sum());upper=float(np.max(abs(c0)))
    for step in range(intervals):
        op=load_action(step)
        if not np.array_equal(op.mass,m):raise ValueError('original actual masses changed between steps')
        diag=op.diagnostics
        if diag['uniformization_minimum_stored_coefficient']<0 or diag['uniformization_row_sum_residual_Linf']>=1e-10 or diag['uniformization_invariant_mass_residual_Linf']/float(m.max())>=1e-10:
            raise ArithmeticError('unchanged raw operator checks failed; no repair')
        after,half=exponential(op,state,dt)
        other,weights,info=uniformization(op,control,dt)
        allowance=arithmetic_allowance(op,c0,dt,info['M']);machine_sum+=allowance
        tail_sum+=float(F(info['tail_upper_exact']))*math.sqrt(M)*upper
        delta=float(np.sqrt(m@((after-other)**2)))
        if not np.isfinite(after).all() or not np.isfinite(other).all() or delta>machine_sum+tail_sum:
            raise ArithmeticError('independent whole-sequence action control failed; no correction')
        before=metrics(m,state,c0);end=metrics(m,after,c0);drift=float(m@(after-c0))
        if abs(drift)/float(m@c0)>1e-10 or after.min()<c0.min()-1e-10 or after.max()>c0.max()+1e-10 or end['variance']>before['variance']+1e-10*sigma*sigma:
            raise ArithmeticError('raw mass/extrema/variance check failed; no correction')
        chi=[dissipation(op,c) for c in (state,half,after)]
        if min(chi)<0:raise ArithmeticError('negative raw shared-face dissipation')
        loss=.5*(before['variance']-end['variance']);simpson=float(dt)*(chi[0]+4*chi[1]+chi[2])/6
        row={'step':step,'dt_exact':str(dt),'operator':diag,'poisson':info,
            'state_metrics':end,'independent_metrics':metrics(m,other,c0),
            'primary_control_mass_norm_difference':delta,'primary_control_relative_difference':delta/sigma,
            'machine_indicator_absolute_this_interval':allowance,'cumulative_machine_indicator_absolute':machine_sum,
            'cumulative_tail_indicator_absolute':tail_sum,'raw_mass_change_from_c0':drift,
            'independent_raw_mass_change_from_c0':float(m@(other-c0)),
            'minimum':float(after.min()),'maximum':float(after.max()),'chi_start_half_end':chi,
            'half_variance_loss':loss,'Simpson_chi_integral':simpson,'raw_energy_balance_difference':loss-simpson,
            'Simpson_is_empirical_not_exact_balance':True}
        persist(step,op,weights)
        state=after;control=other;states.append(state.copy());controls.append(control.copy());halves.append(half);rows.append(row)
        guard('propagated',step+1,intervals)
    return {'steps':rows,'initial_mass_fluctuation_norm':sigma}, {'states':np.array(states),
        'independent_states':np.array(controls),'exponential_half_states':np.array(halves)}
