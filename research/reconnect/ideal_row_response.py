"""One aggregate ideal material-mean row, with no individual edge weights.

The variance formula and direct subcell integration use the unchanged raw
kernel. Exact kernel-moment predictions are kept separate from raw quadrature.
"""
import numpy as np
from contact_continuous import gauss,check_h,transverse_moments


def normal_panels(ell,h):
    check_h(h)
    if not np.isscalar(ell) or not np.isfinite(ell) or ell<=0:raise ValueError('positive finite cell side required')
    last=int(np.floor(2*h/ell))
    cuts=sorted(set([-2*h,2*h]+[k*ell for k in range(-last,last+1) if -2*h<k*ell<2*h]))
    return np.asarray(cuts,dtype=float)


def direct_material_phase_means(s,ell,center):
    """Integrate actual projected neighbor scalars, independently of theta.

    Two-point Gauss on each geometric phase interval is exact here. Neighbor
    scalars are true quadratic cell means, including their ell**2/24 offset.
    This phase integration does not create pair weights or a second row.
    """
    if any(not np.isscalar(v) or not np.isfinite(v) for v in [s,ell,center]) or ell<=0:
        raise ValueError('finite displacement/center and positive side required')
    nodes,weights=gauss(2);left=-ell/2;right=ell/2
    low=int(np.floor((left+s)/ell-.5))-1;high=int(np.ceil((right+s)/ell-.5))+1
    cuts=[left,right]
    for k in range(low,high+1):
        crossing=(k+.5)*ell-s
        if left<crossing<right:cuts.append(float(crossing))
    cuts=sorted(set(cuts));mean_jump=0.;mean_delta=0.;target=center**2/2+ell**2/24
    for a,b in zip(cuts[:-1],cuts[1:]):
        phase=(a+b)/2+(b-a)/2*nodes;w=(b-a)/2*weights
        neighbor_index=np.floor((phase+s)/ell+.5)
        neighbor_center=center+ell*neighbor_index
        neighbor_mean=neighbor_center**2/2+ell**2/24
        mean_jump+=float(w@(neighbor_center-center))/ell
        mean_delta+=float(w@(neighbor_mean-target))/ell
    return mean_jump,mean_delta


def aggregate_row(ell,h,D,rho,mass,center,normal_order,transverse_order):
    panels=normal_panels(ell,h);nodes,weights=gauss(normal_order);gauss(transverse_order)
    if any(not np.isscalar(v) or not np.isfinite(v) for v in [D,rho,mass,center]) or D<0 or rho<=0 or mass<=0:
        raise ValueError('finite D>=0, positive rho/original mass and finite center required')
    ss=[];ww=[];pp=[]
    for index,(left,right) in enumerate(zip(panels[:-1],panels[1:])):
        ss.extend((left+right)/2+(right-left)/2*nodes)
        ww.extend((right-left)/2*weights);pp.extend([index]*normal_order)
    s=np.asarray(ss);w=np.asarray(ww);panel=np.asarray(pp,dtype=int)
    b0,_=transverse_moments(s,h,transverse_order)
    k=np.floor(s/ell);theta=s/ell-k
    if np.any(theta<0) or np.any(theta>=1):raise RuntimeError('invalid fractional phase; never clip')
    eta_integrand=ell**2*b0*theta*(1-theta)
    direct=np.array([direct_material_phase_means(float(v),ell,center) for v in s])
    expected_delta=center*s+(s*s+ell**2*theta*(1-theta))/2
    area=ell**2;factor=rho*D*area/mass
    eta=float(w@eta_integrand);M0=float(w@b0);M2=float(w@(b0*s*s))
    direct_numerator=2*rho*D*area*float(w@(b0*direct[:,1]))
    raw_variance_numerator=rho*D*area*(M2+eta)
    raw_variance=raw_variance_numerator/mass;raw_direct=direct_numerator/mass
    exact_moment_prediction=factor*(1+eta)
    panel_eta=np.asarray([float(w[panel==j]@eta_integrand[panel==j]) for j in range(len(panels)-1)])
    arrays={'normal_edges':panels,'s':s,'normal_weights':w,'panel':panel,'kernel_transverse_b0':b0,
        'cell_shift_integer':k,'fractional_phase':theta,'eta_integrand':eta_integrand,'panel_eta':panel_eta,
        'direct_phase_mean_jump':direct[:,0],'direct_phase_mean_quadratic_delta':direct[:,1],
        'phase_variance_identity_residual':direct[:,1]-expected_delta}
    metrics={'quadrature':{'normal_order':normal_order,'transverse_order':transverse_order},'normal_panels':len(panels)-1,'normal_nodes':len(s),
        'eta':eta,'effective_rate_exact_moment_prediction_per_s':exact_moment_prediction,
        'raw_variance_aggregate_rate_per_s':raw_variance,'raw_direct_aggregate_rate_per_s':raw_direct,
        'raw_variance_mass_rate_numerator_kg_per_s':raw_variance_numerator,'raw_direct_mass_rate_numerator_kg_per_s':direct_numerator,
        'raw_second_moment':M2,'raw_second_moment_minus_exact_one':M2-1,
        'raw_zero_moment_m_inv2':M0,'raw_zero_moment_relative_difference_from_exact':M0/(9/(2*h*h))-1,
        'raw_variance_minus_exact_moment_prediction_per_s':raw_variance-exact_moment_prediction,
        'raw_direct_minus_raw_variance_per_s':raw_direct-raw_variance,
        'direct_mean_jump_max_error_m':float(np.max(abs(direct[:,0]-s))),
        'direct_quadratic_phase_identity_max_error':float(np.max(abs(direct[:,1]-expected_delta))),
        'raw_integrated_mean_jump_per_m':float(w@(b0*direct[:,0])),
        'panel_eta_sum_minus_global':float(panel_eta.sum()-eta),
        'original_mass_denominator_kg':mass,'analytic_quadratic_target_mass_average':center**2/2+ell**2/24,
        'edge_weights_created':0,'raw_kernel_or_response_renormalized':False}
    return arrays,metrics
