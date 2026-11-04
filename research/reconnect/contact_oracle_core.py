"""One static point-FV contact: finite-cell references and fixed positive controls."""
from decimal import Decimal as Dec,localcontext
from fractions import Fraction as F
import math
import numpy as np
from contact_spectral_reference import TrigPi,dec,material_factors,tail_certificate,render
from cartesian_fv_modes import MATERIAL_PRIOR,build_modes,coefficients,reconstruct,propagate,direct_structured_action
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator

D=F(float(.02));TIMES=(F(0),F(1,40),F(1,20));K=16


def pure_contact(labels,ids):
    a=np.asarray(labels);ident=np.asarray(ids);N=math.isqrt(len(ident))
    axis=(np.arange(N)+.5)/N;expected=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    if N<4 or N%2 or not np.array_equal(a,expected) or not np.array_equal(ident,np.arange(N*N)):raise ValueError('complete original even Cartesian labels/ID order required; no repair')
    return {'ids':ident.copy(),'positions':a.copy(),'rho':np.where(a[:,0]<.5,1.,4.)}


def face_control(packet,pure,N):
    points=[tuple(F(float(x)) for x in p) for p in pure['positions']];rho=[F(float(x)) for x in pure['rho']]
    assert packet['ids']==list(map(int,pure['ids'])) and F(packet['D_exact'])==D
    assert len(packet['edges'])==2*N*N and set(map(F,packet['cell_areas_exact']))=={F(1,N*N)}
    counts={'low':0,'high':0,'interface':0};keys=set()
    for (i,j),(sx,sy),g,ratio in zip(packet['edges'],packet['image_shifts'],packet['conductances_exact'],packet['face_over_distance_exact']):
        assert (i,j,sx,sy) not in keys;keys.add((i,j,sx,sy))
        dx=points[j][0]+sx-points[i][0];dy=points[j][1]+sy-points[i][1]
        assert (abs(dx),abs(dy)) in ((F(1,N),F(0)),(F(0),F(1,N))) and F(ratio)==1
        expected=1/(1/(2*rho[i]*D)+1/(2*rho[j]*D));assert F(g)==expected and expected>0
        if rho[i]!=rho[j]:assert dy==0 and expected==F(8,5)*D;counts['interface']+=1
        else:counts['low' if rho[i]==1 else 'high']+=1
    assert counts=={'low':N*N-N,'high':N*N-N,'interface':2*N}
    return {'all_world_axis_faces_and_two_half_resistances_exact':True,'face_counts':counts,
      'g_low_exact':str(D),'g_high_exact':str(4*D),'g_interface_exact':str(F(8,5)*D),'all_cell_areas_exact':True,
      'internal_material_faces_or_precomputed_g_supplied_to_builder':False}


def derive_material(case,pure,N):
    """Evaluate saved coefficients only; no spectral projection or PDE solve."""
    if case['K']!=128 or case['requested_decimal_digits']!=90:raise ValueError('registered128mode90digit source required')
    rho=pure['rho'].copy();m=rho/(N*N);outputs={};details={}
    for precision in (100,120):
        with localcontext() as ctx:
            ctx.prec=precision;trig=TrigPi();alpha=list(map(Dec,case['alpha']));beta=list(map(Dec,case['beta']));refs=[];rounding=[];tails=[]
            for t in TIMES:
                normal,y,initial=material_factors(trig,alpha,beta,N,dec(D),dec(t))
                v0=[Dec('.5')+x*z for z in y for x in initial]
                values=v0 if t==0 else [Dec('.5')+x*z for z in y for x in normal]
                refs.append(np.array(list(map(float,values))))
                rounding.append(float(sum((dec(a)*(dec(float(v))-v)**2 for a,v in zip(m,values)),Dec(0)).sqrt()))
                tails.append(None if t==0 else tail_certificate(128,D,t))
            outputs[precision]=np.array(refs)
            details[precision]={'precision':precision,'initial_normal_cell_means':render(initial,90),'transverse_cell_means':render(y,90),
              'float_serialization_mass_norms':rounding,'initial_serialization_bridge_absolute':rounding[0],'tail_certificates_by_time':tails}
    c0=outputs[120][0].copy();R=64*np.finfo(float).eps*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))
    diffs=[float(np.sqrt(m@((a-b)**2))) for a,b in zip(outputs[100],outputs[120])]
    if max(diffs)>R:raise ArithmeticError('original fixed reference evaluation guard failed')
    return {'mass':m,'rho':rho,'ids':pure['ids'].copy(),'c0':c0,'reference':outputs[120]}, {
      **details[120],'Decimal100_120_mass_norm_differences':diffs,'reference_evaluation_indicator_absolute':float(R),
      'mass_and_initial_means_newly_derived_not_recovered_v080':True,'stored_spectral_coefficients_reused_no_projection_fit':True,
      'reference_initial_direct_formula_not_truncated_spectral_sum':True,'old_T005_tail_not_used_for_T0025':True,
      'rounding_or_initial_bridge_subtracted_from_state':False,'reference_controls_not_interval_certification':True}


def cartesian_control(data,N):
    m,c0=data['mass'],data['c0'];norm=lambda v:float(np.sqrt(m@(v*v)))
    modes=build_modes(data['rho'].reshape(N,N),m.reshape(N,N),float(D),model_prior=MATERIAL_PRIOR)
    coeff=coefficients(modes,c0.reshape(N,N));initial,imag0=reconstruct(modes,coeff)
    action,imag_action=direct_structured_action(modes,c0.reshape(N,N));states=[];rows=[]
    qnorm=math.sqrt(1+modes.orthogonality_residual_frobenius);initial_norm=norm(c0)
    with localcontext() as ctx:
        ctx.prec=90;trig=TrigPi()
        yerror=float(max(abs(dec(float(modes.transverse_eigenvalues[j]))-4*dec(D)*N*N*trig.sc(F(j,N))[0]**2) for j in range(N)))
    residual=modes.eigen_residual_frobenius+float(modes.normal_matrix_rounding_row_bound_exact)*qnorm
    roundtrip=norm(initial.ravel()-c0)
    for i in range(K+1):
        t=F(i,20*K);T=float(t);state,imag=propagate(modes,coeff,T);states.append(state.ravel())
        growth=math.exp(T*max(0.,-float(modes.normal_eigenvalues.min())))
        parts={'spectral_residual_and_matrix_rounding':(modes.orthogonality_residual_frobenius+T*growth*(residual*qnorm+yerror*qnorm*qnorm))*initial_norm,
          'fixed_machine_transform_allowance':32*N*np.finfo(float).eps*growth*(qnorm*qnorm+1)*initial_norm,
          'initial_roundtrip':roundtrip,'imaginary_mass_norm':norm(imag.ravel()),'time_label_rounding':float(abs(F(T)-t))*norm(action.ravel())}
        rows.append({'time_exact':str(t),'components_absolute':parts,'total_absolute':sum(parts.values())})
    return {'states':np.array(states),'normal_matrix':modes.normal_matrix,'normal_eigenvalues':modes.normal_eigenvalues,
      'normal_eigenvectors':modes.normal_eigenvectors,'transverse_eigenvalues':modes.transverse_eigenvalues,'initial_full_coefficients':coeff,
      'initial_structured_rate':action.ravel()}, {'N':N,'all_discrete_mode_slots_retained':N*N,'normal_g_exact':list(map(str,modes.normal_g_exact)),
      'raw_minimum_normal_eigenvalue':float(modes.normal_eigenvalues.min()),'normal_eigen_residual_frobenius':modes.eigen_residual_frobenius,
      'normal_orthogonality_residual_frobenius':modes.orthogonality_residual_frobenius,'normal_matrix_rounding_row_bound_exact':str(modes.normal_matrix_rounding_row_bound_exact),
      'transverse_eigenvalue_vs90digit_max_difference':yerror,'rows':rows,'clipping_normalization_mode_filter_or_zero_eigenvalue_reset':False}


def evaluate(data,reference_meta,run,path,control,control_meta):
    m,c0=data['mass'],data['c0'];norm=lambda v:float(np.sqrt(m@(v*v)));H=lambda v:metrics(m,v,c0)['H']
    sigma=math.sqrt(metrics(m,c0,c0)['variance']);R=reference_meta['reference_evaluation_indicator_absolute'];bridge=reference_meta['initial_serialization_bridge_absolute']
    rows=[];observations=[]
    for i,t in enumerate([F(i,20*K) for i in range(K+1)]):
        c=path['states'][i];other=path['independent_states'][i];cross=control['states'][i]
        machine=0. if i==0 else run['steps'][i-1]['cumulative_machine_indicator_absolute'];tail=0. if i==0 else run['steps'][i-1]['cumulative_tail_indicator_absolute']
        observed=norm(c-other);parts={'machine':machine,'raw_tail':tail,'observed_full_sequence':observed,
          'Cartesian_semidiscrete_indicator':control_meta['rows'][i]['total_absolute'],'twice_fixed_evaluation':2*R,'initial_serialization_bridge':bridge}
        budget=sum(parts.values());delta=norm(c-cross);hg=abs(H(c)-H(cross));hb=h_indicator(m,c0,budget)
        rows.append({'time':float(t),'components_absolute':parts,'actual_vs_Cartesian_absolute':delta,'fixed_indicator_absolute':budget,
          'within_indicator':delta<=budget,'H_difference':hg,'fixed_H_indicator':hb,'H_within_indicator':hg<=hb,
          'full_sequence_within_original_allowance':observed<=machine+tail})
        if i in (0,8,16):
            j=i//8;ref=data['reference'][j];refmeta=reference_meta['tail_certificates_by_time'][j]
            trunc=0. if refmeta is None else float(F(refmeta['material_mass_norm_tail_bound_exact']))
            refnum=R+reference_meta['float_serialization_mass_norms'][j]+trunc+bridge
            spatial=norm(cross-ref);total=spatial+budget+refnum;error=norm(c-ref)
            observations.append({'time':float(t),'reference_error_relative':error/sigma,'semidiscrete_PDE_error_relative':spatial/sigma,
              'actual_H':H(c),'reference_H':H(ref),'semidiscrete_H':H(cross),'raw_mass_change':float(m@(c-c0)),
              'minimum':float(c.min()),'maximum':float(c.max()),'actual_vs_Cartesian_relative':delta/sigma,
              'fixed_action_indicator_relative':budget/sigma,'reference_numeric_indicator_absolute':refnum,'total_PDE_comparison_indicator_relative':total/sigma,
              'reference_decomposition_passed':error<=total,'reference_H_decomposition_passed':abs(H(c)-H(ref))<=h_indicator(m,c0,total),
              'PDE_spatial_error_subtracted_or_fitted':False})
    action=all(v['within_indicator'] and v['H_within_indicator'] and v['full_sequence_within_original_allowance'] for v in rows)
    refs=all(v['reference_decomposition_passed'] and v['reference_H_decomposition_passed'] for v in observations)
    passed=action and refs
    return {'initial_mass_fluctuation_norm':sigma,'observations':observations,'all_endpoint_action_controls':rows,
      'all_action_and_original_H_controls_passed':action,'all_reference_decompositions_passed':refs,
      'registered_same_backend_contact_oracle_control_passed':passed,
      'comparison_decision':'T4_same_backend_static_contact_oracle_control_passed' if passed else 'T4_static_contact_oracle_control_failed_or_unresolved',
      'one_N_not_spatial_convergence':True,'static_L_midpoint_time_error_zero_not_measured_order':True,'indicators_are_certified_total_error_bounds':False,
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
