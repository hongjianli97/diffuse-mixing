"""One registered K32 wrapper and unchanged-rule mixed-time observations."""
from pathlib import Path
from fractions import Fraction as F
import ast,inspect,math
import numpy as np
from audit_route_decision import read
from becker_point_core import observe_columns,T,sample_frames
from becker_spatial_core import common_observe,exact_common
from aligned_becker_check import arrays,norm
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
from shear_union64_study import classify


def sample_frames_k32(mu_edges,flow,rows=8):
    e=np.asarray(mu_edges);centres=(e[:-1]+e[1:])/2;cols=len(centres);y=(np.arange(rows)+.5)/rows;cases={}
    for K in (32,):
        frames=[]
        for i in range(2*K+1):
            t=T*i/(2*K);xx=flow.x_mu(centres,float(t));rho=flow.rho_mu(centres,float(t));bounds=flow.x_mu(e[[0,-1]],float(t))
            frames.append({'time_exact':str(t),'positions':np.stack(np.meshgrid(xx,y,indexing='xy'),axis=-1).reshape(-1,2).tolist(),
              'rho':np.tile(rho,rows).tolist(),'x_bounds':bounds.tolist(),'period_y':1.,'D':.02})
        cases[str(K)]={'K':K,'frames':frames}
    return {'ids':list(range(cols*rows)),'columns':cols,'rows':rows,'cases':cases,'point_rule':'fixed mass-label midpoint trajectory,not geometriccellcentre',
      'mass_c0_reference_internal_material_faces_or_F_C_supplied_to_builder':False}


def verify_sampling_rule():
    old=ast.parse(inspect.getsource(sample_frames)).body[0]
    new=ast.parse(inspect.getsource(sample_frames_k32)).body[0]
    old.name=new.name
    loops=[n for n in ast.walk(old) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='K']
    assert len(loops)==1 and isinstance(loops[0].iter,ast.Name) and loops[0].iter.id=='COUNTS'
    loops[0].iter=ast.parse('(32,)',mode='eval').body
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    assert T==F(1,10)
    return {'same_sampling_AST_except_function_name_and_K_tuple':True,'T_exact':str(T),'old_COUNTS_not_mutated':True}


def analyze(out):
    out=Path(out);src=out/'source';old=read(src/'results.json')
    data=arrays(out/'n128/material.npz');col=arrays(src/'n128/column.npz');base=arrays(src/'n32/column.npz')
    m,c0=data['mass'],data['c0'];mc=col['mass'];mp,cp=base['mass'],base['c0'];sigma=old['common_initial_fluctuation_norm']
    path=arrays(out/'n128/k32/trajectory-states.npz');run=read(out/'n128/k32/native-results.json');u=path['states'][-1];v=path['independent_states'][-1]
    u16=arrays(src/'n128/k16/trajectory-states.npz')['states'][-1];saved_common=arrays(src/'common-observations.npz')
    pu=common_observe(u,m,mp);p16=saved_common['n128_k16'];np.testing.assert_array_equal(p16,common_observe(u16,m,mp))
    reference=arrays(src/'reference/observations-l1-k16384.npz');ref=np.tile(reference['n128_projected'],8);pr=reference['n32_projected']
    last=run['steps'][-1];numeric_parts={'machine':last['cumulative_machine_indicator_absolute'],
        'tail':last['cumulative_tail_indicator_absolute'],'complete_sequence_difference':norm(m,u-v)}
    e=sum(numeric_parts.values());transverse=norm(mc,observe_columns(u,mc)-u.reshape(8,128)[0])
    Bparts={'retained_B128':old['time_controls']['128']['budget_absolute'],'twice_new_path_numeric':2*e,'new_K32_transverse_bridge':transverse}
    B=sum(Bparts.values());p_exact,capacity=exact_common(u,m,mp)
    factor=math.nextafter(math.sqrt(float(max(capacity))),math.inf);defect=float(max(abs(q-1) for q in capacity))
    saved_capacity=next(row for row in old['capacity_controls'] if row['columns']==128)
    assert factor==saved_capacity['norm_factor_upper'] and defect==saved_capacity['capacity_defect']
    oldparts=old['spatial_budget_by_grid']['128']['common_parts']
    Jparts={f'retained_{key}':value for key,value in oldparts.items() if key not in ('observed8_16','capacity_times_original_budget')}
    Jparts['new_K32_common_rounding']=norm(mp,np.array([float(F(float(a))-b) for a,b in zip(pu,p_exact)]))
    Jparts['new_K32_capacity_final_bound']=norm(mp,np.array([float(abs(q-1))*float(np.max(abs(u))) for q in capacity]))
    commonB=factor*B+sum(Jparts.values());d=norm(m,u16-u);dp=norm(mp,p16-pu)
    H=metrics(m,u,c0)['H'];Hp=metrics(mp,pu,cp)['H'];H16=old['observations']['128']['native_H'];Hp16=old['observations']['128']['common_H']
    hd=abs(H-H16);hpd=abs(Hp-Hp16);hb=h_indicator(m,c0,B);hpb=h_indicator(mp,cp,commonB)
    n64=old['spatial_budget_by_grid']['64']
    parts={
        'native':{**{f'retained64_{k}':x for k,x in n64['native_parts'].items()},'new128_observed16_32':d,'new128_B':B},
        'common':{**{f'retained64_{k}':x for k,x in n64['common_parts'].items()},'new128_observed16_32':dp,'new128_capacity_B':factor*B,**{f'new128_{k}':x for k,x in Jparts.items()}},
        'native_H':{**{f'retained64_{k}':x for k,x in n64['native_H_parts'].items()},'new128_observed16_32':hd,'new128_original_H_indicator':hb},
        'common_H':{**{f'retained64_{k}':x for k,x in n64['common_H_parts'].items()},'new128_observed16_32':hpd,'new128_original_H_indicator':hpb}}
    errors={'native':norm(m,u-ref)/sigma,'common':norm(mp,pu-pr)/sigma,
        'native_H':abs(H-old['observations']['128']['native_reference_H']),
        'common_H':abs(Hp-old['observations']['128']['common_reference_H'])}
    coarse=old['observations']['64'];old_errors={'native':coarse['native_error_common_denominator'],'common':coarse['common_error'],
        'native_H':abs(coarse['native_H']-coarse['native_reference_H']),'common_H':abs(coarse['common_H']-coarse['common_reference_H'])}
    gains={key:old_errors[key]-errors[key] for key in errors}
    indicators={key:sum(values.values())/(sigma if key in ('native','common') else 1.) for key,values in parts.items()}
    decisions={key:classify(gains[key],indicators[key]) for key in gains}
    tc=old['time_controls']['128'];state_decrease=tc['state8_16']-d;H_decrease=tc['H8_16']-hd
    sequence=all(row['primary_control_mass_norm_difference']<=row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute'] for row in run['steps'])
    raw=all(abs(row['raw_mass_change_from_c0'])/float(m@c0)<=1e-10 and row['minimum']>=float(c0.min())-1e-10 and row['maximum']<=float(c0.max())+1e-10 and min(row['chi_start_half_end'])>=0 for row in run['steps'])
    V0=metrics(m,c0,c0)['variance'];variances=[metrics(m,x,c0)['variance'] for x in path['states']]
    raw=raw and np.isfinite(path['states']).all() and np.isfinite(path['independent_states']).all() and all(b<=a+1e-10*V0 for a,b in zip(variances[:-1],variances[1:]))
    passed=all(decisions[key]=='fine_better_resolvably' for key in ('native','common')) and state_decrease>B and sequence and raw and defect<=1e-10
    assert tc['H8_16']<=tc['H_indicator']<=hb and not H_decrease>hb
    result={'date':'2026-11-07','timezone':'Asia/Shanghai','comparison_decision':'single_finest_K32_concentration_diagnostic_resolved' if passed else 'single_finest_K32_concentration_diagnostic_failed_or_unresolved',
        'concentration_diagnostic_resolved':passed,'mixed_time_levels':{'64':16,'128':32},'common_initial_fluctuation_norm':sigma,
        'errors':errors,'retained64_errors':old_errors,'gains':gains,'fixed_indicators':indicators,'decisions':decisions,'positive_components':parts,
        'numeric_K32_parts':numeric_parts,'B128_star_parts':Bparts,'B128_star':B,'J128_star_parts':Jparts,'common_norm_budget':commonB,
        'capacity':{'defect':defect,'norm_factor_upper':factor,'original_guard_passed':defect<=1e-10},
        'state_time_control':{'old8_16':tc['state8_16'],'new16_32':d,'decrease':state_decrease,'fixed_budget':B,'resolved':state_decrease>B},
        'shifted_H_time_control':{'old8_16':tc['H8_16'],'new16_32':hd,'decrease':H_decrease,'retained_original_H_floor':tc['H_indicator'],'current_original_H_indicator':hb,
            'resolved':H_decrease>hb,'old_increment_below_retained_floor':True,'no_small_H_exemption':True},
        'native_H_K32':H,'common_H_K32':Hp,'raw_all_steps_guards_passed':raw,'all_complete_sequence_controls_passed':sequence,
        'native_K32_mass_change':float(m@(u-c0)),'min_all_steps':min(row['minimum'] for row in run['steps']),'max_all_steps':max(row['maximum'] for row in run['steps']),
        'old_spatial_decision_unchanged':old['comparison_decision'],'old_K4_8_16_time_controls_preserved':old['all_time_controls_resolved'],
        'new_candidate_graphs':32,'new_candidate_transient_runs':1,'new_complete_independent_uniformization_sequences':1,'old_candidate_reruns':0,
        'new_reference_PDE_solves':0,'new_density_sources':0,'new_software_checks':0,'recorded_checks_total':502,
        'new_full_spatial_or_Stage3_pass_allowed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    return result,{'n128_k32':pu,'n128_reference':common_observe(ref,m,mp)}
