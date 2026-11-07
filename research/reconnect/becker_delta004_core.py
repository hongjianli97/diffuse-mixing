"""Fixed-before-data error and H-response bookkeeping for the single delta=.04 case."""
from pathlib import Path
from fractions import Fraction as F
import math
import numpy as np
from audit_route_decision import read
from aligned_becker_check import arrays,norm
from becker_point_core import observe_columns
from becker_spatial_core import common_observe,exact_common,reference_metadata
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator

KS=(8,16,32)


def sign(value,indicator):
    return 'positive' if value>indicator else ('negative' if value < -indicator else 'unresolved')


def analyze(out):
    out=Path(out);data=arrays(out/'n32/material.npz');col=arrays(out/'n32/column.npz')
    old=read(out/'source/common-K32-results.json');oldspace=read(out/'source/spatial-results.json')
    baseline=old['by_grid']['32'];sigma=old['common_initial_fluctuation_norm']
    m,c0,mc,cc=data['mass'],data['c0'],col['mass'],col['c0']
    rm,rc=reference_metadata(out,32,col);ref=np.tile(rc,8)
    states={};projected={};Hs={};HPs={};numeric={};bridges={};rounds={};caps={};runs={};paths={};raw=True;sequence=True
    for K in KS:
        paths[K]=arrays(out/f'n32/k{K}/trajectory-states.npz');runs[K]=read(out/f'n32/k{K}/native-results.json')
        u,v=paths[K]['states'][-1],paths[K]['independent_states'][-1];states[K]=u
        last=runs[K]['steps'][-1]
        numeric[str(K)]={'machine':last['cumulative_machine_indicator_absolute'],'tail':last['cumulative_tail_indicator_absolute'],
            'complete_sequence_difference':norm(m,u-v)}
        # Every H below comes from the new raw path, never the old N32 reuse branch.
        Hs[K]=metrics(m,u,c0)['H'];projected[K]=common_observe(u,m,mc);HPs[K]=metrics(mc,projected[K],cc)['H']
        exact,capacity=exact_common(u,m,mc)
        rounds[K]=norm(mc,np.array([float(F(float(a))-b) for a,b in zip(projected[K],exact)]))
        caps[K]=norm(mc,np.array([float(abs(q-1))*float(np.max(abs(u))) for q in capacity]))
        bridges[K]=norm(mc,observe_columns(u,mc)-u.reshape(8,32)[0])
        sequence=sequence and all(s['primary_control_mass_norm_difference']<=s['cumulative_machine_indicator_absolute']+s['cumulative_tail_indicator_absolute'] for s in runs[K]['steps'])
        vv=[metrics(m,x,c0)['variance'] for x in paths[K]['states']];V0=vv[0]
        raw=raw and np.isfinite(paths[K]['states']).all() and np.isfinite(paths[K]['independent_states']).all()
        raw=raw and all(b<=a+1e-10*V0 for a,b in zip(vv[:-1],vv[1:]))
        raw=raw and all(abs(s['raw_mass_change_from_c0'])/float(m@c0)<=1e-10 and s['minimum']>=float(c0.min())-1e-10 and s['maximum']<=float(c0.max())+1e-10 and min(s['chi_start_half_end'])>=0 for s in runs[K]['steps'])
    Bparts={'twice_all_path_numeric':2*sum(sum(p.values()) for p in numeric.values()),'twice_reference_all_positive_terms':2*rm['total_absolute'],
        'initial_transverse_bridge':norm(mc,observe_columns(c0,mc)-cc),**{f'k{K}_transverse_bridge':bridges[K] for K in KS}}
    B=sum(Bparts.values());pc=common_observe(c0,m,mc);pr=common_observe(ref,m,mc)
    ec,capacity=exact_common(c0,m,mc);er,_=exact_common(ref,m,mc)
    factor=math.nextafter(math.sqrt(float(max(capacity))),math.inf);defect=float(max(abs(q-1) for q in capacity))
    assert factor==baseline['capacity']['norm_factor_upper'] and defect==baseline['capacity']['defect']
    Jparts={'initial_common_vs_original':norm(mc,pc-cc),'reference_common_vs_direct32':norm(mc,pr-rc),
        'initial_common_rounding':norm(mc,np.array([float(F(float(a))-b) for a,b in zip(pc,ec)])),
        'reference_common_rounding':norm(mc,np.array([float(F(float(a))-b) for a,b in zip(pr,er)])),
        'capacity_initial_bound':norm(mc,np.array([float(abs(q-1))*float(np.max(abs(c0))) for q in capacity]))}
    for K in KS:
        Jparts[f'k{K}_common_rounding']=rounds[K];Jparts[f'k{K}_capacity_final_bound']=caps[K]
    J=factor*B+sum(Jparts.values());hb=h_indicator(m,c0,B);hpb=h_indicator(mc,cc,J)
    d816=norm(m,states[8]-states[16]);d1632=norm(m,states[16]-states[32])
    h816=abs(Hs[8]-Hs[16]);h1632=abs(Hs[16]-Hs[32])
    state_ok=d816-d1632>B;H_ok=h816-h1632>hb
    newparts={'native':{'observed16_32':d1632,'positive_norm_budget':B},
        'common':{'observed16_32':norm(mc,projected[16]-projected[32]),'capacity_times_B':factor*B,**Jparts},
        'native_H':{'observed16_32':h1632,'original_H_indicator':hb},
        'common_H':{'observed16_32':abs(HPs[16]-HPs[32]),'original_H_indicator':hpb}}
    Hr=metrics(m,ref,c0)['H'];Hrp=metrics(mc,rc,cc)['H']
    errors={'native':norm(m,states[32]-ref)/sigma,'common':norm(mc,projected[32]-rc)/sigma,
        'native_H':abs(Hs[32]-Hr),'common_H':abs(HPs[32]-Hrp)}
    crossparts={k:{**{'retained_old_'+name:value for name,value in baseline['spatial_parts'][k].items()},
        **{'new_'+name:value for name,value in newparts[k].items()}} for k in errors}
    U={k:sum(p.values())/(sigma if k in ('native','common') else 1.) for k,p in crossparts.items()}
    changes={k:errors[k]-baseline['errors'][k] for k in errors}
    decisions={k:('thinner_error_larger_resolvably' if changes[k]>U[k] else
        'thinner_error_smaller_resolvably' if changes[k] < -U[k] else 'error_change_unresolved') for k in errors}
    responses={}
    for label,hnew,hold,rnew,rold in [('native_H',Hs[32],baseline['native_H_K32'],Hr,oldspace['observations']['32']['native_reference_H']),
                                    ('common_H',HPs[32],baseline['common_H_K32'],Hrp,oldspace['observations']['32']['common_reference_H'])]:
        dc,dr=hnew-hold,rnew-rold;sc,sr=sign(dc,U[label]),sign(dr,U[label])
        status='direction_unresolved' if 'unresolved' in (sc,sr) else ('direction_consistent' if sc==sr else 'direction_inconsistent')
        responses[label]={'candidate_response':dc,'reference_response':dr,'response_difference':dc-dr,
            'fixed_indicator':U[label],'candidate_sign':sc,'reference_sign':sr,'direction_status':status,
            'magnitude_accuracy_acceptance_claimed':False}
    reference_ok=all(g['steps_completed']==k and g['max_relative_mass_drift']<=1e-9 and not g['states_clipped_or_normalized']
        for (level,k),g in zip(((0,8192),(1,8192),(1,16384)),read(out/'reference/reference-runs.json')))
    qualified=bool(state_ok and H_ok and sequence and raw and defect<=1e-10 and reference_ok)
    response_ok=all(v['direction_status']=='direction_consistent' for v in responses.values())
    return {'date':'2026-11-07','timezone':'Etc/UTC','delta':.04,'old_delta':.08,'columns':32,'rows':8,'K':[8,16,32],
        'comparison_decision':'single_delta004_time_qualified_H_direction_consistent' if qualified and response_ok else 'single_delta004_diagnostic_failed_or_unresolved',
        'new_delta_temporal_qualified':qualified,'H_response_directions_consistent':response_ok,
        'common_initial_fluctuation_norm':sigma,'reference_indicators':rm,'numeric_by_K':numeric,
        'B_parts':Bparts,'B':B,'J_parts':Jparts,'J':J,'capacity':{'defect':defect,'norm_factor_upper':factor},
        'new_spatial_parts':newparts,'retained_old_case':baseline,'errors':errors,'error_changes_new_minus_old':changes,
        'fixed_cross_indicators':U,'cross_positive_components':crossparts,'error_change_decisions':decisions,'H_responses':responses,
        'H_by_K':{str(k):Hs[k] for k in KS},'common_H_by_K':{str(k):HPs[k] for k in KS},
        'reference_H':{'native':Hr,'common':Hrp},
        'time_control':{'state8_16':d816,'state16_32':d1632,'state_decrease':d816-d1632,'state_indicator':B,'state_resolved':state_ok,
            'H8_16':h816,'H16_32':h1632,'H_decrease':h816-h1632,'H_indicator':hb,'H_resolved':H_ok},
        'raw_guards_passed':bool(raw),'complete_independent_sequences_passed':sequence,'reference_raw_guards_passed':reference_ok,
        'raw_K32_mass_change':float(m@(states[32]-c0)),
        'all_path_extrema':[min(s['minimum'] for v in runs.values() for s in v['steps']),max(s['maximum'] for v in runs.values() for s in v['steps'])],
        'old_shifted_H_outcomes_preserved':old['shifted_H_time_controls_resolved'],'old_K16_spatial_decision_unchanged':old['old_spatial_decision_unchanged'],
        'new_graphs':56,'new_actual_paths':3,'new_independent_paths':3,'new_reference_PDE_solves':3,'old_candidate_reruns':0,
        'new_density_sources':0,'new_software_checks':0,'recorded_checks_total':502,
        'new_delta_three_grid_control_completed':False,'delta_zero_limit_claimed':False,'H_check_waived':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}, {
        'c0':pc,'reference_aggregated':pr,**{f'k{K}':projected[K] for K in KS}}
