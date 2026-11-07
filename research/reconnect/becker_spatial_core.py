"""Frozen endpoint spatial budgets for the same bounded point-FV backend."""
from pathlib import Path
from fractions import Fraction as F
import math
import numpy as np
from audit_route_decision import read
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
from becker_point_core import observe_columns
from aligned_becker_check import arrays,norm
from shear_union64_study import classify

NS=(32,64,128);KS=(4,8,16);REFS=((0,8192),(1,8192),(1,16384))


def common_observe(v,m,mp):
    ratio=len(m)//(8*len(mp))
    if ratio not in (1,2,4) or np.asarray(v).shape!=np.asarray(m).shape:raise ValueError('fixed complete32-parent observation required')
    return (np.asarray(m)*v).reshape(8,len(mp),ratio).sum(axis=(0,2))/mp


def exact_common(v,m,mp):
    ratio=len(m)//(8*len(mp));N=len(m)//8;answer=[];capacity=[]
    for i,parent in enumerate(mp):
        ids=[row*N+i*ratio+j for row in range(8) for j in range(ratio)]
        mass=sum((F(float(m[j])) for j in ids),F(0));total=sum((F(float(m[j]))*F(float(v[j])) for j in ids),F(0))
        answer.append(total/F(float(parent)));capacity.append(mass/F(float(parent)))
    return answer,capacity


def reference_metadata(out,N,column):
    mc=column['mass'];recs=[];ps=[]
    for level,k in REFS:
        z=arrays(Path(out)/f'reference/observations-l{level}-k{k}.npz');get=lambda key:z[f'n{N}_{key}']
        ps.append(get('projected'))
        parts={key:norm(mc,get(key)) for key in ('exact_initial_mean_gap','initial_aggregation_rounding',
          'terminal_aggregation_rounding','initial_geometry_bound','terminal_geometry_bound')}
        recs.append({'level':level,'steps':k,'components_absolute':parts})
    parts={'space':norm(mc,ps[0]-ps[1]),'time':norm(mc,ps[1]-ps[2])}
    for i,row in enumerate(recs):parts.update({f'reference{i}_{key}':value for key,value in row['components_absolute'].items()})
    return {'components_absolute':parts,'total_absolute':sum(parts.values()),'rows':recs},ps[-1]


def analyze(out):
    out=Path(out);columns={N:arrays(out/f'n{N}/column.npz') for N in NS};data={N:arrays(out/f'n{N}/material.npz') for N in NS}
    mp,cp=columns[32]['mass'],columns[32]['c0'];sigma=math.sqrt(metrics(mp,cp,cp)['variance'])
    old=read(out/'reused-original-results.json');prior=read(out/'reused-aligned-results.json')
    paths={};runs={};refs={};rm={};common={};observations={};time_controls={};spatial_budgets={};capacities=[]
    for N in NS:
        m,c0=data[N]['mass'],data[N]['c0'];mc=columns[N]['mass'];cc=columns[N]['c0']
        rm[N],refs[N]=reference_metadata(out,N,columns[N]);ref=np.tile(refs[N],8)
        states={};H={};nums={};rowbridges={};common_round={};proj={}
        for K in KS:
            paths[N,K]=arrays(out/f'n{N}/k{K}/trajectory-states.npz');runs[N,K]=read(out/f'n{N}/k{K}/native-results.json')
            u=paths[N,K]['states'][-1];v=paths[N,K]['independent_states'][-1];step=runs[N,K]['steps'][-1];states[K]=u
            nums[K]=float(step['cumulative_machine_indicator_absolute'])+float(step['cumulative_tail_indicator_absolute'])+norm(m,u-v)
            H[K]=old['observations'][KS.index(K)]['H'] if N==32 else metrics(m,u,c0)['H']
            hd=metrics(m,u,c0)['H'];assert abs(hd-H[K])<=h_indicator(m,c0,nums[K])
            rowbridges[K]=norm(mc,observe_columns(u,mc)-u.reshape(8,N)[0])
            proj[K]=common_observe(u,m,mp);exact,_=exact_common(u,m,mp)
            common_round[K]=norm(mp,np.array([float(F(float(a))-b) for a,b in zip(proj[K],exact)]))
        pc=common_observe(c0,m,mp);pr=common_observe(ref,m,mp)
        exc,capacity=exact_common(c0,m,mp);exr,_=exact_common(ref,m,mp)
        init_round=norm(mp,np.array([float(F(float(a))-b) for a,b in zip(pc,exc)]))
        ref_round=norm(mp,np.array([float(F(float(a))-b) for a,b in zip(pr,exr)]))
        factor=math.nextafter(math.sqrt(float(max(capacity))),math.inf)
        defect=float(max(abs(q-1) for q in capacity));capacities.append({'columns':N,'capacity_defect':defect,'norm_factor_upper':factor,'original_capacity_guard_passed':defect<=1e-10})
        parts={'twice_all_path_numeric':2*sum(nums.values()),'twice_reference_all_positive_terms':2*rm[N]['total_absolute'],
          'initial_transverse_bridge':norm(mc,observe_columns(c0,mc)-cc)}
        parts.update({f'k{K}_transverse_bridge':rowbridges[K] for K in KS})
        B=sum(parts.values());hb=h_indicator(m,c0,B)
        d48=norm(m,states[4]-states[8]);d816=norm(m,states[8]-states[16]);h48=abs(H[4]-H[8]);h816=abs(H[8]-H[16])
        sequence=all(row['primary_control_mass_norm_difference']<=row['cumulative_machine_indicator_absolute']+row['cumulative_tail_indicator_absolute'] for K in KS for row in runs[N,K]['steps'])
        passed=d48-d816>B and h48-h816>hb and sequence
        time_controls[str(N)]={'components_absolute':parts,'budget_absolute':B,'H_indicator':hb,'state4_8':d48,'state8_16':d816,
          'H4_8':h48,'H8_16':h816,'state_decrease':d48-d816,'H_decrease':h48-h816,'state_decrease_resolved':d48-d816>B,
          'H_decrease_resolved':h48-h816>hb,'all_complete_sequence_controls_passed':sequence,'time_control_resolved':passed,
          'numeric_by_K':{str(k):v for k,v in nums.items()},'H_by_K':{str(k):v for k,v in H.items()}}
        base_ref=refs[32];Hc=lambda a:metrics(mp,a,cp)['H']
        bridgeparts={'initial_common_vs_original':norm(mp,pc-cp),'reference_common_vs_direct32':norm(mp,pr-base_ref),
          'initial_common_rounding':init_round,'reference_common_rounding':ref_round,'k8_common_rounding':common_round[8],
          'k16_common_rounding':common_round[16],
          'capacity_initial_bound':norm(mp,np.array([float(abs(q-1))*float(np.max(abs(c0))) for q in capacity])),
          'capacity_final_bound':norm(mp,np.array([float(abs(q-1))*float(np.max(abs(states[16]))) for q in capacity]))}
        common_B=factor*B+sum(bridgeparts.values())
        native_parts={'observed8_16':d816,'original_time_control_positive_budget':B}
        nativeH_parts={'observed8_16':h816,'original_H_numeric_reference':h_indicator(m,c0,B)}
        common_parts={'observed8_16':norm(mp,proj[8]-proj[16]),'capacity_times_original_budget':factor*B,**bridgeparts}
        commonH_parts={'observed8_16':abs(Hc(proj[8])-Hc(proj[16])),'original_H_positive_norm_budget':h_indicator(mp,cp,common_B)}
        spatial_budgets[str(N)]={'native_parts':native_parts,'native_H_parts':nativeH_parts,'common_parts':common_parts,'common_H_parts':commonH_parts}
        observations[str(N)]={'native_error_common_denominator':norm(m,states[16]-ref)/sigma,
          'native_error_native_denominator':norm(m,states[16]-ref)/math.sqrt(metrics(m,c0,c0)['variance']),
          'common_error':norm(mp,proj[16]-base_ref)/sigma,'native_H':H[16],'native_reference_H':metrics(m,ref,c0)['H'],
          'common_H':Hc(proj[16]),'common_reference_H':Hc(base_ref),'native_K16_mass_change':float(m@(states[16]-c0)),
          'current_K4_H_minus_saved_diagnostic':metrics(m,states[4],c0)['H']-H[4]}
        common[f'n{N}_c0']=pc;common[f'n{N}_reference']=pr
        for K in KS:common[f'n{N}_k{K}']=proj[K]
    adjacent=[]
    for C,N in ((32,64),(64,128)):
        a,b=observations[str(C)],observations[str(N)]
        gains={'native':a['native_error_common_denominator']-b['native_error_common_denominator'],'common':a['common_error']-b['common_error'],
          'native_H':abs(a['native_H']-a['native_reference_H'])-abs(b['native_H']-b['native_reference_H']),
          'common_H':abs(a['common_H']-a['common_reference_H'])-abs(b['common_H']-b['common_reference_H'])}
        parts={label:{f'n{Q}_{key}':v for Q in (C,N) for key,v in spatial_budgets[str(Q)][name].items()} for label,name in
          [('native','native_parts'),('common','common_parts'),('native_H','native_H_parts'),('common_H','common_H_parts')]}
        indicators={label:sum(p.values())/(sigma if label in ('native','common') else 1.) for label,p in parts.items()}
        adjacent.append({'coarse_columns':C,'fine_columns':N,'time':.1,'gains':gains,'fixed_indicators':indicators,
          'decisions':{key:classify(gains[key],indicators[key]) for key in gains},'positive_components':parts})
    spatial=all(v=='fine_better_resolvably' for row in adjacent for v in row['decisions'].values())
    temporal=all(row['time_control_resolved'] for row in time_controls.values());capacity_ok=all(c['original_capacity_guard_passed'] for c in capacities)
    passed=spatial and temporal and capacity_ok
    result={'comparison_decision':'T4_three_column_resolution_moving_Becker_oracle_control_passed' if passed else 'T4_moving_Becker_spatial_control_failed_or_unresolved',
      'registered_spatial_control_passed':passed,'common_initial_fluctuation_norm':sigma,'observations':observations,
      'reference_indicators':{str(N):rm[N] for N in NS},'time_controls':time_controls,'spatial_budget_by_grid':spatial_budgets,
      'capacity_controls':capacities,'adjacent_spatial_comparisons':adjacent,'all_native_common_H_spatial_gains_resolved':spatial,
      'all_time_controls_resolved':temporal,'old_32_temporal_pass_preserved':prior['new_temporal_comparison_resolved'],
      'old_two_unresolved_records_reclassified':False,'new_candidate_graphs':56,'new_candidate_transient_runs':6,
      'new_complete_independent_uniformization_sequences':6,'old_candidate_reruns':0,'new_reference_PDE_solves':3,
      'new_software_checks':0,'recorded_checks_total':502,'one_dimensional_column_refinement_only':True,'general_order_claimed':False,
      'indicators_are_certified_error_bounds':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    return result,common
