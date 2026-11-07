"""Append-only reporting completion for saved-H cross-environment reduction rounding.

No reference PDE/candidate rerun, no change to the original H formula or budget.
"""
from pathlib import Path
import argparse,csv,json,resource,time
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from becker_reference_check import arrays,LEVELS,key,PARENT,SOURCE
from becker_budget_review import preserve
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator

def analyze(out):
    out=Path(out);col=arrays(out/'source/column.npz');data=arrays(out/'source/material.npz');old=read(out/'source/results.json')
    m,c0=data['mass'],data['c0'];mc=col['mass'];norm=lambda mass,v:float(np.sqrt(mass@(v*v)))
    sigma=old['original_column_initial_fluctuation_norm'];refs={};rows=[]
    for n,k in LEVELS:
        z=arrays(out/f'reference-{key(n,k)}.npz');refs[n,k]=z
        p=z['projected'];ip=z['initial_projected']
        rows.append({'cells':n,'steps':k,'projection_bridge_absolute':norm(mc,p-z['primitive_projected']),
          'initial_mean_bridge_absolute':norm(mc,ip-col['c0']),
          'column_H':metrics(mc,p,col['c0'])['H'],'old_reference_difference_absolute':norm(mc,p-col['reference'])})
    space=norm(mc,refs[LEVELS[0]]['projected']-refs[LEVELS[1]]['projected'])
    temporal=norm(mc,refs[LEVELS[1]]['projected']-refs[LEVELS[2]]['projected'])
    pfinal=refs[LEVELS[-1]]['projected'];reference=np.tile(pfinal,8)
    parts={}
    for name,value in old['time_control']['components_absolute'].items():
        if name=='twice_saved_reference_space_time':parts['twice_refined_reference_space_time']=2*(space+temporal)
        else:parts[name]=value
    parts['twice_all_new_projection_bridges']=2*sum(row['projection_bridge_absolute'] for row in rows)
    parts['twice_all_new_initial_mean_bridges']=2*sum(row['initial_mean_bridge_absolute'] for row in rows)
    budget=sum(parts.values());hb=h_indicator(m,c0,budget);observations=[];states={};H={}
    for K in (4,8,16):
        u=arrays(out/f'source/k{K}-states.npz')['states'][-1];states[K]=u;H[K]=metrics(m,u,c0)['H']
        saved_H=old['observations'][(4,8,16).index(K)]['H']
        diagnostic_H=H[K];H[K]=saved_H
        allowance=h_indicator(m,c0,old['observations'][(4,8,16).index(K)]['numeric_indicator_relative']*sigma)
        assert abs(diagnostic_H-saved_H)<=allowance
        observations.append({'K':K,'native_error_relative':norm(m,u-reference)/sigma,'H_unchanged':H[K],'H_current_platform_evaluation':diagnostic_H,'H_current_minus_saved_diagnostic':diagnostic_H-saved_H,
          'refined_reference_H':metrics(m,reference,c0)['H'],'old_reference_error_relative':old['observations'][(4,8,16).index(K)]['native_error_relative']})
    state48=norm(m,states[4]-states[8]);state816=norm(m,states[8]-states[16]);h48=abs(H[4]-H[8]);h816=abs(H[8]-H[16])
    assert abs(state48/sigma-old['time_control']['state4_8_relative'])<=old['time_control']['fixed_state_decrease_indicator_relative']
    assert abs(state816/sigma-old['time_control']['state8_16_relative'])<=old['time_control']['fixed_state_decrease_indicator_relative']
    assert h48==old['time_control']['H4_8'] and h816==old['time_control']['H8_16']
    statepass=state48-state816>budget;hpass=h48-h816>hb;passed=statepass and hpass and old['all_complete_sequence_controls_passed']
    return {'decision':'refined_reference_saved_temporal_comparison_resolved' if passed else 'refined_reference_saved_temporal_comparison_unresolved',
      'new_temporal_comparison_resolved':passed,'original_case_decision_retained':old['comparison_decision'],'original_case_reclassified':False,
      'reference_space_indicator_absolute':space,'reference_time_indicator_absolute':temporal,'reference_indicator_absolute':space+temporal,
      'reference_indicators_are_certified_error_bounds':False,'reference_rows':rows,'positive_budget_components_absolute':parts,
      'budget_absolute':budget,'budget_relative':budget/sigma,'H_indicator_unchanged_formula':hb,
      'state_decrease_absolute':state48-state816,'H_decrease_unchanged':h48-h816,'state_decrease_resolved':statepass,'H_decrease_resolved':hpass,
      'original_H_indicator':old['time_control']['fixed_H_decrease_indicator'],'observations':observations,
      'initial_mean_bridges_include_finite_cell_representation_not_only_rounding':True,
      'new_reference_PDE_solves':3,'new_candidate_graphs':0,'new_candidate_transient_runs':0,'old_candidate_reruns':0,'new_software_checks':0,
      'recorded_checks_total':500,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'single_candidate_N_not_spatial_convergence':True,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def main(directory):
    started=time.monotonic();out=Path(directory).resolve();snapshot=read(out/'pre-completion-sha256.json')
    for rel,h in snapshot.items():assert digest(out/rel)==h
    assert read(out/'failed-execution.json')['type']=='AssertionError'
    save(out/'completion-lock.json',{'code_sha256':digest(Path(__file__)),'snapshot_sha256':digest(out/'pre-completion-sha256.json'),
      'repair':'retain original saved H scalars; disclose current-platform reevaluation difference under original numeric allowance; no formula/budget/threshold changes'})
    result=analyze(out);save(out/'results.json',result)
    with (out/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result['observations'][0]));w.writeheader();w.writerows(result['observations'])
    pres=preserve();parent=read(PARENT/'sha256.json')
    for rel,h in parent.items():assert digest(ROOT/rel)==h
    pres['formal_manifests_verified'].append({'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(parent)})
    assert len(pres['formal_manifests_verified'])==103;save(out/'preservation.json',pres)
    for rel,h in snapshot.items():assert digest(out/rel)==h
    save(out/'execution.json',{'exit_code':0,'first_attempt_passed':False,'original_main_exit_code':1,'completion_exit_code':0,
      'elapsed_original_seconds':read(out/'failed-execution.json')['elapsed_seconds'],'completion_seconds':time.monotonic()-started,
      'peak_completion_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'pre_completion_files_unchanged':len(snapshot),
      'new_reference_PDE_solves_total':3,'reference_solves_during_completion':0,'candidate_solver_calls':0,
      'original_saved_H_retained':True,'new_software_checks':0,'recorded_checks_total':500,'published':False})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
