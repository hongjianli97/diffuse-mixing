"""One frozen three-solve reference check; existing candidate paths are read only."""
from pathlib import Path
import argparse,csv,json,math,platform,resource,shutil,signal,sys,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from becker_budget_review import arrays,overlap,preserve
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
sys.path.insert(0,str(ROOT/'research/rebuild'))
from becker import Becker
from reference import solve,project

PARENT=ROOT/'runs/reconnect-becker-budget-review'
SOURCE=ROOT/'runs/reconnect-becker-point'
PROTOCOL=ROOT/'research/reconnect/becker_reference_protocol.json'
LEVELS=((16384,8192),(32768,8192),(32768,16384))


def key(n,k):return f'n{n}-k{k}'


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
        assert H[K]==old['observations'][(4,8,16).index(K)]['H']
        observations.append({'K':K,'native_error_relative':norm(m,u-reference)/sigma,'H_unchanged':H[K],
          'refined_reference_H':metrics(m,reference,c0)['H'],'old_reference_error_relative':old['observations'][(4,8,16).index(K)]['native_error_relative']})
    state48=norm(m,states[4]-states[8]);state816=norm(m,states[8]-states[16]);h48=abs(H[4]-H[8]);h816=abs(H[8]-H[16])
    assert state48/sigma==old['time_control']['state4_8_relative'] and state816/sigma==old['time_control']['state8_16_relative']
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


def main(output):
    start=time.monotonic();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def cap(*args):raise TimeoutError('registered300s reference-only cap')
    signal.signal(signal.SIGALRM,cap);signal.alarm(300)
    try:
        protocol=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert not gate['already_executed'] and gate['new_reference_PDE_solves']==3
        assert digest(PARENT/'next-gate-spec.json')==protocol['source_gate_sha256']
        for item in gate['sources']:assert digest(ROOT/item['path'])==item['sha256']
        parent=read(PARENT/'sha256.json')
        for rel,h in parent.items():assert digest(ROOT/rel)==h,rel
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'code_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),
          'reference_source_sha256':digest(ROOT/'research/rebuild/reference.py'),'Becker_source_sha256':digest(ROOT/'research/rebuild/becker.py'),
          'all_fixed_before_new_reference_grids_states':True})
        (out/'source').mkdir();copies=[]
        for src,dest in [('source-column-inputs.npz','column.npz'),('material-inputs.npz','material.npz'),('results.json','results.json')]+[(f'k{K}/trajectory-states.npz',f'k{K}-states.npz') for K in (4,8,16)]:
            shutil.copyfile(SOURCE/src,out/'source'/dest);copies.append({'source':str((SOURCE/src).relative_to(ROOT)),'copy':'source/'+dest,'sha256':digest(SOURCE/src)})
        save(out/'source-copies.json',copies)
        col=arrays(out/'source/column.npz');flow=Becker(delta=.08);receipts=[]
        for n,k in LEVELS:
            tic=time.monotonic();e,initial,c,diag=solve(flow,n,k,T=.1,D=.02)
            # Persist the full returned endpoint before checking the unchanged guards.
            np.savez_compressed(out/f'raw-reference-{key(n,k)}.npz',mass_edges=e,c0=initial,c=c)
            receipt={'cells':n,'steps':k,'solve_wall_seconds':time.monotonic()-tic,**diag}
            save(out/f'guard-{key(n,k)}.json',receipt)
            assert diag['max_relative_mass_drift']<=1e-9
            assert diag['min_all_times']>=diag['initial_min']-1e-10 and diag['max_all_times']<=diag['initial_max']+1e-10
            assert np.isfinite(c).all() and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<1024
            p=overlap(e,c,col['mass_edges']);ip=overlap(e,initial,col['mass_edges']);primitive=project(e,c,col['mass_edges'])
            np.savez_compressed(out/f'reference-{key(n,k)}.npz',mass_edges=e,c0=initial,c=c,projected=p,initial_projected=ip,primitive_projected=primitive)
            receipts.append(receipt);print(f'reference {key(n,k)} complete in {receipt["solve_wall_seconds"]:.3f}s; guards retained',flush=True)
        save(out/'reference-runs.json',receipts);result=analyze(out);save(out/'results.json',result)
        with (out/'comparison.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result['observations'][0]));w.writeheader();w.writerows(result['observations'])
        pres=preserve();pres['formal_manifests_verified'].append({'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
          'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(parent)})
        assert len(pres['formal_manifests_verified'])==103;save(out/'preservation.json',pres)
        elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<300 and peak<1024
        save(out/'execution.json',{'exit_code':0,'first_attempt_passed':True,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,
          'python':platform.python_version(),'numpy':np.__version__,'new_reference_PDE_solves':3,'candidate_solver_calls':0,'old_states_changed':False,
          'published':False,'network_used':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'exit_code':1,'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start})
        raise
    finally:signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
