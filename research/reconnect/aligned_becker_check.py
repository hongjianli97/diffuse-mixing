"""Frozen three-solve aligned Becker reference comparison; no candidate reruns."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,math,resource,shutil,signal,sys,time,traceback
import numpy as np
from audit_route_decision import ROOT,read,save,digest
from aligned_mass_reference import build_mesh,observe,solve,RawStateError,StepRestrictionError
from voronoi_time_action import metrics
from t1_uniform_core import h_indicator
sys.path.insert(0,str(ROOT/'research/rebuild'))
from becker import Becker,initial

PARENT=ROOT/'runs/reconnect-aligned-reference';CANDIDATE=ROOT/'runs/reconnect-becker-point'
LEVELS=((0,8192),(1,8192),(1,16384))
PROTOCOL=ROOT/'research/reconnect/aligned_becker_protocol.json'


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def norm(m,v):return float(np.sqrt(m@(v*v)))


def exact_observation(mesh,values,parent_mass):
    answer=[];geometry=[]
    for i,(left,right) in enumerate(zip(mesh.starts[:-1],mesh.starts[1:])):
        total=F(0);defect=F(0)
        for j in range(left,right):
            m=F(float(mesh.mass[j]));v=F(float(values[j]));width=F(float(mesh.edges[j+1]))-F(float(mesh.edges[j]))
            total+=m*v;defect+=abs(m-width)*abs(v)
        denom=F(float(parent_mass[i]));answer.append(total/denom);geometry.append(float(defect/denom))
    return answer,np.array(geometry)


def projected(mesh,c0,c,col):
    p=observe(mesh,c,col['mass']);ip=observe(mesh,c0,col['mass'])
    exact,g=exact_observation(mesh,c,col['mass']);iexact,ig=exact_observation(mesh,c0,col['mass'])
    bridge=np.array([float(F(float(v))-q) for v,q in zip(p,exact)])
    ibridge=np.array([float(F(float(v))-q) for v,q in zip(ip,iexact)])
    initial_gap=np.array([float(q-F(float(v))) for q,v in zip(iexact,col['c0'])])
    return {'projected':p,'initial_projected':ip,'exact_projected_rounded':np.array(list(map(float,exact))),
      'exact_initial_projected_rounded':np.array(list(map(float,iexact))),'terminal_aggregation_rounding':bridge,
      'initial_aggregation_rounding':ibridge,'exact_initial_mean_gap':initial_gap,
      'terminal_geometry_bound':g,'initial_geometry_bound':ig}


def analyze(out):
    out=Path(out);col=arrays(out/'source/column.npz');material=arrays(out/'source/material.npz')
    old=read(out/'source/results.json');prior=read(out/'source/uniform-reference-results.json')
    mc,m,c0=col['mass'],material['mass'],material['c0'];sigma=old['original_column_initial_fluctuation_norm'];refs=[];rows=[]
    for level,k in LEVELS:
        z=arrays(out/f'reference-l{level}-k{k}.npz');refs.append(z['projected'])
        rows.append({'level':level,'cells':len(z['mass']),'steps':k,
          'initial_exact_mean_bridge':norm(mc,z['exact_initial_mean_gap']),
          'initial_aggregation_rounding_bridge':norm(mc,z['initial_aggregation_rounding']),
          'initial_mean_bridge':norm(mc,z['exact_initial_mean_gap'])+norm(mc,z['initial_aggregation_rounding']),
          'terminal_aggregation_rounding_bridge':norm(mc,z['terminal_aggregation_rounding']),
          'initial_geometry_bridge':norm(mc,z['initial_geometry_bound']),'terminal_geometry_bridge':norm(mc,z['terminal_geometry_bound']),
          'original_reference_difference':norm(mc,z['projected']-col['reference']),
          'reference_H':metrics(mc,z['projected'],col['c0'])['H']})
    space=norm(mc,refs[0]-refs[1]);temporal=norm(mc,refs[1]-refs[2]);parts={}
    for name,value in old['time_control']['components_absolute'].items():
        if name=='twice_saved_reference_space_time':parts['twice_aligned_reference_space_time']=2*(space+temporal)
        else:parts[name]=value
    parts['twice_all_initial_mean_bridges']=2*sum(row['initial_mean_bridge'] for row in rows)
    parts['twice_all_terminal_aggregation_bridges']=2*sum(row['terminal_aggregation_rounding_bridge'] for row in rows)
    parts['twice_all_initial_and_terminal_geometry_bridges']=2*sum(row['initial_geometry_bridge']+row['terminal_geometry_bridge'] for row in rows)
    budget=sum(parts.values());hb=h_indicator(m,c0,budget);observations=[];states=[]
    for i,K in enumerate((4,8,16)):
        u=arrays(out/f'source/k{K}-states.npz')['states'][-1];states.append(u);saved=old['observations'][i]['H'];current=metrics(m,u,c0)['H']
        allowance=h_indicator(m,c0,old['observations'][i]['numeric_indicator_relative']*sigma);assert abs(current-saved)<=allowance
        observations.append({'K':K,'native_error_relative':norm(m,u-np.tile(refs[-1],8))/sigma,'saved_H':saved,
          'current_H_diagnostic':current,'current_minus_saved_H':current-saved,
          'old_original_reference_error_relative':old['observations'][i]['native_error_relative'],
          'old_uniform_refined_reference_error_relative':prior['observations'][i]['native_error_relative']})
    state_gap=prior['state_decrease_absolute'];hgap=prior['H_decrease_unchanged']
    current_gap=norm(m,states[0]-states[1])-norm(m,states[1]-states[2])
    assert abs(current_gap-state_gap)<=old['time_control']['components_absolute']['twice_all_path_numeric']
    assert hgap==old['time_control']['H4_8']-old['time_control']['H8_16']
    sp=state_gap>budget;hp=hgap>hb;passed=sp and hp and old['all_complete_sequence_controls_passed']
    return {'decision':'aligned_reference_saved_temporal_comparison_resolved' if passed else 'aligned_reference_saved_temporal_comparison_unresolved',
      'new_temporal_comparison_resolved':bool(passed),'reference_rows':rows,'reference_space_indicator_absolute':space,
      'reference_time_indicator_absolute':temporal,'reference_indicator_absolute':space+temporal,'positive_budget_components_absolute':parts,
      'budget_absolute':budget,'budget_relative':budget/sigma,'H_indicator_unchanged_formula':hb,'state_decrease_saved':state_gap,
      'state_decrease_current_minus_saved_diagnostic':current_gap-state_gap,'H_decrease_saved':hgap,'state_decrease_resolved':bool(sp),'H_decrease_resolved':bool(hp),
      'observations':observations,'old_original_decision_preserved':old['comparison_decision'],'old_uniform_reference_decision_preserved':prior['decision'],
      'old_results_reclassified':False,'old_uniform_H_indicator_preserved':prior['H_indicator_unchanged_formula'],
      'reference_indicator_is_certified_PDE_error_bound':False,'new_reference_PDE_solves':3,'new_candidate_graphs':0,'new_candidate_transient_runs':0,
      'old_candidate_reruns':0,'new_software_checks':0,'recorded_checks_total':502,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}


def preservation():
    old=read(PARENT/'preservation.json');formal=old['formal_manifests_verified']+[{'manifest':str((PARENT/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(PARENT/'sha256.json'),'entries_verified':len(read(PARENT/'sha256.json'))}]
    assert len(formal)==106
    for item in formal:
        assert digest(ROOT/item['manifest'])==item['manifest_sha256']
        for rel,h in read(ROOT/item['manifest']).items():assert digest(ROOT/rel)==h,rel
    originals=read('/workspace/original-handoff-sha256.json');attachments=read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals)==782 and len(attachments)==15
    for rel,h in originals.items():assert digest(ROOT/rel)==h
    for item in attachments:assert digest(Path('/workspace')/item['path'])==item['sha256']
    return {'formal_manifests_verified':formal,'originals_verified':782,'attachments_verified':15,
      'historical_diagnostic_differences_retained':old['historical_diagnostic_differences_retained'],'all_parent_scientific_files_unchanged':True}


def main(output):
    started=time.monotonic();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def cap(*args):raise TimeoutError('registered300s aligned reference cap')
    signal.signal(signal.SIGALRM,cap);signal.alarm(300);completed=[]
    try:
        protocol=read(PROTOCOL);gate=read(PARENT/'next-gate-spec.json')
        assert protocol['source_gate_sha256']==digest(PARENT/'next-gate-spec.json') and not gate['already_executed']
        for item in protocol['sources']:assert digest(ROOT/item['path'])==item['sha256']
        for rel,h in read(PARENT/'sha256.json').items():assert digest(ROOT/rel)==h,rel
        shutil.copyfile(PROTOCOL,out/'protocol.json');shutil.copyfile(PARENT/'next-gate-spec.json',out/'source-gate-spec.json')
        save(out/'implementation-lock.json',{'files':{p:digest(ROOT/p) for p in ('research/reconnect/aligned_becker_check.py',
          'research/reconnect/aligned_becker_protocol.json','research/reconnect/aligned_mass_reference.py','research/rebuild/becker.py',
          'research/reconnect/t1_uniform_core.py','research/reconnect/voronoi_time_action.py')},'fixed_before_new_physical_inputs_meshes_or_solves':True})
        (out/'source').mkdir();copies=[]
        paths=[(CANDIDATE/'source-column-inputs.npz','column.npz'),(CANDIDATE/'material-inputs.npz','material.npz'),
          (CANDIDATE/'results.json','results.json'),(ROOT/'runs/reconnect-becker-reference/results.json','uniform-reference-results.json')]
        paths += [(CANDIDATE/f'k{K}/trajectory-states.npz',f'k{K}-states.npz') for K in (4,8,16)]
        for src,dest in paths:
            shutil.copyfile(src,out/'source'/dest);copies.append({'source':str(src.relative_to(ROOT)),'copy':'source/'+dest,'sha256':digest(src)})
        save(out/'source-copies.json',copies)
        col=arrays(out/'source/column.npz');p=col['mass_edges'];m=col['mass'];flow=Becker(delta=.08)
        assert len(m)==32 and p[0]==0 and p[-1]==flow.total and np.array_equal(np.diff(p),m)
        assert all(F(float(p[i+1]))-F(float(p[i]))==F(float(m[i])) for i in range(32))
        h=(F(float(p[-1]))-F(float(p[0])))/16384;counts=tuple(math.ceil(F(float(v))/h) for v in m)
        assert 16384<=sum(counts)<=16415
        meshes=[build_mesh(p,counts,level=l) for l in (0,1)]
        assert np.array_equal(meshes[0].edges,meshes[1].edges[::2])
        save(out/'mesh-contract.json',{'counts':list(counts),'base_cells':sum(counts),'fine_cells':2*sum(counts),'h_exact':str(h),
          'all_original_parent_masses_equal_exact_endpoint_differences':True,'nested_edges_bitwise':True,'mass_or_rows_normalized':False})
        for level,k in LEVELS:
            mesh=meshes[level];c0=initial(mesh.edges);name=f'l{level}-k{k}'
            np.savez_compressed(out/f'input-{name}.npz',mass_edges=mesh.edges,mass=mesh.mass,centres=mesh.centres,counts=np.array(mesh.counts),starts=mesh.starts,c0=c0)
            tic=time.monotonic()
            try:c,diag,_=solve(mesh,c0,flow.rho_mu,k,T=.1,D=.02,positivity='monitor',history=False)
            except (RawStateError,StepRestrictionError) as exc:
                np.savez_compressed(out/f'rejected-{name}.npz',raw_state=getattr(exc,'raw_state',np.array([])),margin=getattr(exc,'margin',np.array([])))
                save(out/f'rejected-{name}.json',{'step':exc.step,'diagnostics':exc.diagnostics});raise
            np.savez_compressed(out/f'raw-{name}.npz',mass_edges=mesh.edges,mass=mesh.mass,c0=c0,c=c)
            receipt={'level':level,'cells':len(mesh.mass),'steps':k,'solve_seconds':time.monotonic()-tic,**diag};save(out/f'guard-{name}.json',receipt)
            assert diag['steps_completed']==diag['linear_solves']==k and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<1024
            observation=projected(mesh,c0,c,col)
            np.savez_compressed(out/f'reference-{name}.npz',mass_edges=mesh.edges,mass=mesh.mass,c0=c0,c=c,**observation)
            completed.append(receipt);print(f'{name}: {len(mesh.mass)} cells, {receipt["solve_seconds"]:.3f}s, all raw guards passed; insufficient RHS diagonal at {diag["steps_without_sufficient_positivity"]}/{k} steps',flush=True)
        save(out/'reference-runs.json',completed);save(out/'results.json',analyze(out));save(out/'preservation.json',preservation())
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;assert elapsed<300 and peak<1024
        save(out/'execution.json',{'first_attempt_passed':True,'exit_code':0,'elapsed_seconds':elapsed,'peak_RSS_MiB':peak,'new_reference_PDE_solves':3,
          'new_candidate_transient_runs':0,'published_or_pushed':False});print(json.dumps(read(out/'results.json'),indent=2),flush=True)
    except BaseException as exc:
        save(out/'failed-execution.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
          'elapsed_seconds':time.monotonic()-started,'completed_reference_observations':len(completed)});raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();main(args.output)
