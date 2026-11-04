"""One registered N64 material-union control; reuse old128/256 paths without reruns."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math,platform,resource,shutil,signal,time
import numpy as np
import scipy
from audit_route_decision import ROOT,read,save,digest
from periodic_voronoi_fv import build_periodic_faces,coefficient_packet
from shear_voronoi_fv_study import record,verify_probe_cells
from voronoi_time_action import operators,metrics
from voronoi_time16_core import sample_points,ordered_path
from material_union import group_sum,block_index,union_labels,oracle_density,define_union,observe_path

SOURCE=ROOT/'runs/reconnect-shear-voronoi-time16'
PROTOCOL=ROOT/'research/reconnect/union64_protocol.json'
CONTROL=ROOT/'research/reconnect/SHEAR_UNION64_CONTROL.md'
SEED=20261101
COPIES={
 'original-comparison.npz':SOURCE/'original-comparison.npz',
 'old-n128-k8-states.npz':ROOT/'runs/reconnect-shear-voronoi-time/trajectory-states.npz',
 'old-n256-k8-states.npz':SOURCE/'source-spatial-states.npz',
 'old-n128-k8-results.json':SOURCE/'source-coarse-results.json',
 'old-n256-k8-results.json':SOURCE/'source-fine-results.json',
 'old-n128-k16-states.npz':SOURCE/'n128/trajectory-states.npz',
 'old-n256-k16-states.npz':SOURCE/'n256/trajectory-states.npz',
 'old-n128-k16-results.json':SOURCE/'n128/native-results.json',
 'old-n256-k16-results.json':SOURCE/'n256/native-results.json',
 'old-results.json':SOURCE/'results.json'}


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def native_data(d):
    m,c,r=define_union(d['parent_mass'],d['parent_c0'],d['parent_reference_states'])
    data={64:{'mass':m,'c0':c,'reference':r,
       'space':observe_path(d['parent_reference_space_difference'],d['parent_mass'],m),
       'time':observe_path(d['parent_reference_time_difference'],d['parent_mass'],m)}}
    for N,prefix in [(128,'parent_'),(256,'')]:
        data[N]={'mass':d[prefix+'mass'],'c0':d[prefix+'c0'],'reference':d[prefix+'reference_states'],
          'space':d[prefix+'reference_space_difference'],'time':d[prefix+'reference_time_difference']}
    return data


def load_old(out):
    runs={};paths={}
    for N in (128,256):
        for K in (8,16):
            z=arrays(out/f'old-n{N}-k{K}-states.npz');key=(N,K)
            paths[key]={field:z[(f'k{K}_' if K==8 else '')+field] for field in ('states','independent_states','exponential_half_states')}
            r=read(out/f'old-n{N}-k{K}-results.json');runs[key]=r['cases'][str(K)] if K==8 else r
    return runs,paths


def native_run(out,K,data,index,guard=lambda *a:None,replay=False):
    folder=out/f'k{K}';m,c0=data['mass'],data['c0']
    def load(step):
        item=index[f'k{K}-step{step:02d}'];p=out/item['file'];assert digest(p)==item['sha256']
        packet=read(p);op=operators(m,np.array(packet['edges'],dtype=np.int64),map(F,packet['conductances_exact']))
        del packet;gc.collect();return op
    def persist(step,op,w):
        a={'symmetric_data':op.symmetric_generator.data,'symmetric_indices':op.symmetric_generator.indices,
           'symmetric_indptr':op.symmetric_generator.indptr,'uniformization_data':op.uniformization_matrix.data,
           'uniformization_indices':op.uniformization_matrix.indices,'uniformization_indptr':op.uniformization_matrix.indptr,
           'shape':np.array(op.symmetric_generator.shape),'edges':op.edges,'g_float':op.conductance,'weights':w}
        p=folder/f'action-step{step:02d}.npz'
        if replay:
            saved=arrays(p)
            for k,v in a.items():assert np.array_equal(v,saved[k]),k
        else:np.savez_compressed(p,**a)
    np.random.seed(SEED)
    run,path=ordered_path(m,c0,load,K,persist,lambda s,k,total:guard(f'k{K}-{s}',k,total))
    for step,row in enumerate(run['steps']):
        item=index[f'k{K}-step{step:02d}'];row['coefficient_cache_sha256']=item['sha256'];row['midpoint_time_saved']=item['midpoint_time_saved']
    return run,path


def classify(gain,budget):
    return 'fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')


def compare(data,runs,paths):
    mp,cp=data[64]['mass'],data[64]['c0'];E0=metrics(mp,cp,cp)['variance'];sigma=math.sqrt(E0)
    norm=lambda m,v:float(np.sqrt(m@(v*v)))/sigma
    cnorm=lambda v:norm(mp,v)
    H=lambda c:metrics(mp,c,cp)['H']
    common={};projection={};initial={};factors={};native_sigma={};per_grid={}
    for N,d in data.items():
        m,c0=d['mass'],d['c0'];ratio=N//64;project=lambda v:observe_path(v,m,mp,ratio=ratio)
        ref=project(d['reference']);projection[N]=ref;initial[N]=project(c0[None,:])[0]
        factors[N]=math.sqrt(float(np.max(group_sum(m,64,ratio)/mp)))
        native_sigma[N]=math.sqrt(metrics(m,c0,c0)['variance'])
        common[f'n{N}_reference']=ref;common[f'n{N}_c0']=initial[N]
        for K in (8,16):
            for field in ('states','independent_states'):
                common[f'n{N}_k{K}_{field}']=project(paths[N,K][field])
        per_grid[str(N)]=[]
        for k,t in enumerate((0.,.025,.05)):
            c=paths[N,16]['states'][8*k];p=common[f'n{N}_k16_states'][8*k];refnative=d['reference'][k]
            per_grid[str(N)].append({'time':t,'native_error_common_denominator':norm(m,c-refnative),
              'native_error_native_denominator':norm(m,c-refnative)*sigma/native_sigma[N],
              'common_error':cnorm(p-data[64]['reference'][k]),'native_H':metrics(m,c,c0)['H'],
              'native_reference_H':metrics(m,refnative,c0)['H'],'common_H':H(p),'reference_common_H':H(data[64]['reference'][k]),
              'native_raw_mass_change':float(m@(c-c0)),'common_raw_mass_change':float(mp@(p-cp))})
    rows=[]
    for C,N in ((64,128),(128,256)):
        mC,mN=data[C]['mass'],data[N]['mass'];bridges=np.array([data[64]['reference'][0]-cp,initial[N]-cp,projection[N][0]-initial[N]])
        common[f'pair{C}_{N}_initial_bridges']=bridges
        Hbridge=abs(H(data[64]['reference'][0])-H(cp))+abs(H(initial[N])-H(cp))+abs(H(projection[N][0])-H(initial[N]))
        for k,t in enumerate((0.,.025,.05)):
            c=common[f'n{C}_k16_states'][8*k];f=common[f'n{N}_k16_states'][8*k];ref=data[64]['reference'][k]
            c8=common[f'n{C}_k8_states'][4*k];f8=common[f'n{N}_k8_states'][4*k]
            solver_abs=solver_native_abs=observed=observed_native=observedH=0.
            for Q in (C,N):
                for K in (8,16):
                    i=k*K//2
                    if not i:continue
                    step=runs[Q,K]['steps'][i-1]
                    a=step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute']
                    solver_abs+=factors[Q]*a;solver_native_abs+=a
                    u,v=common[f'n{Q}_k{K}_states'][i],common[f'n{Q}_k{K}_independent_states'][i]
                    observed+=cnorm(u-v);observedH+=abs(H(u)-H(v))
                    observed_native+=norm(data[Q]['mass'],paths[Q,K]['states'][i]-paths[Q,K]['independent_states'][i])
            components={'coarse8_16':cnorm(c8-c),'fine8_16':cnorm(f8-f),
              'twice_reference_space':2*cnorm(data[64]['space'][k]),'twice_reference_time':2*cnorm(data[64]['time'][k]),
              'coarse_reference_projection':cnorm(projection[C][k]-ref),'fine_reference_projection':cnorm(projection[N][k]-ref),
              'initial_bridges':sum(cnorm(v) for v in bridges),'solver_machine_and_tails':solver_abs/sigma,'observed_solver_sequences':observed}
            native_components={'coarse8_16':norm(mC,paths[C,8]['states'][4*k]-paths[C,16]['states'][8*k]),
              'fine8_16':norm(mN,paths[N,8]['states'][4*k]-paths[N,16]['states'][8*k]),
              'twice_reference_space_time':2*sum(norm(data[Q]['mass'],data[Q][field][k]) for Q in (C,N) for field in ('space','time')),
              'initial_reference_bridges':sum(norm(data[Q]['mass'],data[Q]['reference'][0]-data[Q]['c0']) for Q in (C,N)),
              'solver_machine_and_tails':solver_native_abs/sigma,'observed_solver_sequences':observed_native}
            rt=ref+data[64]['time'][k];rs=rt+data[64]['space'][k]
            hp={'coarse8_16':abs(H(c8)-H(c)),'fine8_16':abs(H(f8)-H(f)),
              'twice_reference_space':2*abs(H(rs)-H(rt)),'twice_reference_time':2*abs(H(rt)-H(ref)),
              'coarse_reference_projection':abs(H(projection[C][k])-H(ref)),
              'fine_reference_projection':abs(H(projection[N][k])-H(ref)),
              'initial_bridges':Hbridge,'observed_solver_sequences':observedH,
              'solver_machine_to_H':(2*math.sqrt(float(mp.sum()))*max(1.,float(np.max(abs(cp))))*solver_abs+solver_abs**2)/E0}
            ec,ef=cnorm(c-ref),cnorm(f-ref)
            nc=norm(mC,paths[C,16]['states'][8*k]-data[C]['reference'][k]);nf=norm(mN,paths[N,16]['states'][8*k]-data[N]['reference'][k])
            hg=abs(H(c)-H(ref))-abs(H(f)-H(ref))
            ratio=lambda a,b:None if b==0 else a/b
            order=lambda a,b:None if min(a,b)<=0 else math.log2(a/b)
            row={'coarse_N':C,'fine_N':N,'time':t,'coarse_common_error':ec,'fine_common_error':ef,
              'common_gain':ec-ef,'common_indicator_components':components,'common_indicator':sum(components.values()),
              'native_coarse_error':nc,'native_fine_error':nf,'native_gain':nc-nf,'native_indicator_components':native_components,
              'native_indicator':sum(native_components.values()),'coarse_H_common':H(c),'fine_H_common':H(f),'reference_H_common':H(ref),
              'H_gain':hg,'H_indicator_components':hp,'H_indicator':sum(hp.values()),
              'common_error_ratio':ratio(ec,ef),'native_error_ratio':ratio(nc,nf),
              'common_observed_log2_ratio':order(ec,ef),'native_observed_log2_ratio':order(nc,nf)}
            for label in ('common','native','H'):row[label+'_decision']=classify(row[label+'_gain'],row[label+'_indicator'])
            rows.append(row)
    active=[r for r in rows if r['time']>0]
    resolved=all(r[label+'_decision']=='fine_better_resolvably' for r in active for label in ('common','native','H'))
    adverse=any(r[label+'_decision']=='fine_worse_resolvably' for r in active for label in ('common','native','H'))
    decision='three_level_improvement_resolved_with_native_control' if resolved else ('adverse_at_a_checkpoint' if adverse else 'three_level_comparison_inconclusive')
    result={'common_initial_fluctuation_norm':sigma,'native_initial_fluctuation_norms':{str(k):v for k,v in native_sigma.items()},
      'raw_projection_solver_factors':{str(k):v for k,v in factors.items()},'per_grid_observations':per_grid,'adjacent_comparisons':rows,
      'comparison_decision':decision,'all_adjacent_times_common_native_H_resolved':resolved,
      'empirical_indicators_are_certified_bounds':False,'observed_ratios_certify_general_order':False,
      'new_reference_PDE_solves':0,'old_candidate_reruns':0,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    return result,common


def main(output,tests,test_log):
    prior=read(SOURCE/'sha256.json');assert len(prior)==2174
    for rel,h in prior.items():assert digest(ROOT/rel)==h,rel
    gate=read(SOURCE/'next-gate-spec.json');protocol=read(PROTOCOL)
    assert not gate['already_executed'] and gate['new_midpoint_graphs']==24 and gate['new_ordered_interval_counts']==[8,16]
    assert gate['main_wall_seconds_cap']==600 and gate['main_peak_RSS_MiB_cap']==1024
    assert read(tests)=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n,p in [('source-gate-spec.json',SOURCE/'next-gate-spec.json'),('implementation-protocol.json',PROTOCOL),
       ('control-at-lock.md',CONTROL),('tests.json',Path(tests)),('tests.log',Path(test_log))]:shutil.copyfile(p,out/n)
    locked=['periodic_voronoi_fv.py','voronoi_time_action.py','voronoi_time16_core.py','shear_probe.py',
      'material_union.py','test_material_union.py','shear_union64_study.py']
    save(out/'implementation-lock.json',{'files':{n:digest(ROOT/'research/reconnect'/n) for n in locked},
      'protocol_sha256':digest(PROTOCOL),'control_note_sha256':digest(CONTROL),'fixed_before_any_new_input_or_geometry':True})
    save(out/'config.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','parent_manifest_entries_verified':2174,
      'parent_manifest_sha256':digest(SOURCE/'sha256.json'),'new_software_checks':4,'recorded_checks_total':457,
      'previous_checks_rerun':0,'new_reference_PDE_solves':0,'new_spatial_levels':[64]})
    started=time.monotonic()
    def guard(stage,k=0,total=0):
        elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if elapsed>600 or peak>1024:raise RuntimeError(f'registered resource cap exceeded:{elapsed}s/{peak}MiB')
        if not total or k%4096==0 or k==total:print(f'{stage}:{k}/{total};{elapsed:.3f}s;peak{peak:.2f}MiB',flush=True)
    def alarm(signum,frame):raise TimeoutError('registered600second main cap reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        source=arrays(SOURCE/'trajectory-source.npz');checks={}
        for N in (128,256):
            r=oracle_density(source[f'n{N}_labels']);assert np.array_equal(r,source[f'n{N}_rho0_point'])
            checks[str(N)]={'same_given_oracle_density_bitwise_verified':True,'rho_record':record(r)}
        a=union_labels(source['n128_labels']);ids=np.arange(64**2,dtype=np.int64);rho=oracle_density(a)
        membership={'labels':a,'ids':ids,'rho0_point':rho,'n128_to_n64':block_index(64,2),'n256_to_n64':block_index(64,4)}
        for k,v in source.items():membership[k]=v
        np.savez_compressed(out/'trajectory-source.npz',**membership)
        b={'ids':ids}
        for K in (8,16):
            p=sample_points(a,ids,rho,np.linspace(0.,.05,2*K+1))
            for key in ('time','positions','rho'):b[f'k{K}_{key}']=p[key]
            b[f'k{K}_boundary_time']=p['time'][::2];b[f'k{K}_midpoint_time']=p['time'][1::2]
            (out/f'k{K}').mkdir()
        np.savez_compressed(out/'builder-inputs.npz',**b)
        save(out/'input-matching.json',{'source':str((SOURCE/'trajectory-source.npz').relative_to(ROOT)),
          'source_sha256':digest(SOURCE/'trajectory-source.npz'),'old_density_checks':checks,
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'actual_mass_c0_reference_loaded':False,
          'new_label_center_record':record(a),'new_point_density_record':record(rho)})
        del source,membership;gc.collect();index={};guard('pure-union-points-sealed')
        for K in (8,16):
            for step in range(K):
                frame=2*step+1
                graph=build_periodic_faces(ids,b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],.02,(1,1),1,guard)
                probes=verify_probe_cells(graph,protocol['fixed_geometry_probe_ids']);packet=coefficient_packet(graph)
                name=f'k{K}/geometry-step{step:02d}.json';save(out/name,packet)
                index[f'k{K}-step{step:02d}']={'N':64,'K':K,'step':step,'frame':frame,'file':name,'sha256':digest(out/name),
                  'midpoint_time_saved':float(b[f'k{K}_midpoint_time'][step]),'geometry_diagnostics':graph.diagnostics,'independent_fixed_probes':probes}
                del graph,packet;gc.collect();guard(f'sealed-geometry-k{K}-step{step:02d}')
        save(out/'graph-index.json',index);save(out/'coefficient-seal.json',{'graphs':24,'graph_index_sha256':digest(out/'graph-index.json'),
          'builder_packet_sha256':digest(out/'builder-inputs.npz'),'all24_coefficients_sealed_before_mass_c0_reference_loading':True,
          'builder_received_mass_c0_reference_true_faces_or_F_C':False})
        del b;gc.collect();guard('all24-coefficients-sealed-before-state')
        sources=[]
        for name,p in COPIES.items():
            rel=str(p.relative_to(ROOT));assert digest(p)==prior[rel],rel;shutil.copyfile(p,out/name)
            sources.append({'copy':name,'source':rel,'sha256':digest(p)})
        save(out/'source-copies.json',sources)
        d=arrays(out/'original-comparison.npz');data=native_data(d)
        u=data[64];np.savez_compressed(out/'union-definition.npz',**u,
          n128_mass_sum=group_sum(d['parent_mass'],64,2),n256_mass_sum=group_sum(d['mass'],64,4))
        runs,paths=load_old(out)
        for K in (8,16):
            r,p=native_run(out,K,u,index,guard);runs[64,K]=r;paths[64,K]=p
            save(out/f'k{K}/native-results.json',r);np.savez_compressed(out/f'k{K}/trajectory-states.npz',**p)
        result,common=compare(data,runs,paths);save(out/'results.json',result);np.savez_compressed(out/'common-observations.npz',**common)
        guard('union64-comparison-saved')
        save(out/'stage3-status.json',{'updated':'2026-11-02','timezone':'Asia/Shanghai','previous_recorded_checks':453,'new_software_checks':4,
          'recorded_checks_total':457,'previous_checks_rerun':0,'new_candidate_graphs':24,'new_candidate_transient_runs':2,
          'new_independent_uniformization_sequences':2,'new_reference_PDE_solves':0,'old_candidate_reruns':0,'new_spatial_levels':[64],
          'registered_N64_union_control_executed':True,'source_gate_already_executed':True,'comparison_decision':result['comparison_decision'],
          'no_automatic_K32_or_grid_extension':True,'shock_extension_deferred':True,'general_smooth_flow_admitted':False,
          'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,
          'pdf_generated':False,'software_release_created':False,'published':False})
        save(out/'execution.json',{'tests_exit_code':0,'study_exit_code':0,'tests_first_attempt_passed':True,'main_first_attempt_passed':True,
          'elapsed_main_seconds':time.monotonic()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'resource_caps_passed':True,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'threads':4,'numerical_norm_estimator_seed':SEED,'network_used':False,'environment_configuration_changed':False,
          'committed_or_pushed':False,'published':False})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        save(out/'failed-execution.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,
          'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'partial_evidence_preserved':True,'no_state_or_threshold_repair':True});raise
    finally:signal.alarm(0)


def replay(directory):
    out=Path(directory).resolve();data=native_data(arrays(out/'original-comparison.npz'));runs,paths=load_old(out);index=read(out/'graph-index.json')
    saved=arrays(out/'union-definition.npz')
    for k,v in data[64].items():assert np.array_equal(v,saved[k]),k
    for K in (8,16):
        r,p=native_run(out,K,data[64],index,replay=True);assert r==read(out/f'k{K}/native-results.json')
        old=arrays(out/f'k{K}/trajectory-states.npz')
        for k,v in p.items():assert np.array_equal(v,old[k]),k
        runs[64,K]=r;paths[64,K]=p
    r,c=compare(data,runs,paths);assert r==read(out/'results.json')
    saved=arrays(out/'common-observations.npz')
    for k,v in c.items():assert np.array_equal(v,saved[k]),k
    print('Sealed24 graphs: actual N64 K8/16 plus independent complete sequences replayed bitwise; original128/256 paths reused; all common/native/H comparisons reproduced. No geometry/PDE/old candidate rerun, no normalization or ORC.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--tests');p.add_argument('--test-log');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    elif a.output and a.tests and a.test_log:main(a.output,a.tests,a.test_log)
    else:p.error('--replay or --output/--tests/--test-log required')
