"""Independent N256 exact-rate actions, original material projection and three criteria."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from compression_spatial_fv_study import SOURCE,COPIES,inputs
from compression_spatial_fv_core import compare
from compression_probe import make_compression_samples
from shear_union64_study import arrays,classify


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');config=read(out/'config.json')
    assert config['parent_manifest_entries_verified']==2689 and config['recorded_checks_total']==466
    assert read(out/'tests.json')=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ex['elapsed_main_seconds']<8400 and ex['peak_RSS_MiB']<2048
    assert ex['numerical_norm_estimator_seed']==20261101
    lock=read(out/'implementation-lock.json');index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json')
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/compression_spatial_fv_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md') and read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    assert seal['graphs']==len(index)==48 and seal['graph_index_sha256']==digest(out/'graph-index.json')
    assert seal['all48_coefficients_sealed_before_mass_c0_reference_or_old_state_loading'] and not seal['builder_received_mass_c0_reference_true_faces_or_F_C']
    assert seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==48 and geo['all48_pure_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_PDE_output_loaded'] and not geo['geometry_builder_rerun']
    schedule=read(out/'geometry-scheduling-lock.json');ge=read(out/'geometry-execution.json');workers=schedule['workers_selected']
    assert workers in (1,2,3) and ge['all48_jobs_completed'] and ge['all_packet_creation_serialized']
    assert ge['conservative_main_and_workers_peak_RSS_MiB']<2048 and ge['sampled_main_and_active_workers_peak_RSS_MiB']<2048
    assert not schedule['adaptation_uses_scalar_mass_or_reference']
    for name,p in COPIES.items():assert digest(out/name)==digest(p),name
    d,ref,runs,paths=inputs(out);pure=arrays(out/'trajectory-source.npz');b=arrays(out/'builder-inputs.npz')
    m,c0,ids=d['mass'],d['c0'],d['ids'];n=len(m);assert n==256**2
    np.testing.assert_array_equal(pure['labels'],d['material_label_centers']);np.testing.assert_array_equal(pure['ids'],ids)
    np.testing.assert_array_equal(pure['rho0_point'],d['rho0_point']);np.testing.assert_array_equal(ids,b['ids'])
    for K in (16,32):
        expected,_,_=make_compression_samples(pure['labels'],ids,m,c0,pure['rho0_point'],b[f'k{K}_time'])
        np.testing.assert_array_equal(expected.positions,b[f'k{K}_positions']);np.testing.assert_array_equal(expected.rho,b[f'k{K}_rho']);del expected
        for key in ('time','positions','rho'):np.testing.assert_array_equal(b[f'k{K}_{key}'],b[f'k32_{key}'][::(32//K)])
    for K in (4,8):
        for key in ('time','positions','rho'):np.testing.assert_array_equal(d[f'k{K}_{key}'],b[f'k32_{key}'][::(32//K)])
    for K in (16,32):
        for step in range(K):
            build=read(out/f'k{K}/geometry-build-step{step:02d}.json')
            assert build['geometry_builder_sha256']==digest(ROOT/'research/reconnect/periodic_voronoi_fv.py')
            assert build['worker_source_sha256']==digest(ROOT/'research/reconnect/spatial_geometry_worker.py')
            assert build['only_pure_point_inputs_received'] and build['coefficient_packet_creation_serialized']
            frame=out/f'k{K}/frame-input-step{step:02d}.npz';assert digest(frame)==build['frame_input_sha256']
            data=arrays(frame);assert set(data)=={'ids','positions','rho'}
            np.testing.assert_array_equal(data['ids'],ids)
            np.testing.assert_array_equal(data['positions'],b[f'k{K}_positions'][2*step+1])
            np.testing.assert_array_equal(data['rho'],b[f'k{K}_rho'][2*step+1])
    E0=float(m@((c0-float(m@c0/m.sum()))**2));sigma=math.sqrt(E0);rel=lambda v:float(np.sqrt(m@(v*v)))/sigma
    max_action=max_control=max_raw=0.
    for K in (16,32):
        path=arrays(out/f'k{K}/trajectory-states.npz');run=read(out/f'k{K}/native-results.json');paths[256,K]=path;runs[256,K]=run
        np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
        assert run['initial_mass_fluctuation_norm']==sigma
        mm=tuple(F(float(v)) for v in m);dt=F(1,20*K);last=c0.copy();cum_machine=cum_tail=0.;previous_variance=E0
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];row=run['steps'][step]
            assert (item['N'],item['K'],item['step'],item['frame'])==(256,K,step,2*step+1)
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            assert item['midpoint_time_saved']==float(b[f'k{K}_midpoint_time'][step])==row['midpoint_time_saved']
            receipt=read(out/f'k{K}/geometry-only-audit-step{step:02d}.json')
            assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==seal['builder_packet_sha256']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py')
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_compression_spatial_fv_geometry.py')
            assert receipt['pure_geometry_only_no_mass_c0_reference_loaded'] and receipt['certificate']['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
            packet=read(out/item['file']);edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
            np.testing.assert_array_equal(packet['ids'],ids);del packet;gc.collect();degree=[F(0)]*n
            for (i,j),w in zip(edges,g):degree[i]+=w;degree[j]+=w
            Lambda=math.ceil(max(a/v for a,v in zip(degree,mm)));assert Lambda==row['operator']['Lambda_exact_integer']
            adj=[{} for _ in m]
            for (i,j),w in zip(edges,g):
                assert j not in adj[i] and i not in adj[j]
                adj[i][j]=float(w/(Lambda*mm[i]));adj[j][i]=float(w/(Lambda*mm[j]))
            for i,(w,v) in enumerate(zip(degree,mm)):
                diag=1-w/(Lambda*v);assert diag>=0;adj[i][i]=float(diag)
            ptr=[0];cols=[];vals=[]
            for a in adj:
                for j in sorted(a):cols.append(j);vals.append(a[j])
                ptr.append(len(cols))
            P=csr_matrix((vals,cols,ptr),shape=(n,n));act=arrays(out/f'k{K}/action-step{step:02d}.npz')
            for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(v,act[key])
            gf=np.array(list(map(float,g)));np.testing.assert_array_equal(gf,act['g_float']);weights=act['weights']
            A=csr_matrix((act['symmetric_data'],act['symmetric_indices'],act['symmetric_indptr']),shape=tuple(act['shape']))
            ze=Lambda*dt;M=len(weights)-1;assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(ze) and row['poisson']['M']==M
            tail=ze**(M+1)/math.factorial(M+1)/(1-ze/(M+2));assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            ratio=ze/(M+1);assert ratio>=1 or ze**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            degree_max=int(np.bincount(edges.ravel(),minlength=n).max())
            machine=float(64*np.finfo(float).eps*(degree_max+1)*(M+1)*(1+float(ze))*max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
            cum_machine+=machine;cum_tail+=float(tail)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
            assert machine==row['machine_indicator_absolute_this_interval'] and cum_machine==row['cumulative_machine_indicator_absolute'] and cum_tail==row['cumulative_tail_indicator_absolute']
            v=last.copy();w=math.exp(-float(ze));following=w*v;assert weights[0]==w
            for j in range(1,M+1):v=P@v;w*=float(ze)/j;assert weights[j]==w;following+=w*v
            np.testing.assert_array_equal(following,path['independent_states'][step+1]);last=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            delta=float(np.sqrt(m@((path['states'][step+1]-following)**2)))
            assert delta==row['primary_control_mass_norm_difference'] and delta<=cum_machine+cum_tail;max_control=max(max_control,delta/sigma)
            u=path['states'][step];i,j=edges.T;net=np.zeros(n);flux=gf*(u[j]-u[i]);np.add.at(net,i,flux);np.add.at(net,j,-flux);root=np.sqrt(m)
            diff=float(np.sqrt(m@((A@(root*u)/root-net/m)**2)));assert diff<1e-10*max(1.,float(np.sqrt(m@((net/m)**2))));max_action=max(max_action,diff)
            for u,chi in zip((path['states'][step],path['exponential_half_states'][step],path['states'][step+1]),row['chi_start_half_end']):
                assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
                assert abs(-float((root*u)@(A@(root*u)))-chi)<1e-10*max(1.,chi)
            u=path['states'][step+1];drift=float(m@(u-c0));assert drift==row['raw_mass_change_from_c0'];assert abs(drift)/float(m@c0)<1e-10
            assert u.min()>=c0.min()-1e-10 and u.max()<=c0.max()+1e-10;max_raw=max(max_raw,abs(drift))
            variance=float(m@((u-float(m@u/m.sum()))**2));assert variance<=previous_variance+1e-10*E0;previous_variance=variance
            save(out/f'k{K}/action-audit-step{step:02d}.json',{'cache_sha256':item['sha256'],
              'exact_rates_whole_control_sequence_and_minimal_raw_tail_verified':True,'symmetric_vs_shared_action_mass_norm_difference':diff})
            print(f'independent action:k{K} step{step+1}/{K}',flush=True);del A,P,adj,g,degree,act;gc.collect()
    result,common=compare(d,ref,runs,paths);assert result==read(out/'results.json')
    saved=arrays(out/'common-observations.npz')
    for key,v in common.items():np.testing.assert_array_equal(v,saved[key])
    mp,cp=d['parent_mass'],d['parent_c0'];pE0=float(mp@((cp-float(mp@cp/mp.sum()))**2));ps=math.sqrt(pE0)
    norm=lambda mass,v:float(np.sqrt(mass@(v*v)))/ps;cn=lambda v:norm(mp,v)
    H=lambda u:1-float(mp@((u-float(mp@u/mp.sum()))**2))/pE0
    max_projection=0.
    for K in (16,32):
        for field in ('states','independent_states'):
            for native,projected in zip(paths[256,K][field],common[f'fine_k{K}_{field}']):
                other=np.bincount(d['parent_index'],weights=m*native,minlength=len(mp))/mp
                delta=float(np.sqrt(mp@((other-projected)**2)))
                assert delta<64*np.finfo(float).eps*max(1.,float(np.sqrt(m@(native*native))));max_projection=max(max_projection,delta)
    reference_delta=common['reference_projection_difference']-d['reference_parent_projection_difference']
    ref_replication=max(cn(v) for v in reference_delta)
    assert ref_replication<64*np.finfo(float).eps
    for k,row in enumerate(result['comparison_observations']):
        coarse=paths[128,32]['states'][16*k];fine=common['fine_k32_states'][16*k];r=ref['best_projected_states'][k]
        assert row['coarse32_error_common']==cn(coarse-r) and row['fine32_error_common']==cn(fine-r)
        assert row['native_coarse32_error']==norm(mp,coarse-r)
        assert row['native_fine32_error']==norm(m,paths[256,32]['states'][16*k]-d['reference_n1024-k1024_states'][k])
        common_allow=native_allow=obs_common=obs_native=obsH=0.
        factor=result['projection_solver_norm_factor']
        for N,mass,fac in [(128,mp,1.),(256,m,factor)]:
            for K in (16,32):
                i=k*K//2
                if not i:continue
                step=runs[N,K]['steps'][i-1];allow=step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute']
                native_allow+=allow;common_allow+=fac*allow
                u,v=paths[N,K]['states'][i],paths[N,K]['independent_states'][i];obs_native+=norm(mass,u-v)
                a=u if N==128 else common[f'fine_k{K}_states'][i]
                c=v if N==128 else common[f'fine_k{K}_independent_states'][i];obs_common+=cn(a-c);obsH+=abs(H(a)-H(c))
        dc=paths[128,16]['states'][8*k]-coarse;df=paths[256,16]['states'][8*k]-paths[256,32]['states'][16*k]
        projected_df=common['fine_k16_states'][8*k]-fine
        refspace=d['reference_n512-k256_states'][k]-d['reference_n1024-k256_states'][k]
        reftime=d['reference_n1024-k512_states'][k]-d['reference_n1024-k1024_states'][k]
        expected_common=[cn(dc),cn(projected_df),2*cn(ref['space_difference'][k]),2*cn(ref['time_fine_difference'][k]),0.,
          cn(common['reference_projection_difference'][k]),sum(cn(v) for v in common['initial_bridges']),common_allow/ps,obs_common]
        np.testing.assert_array_equal(expected_common,list(row['common_indicator_components'].values()))
        expected_native=[norm(mp,dc),norm(m,df),2*(norm(mp,ref['space_difference'][k])+norm(mp,ref['time_fine_difference'][k])+norm(m,refspace)+norm(m,reftime)),
          norm(mp,ref['best_projected_states'][0]-cp)+norm(m,d['reference_n1024-k1024_states'][0]-d['c0']),native_allow/ps,obs_native]
        np.testing.assert_allclose(expected_native,list(row['native_indicator_components'].values()),rtol=64*np.finfo(float).eps,atol=1e-30)
        hbridge=abs(H(ref['best_projected_states'][0])-H(cp))+abs(H(common['fine_c0_common'])-H(cp))+abs(H(common['reference_fine_common'][0])-H(common['fine_c0_common']))
        hp=[abs(H(paths[128,16]['states'][8*k])-H(coarse)),abs(H(common['fine_k16_states'][8*k])-H(fine)),
          2*abs(H(ref['n512-k256_projected_states'][k])-H(ref['n1024-k256_projected_states'][k])),
          2*abs(H(ref['n1024-k512_projected_states'][k])-H(r)),0.,abs(H(common['reference_fine_common'][k])-H(r)),
          hbridge,obsH,(2*math.sqrt(float(mp.sum()))*max(1.,float(np.max(abs(cp))))*common_allow+common_allow**2)/pE0]
        np.testing.assert_array_equal(hp,list(row['H_indicator_components'].values()))
        for label in ('common','native','H'):
            assert all(v>=0 for v in row[label+'_indicator_components'].values())
            assert sum(row[label+'_indicator_components'].values())==row[label+'_indicator']
            assert classify(row[label+'_gain'],row[label+'_indicator'])==row[label+'_decision']
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==86 and status['recorded_checks_total']==466 and status['new_candidate_graphs']==48
    assert status['registered_N256_compression_control_executed'] and not status['fluid_orc_enabled'];save(out/'preservation.json',preserved)
    summary={'all48_geometries_exact_rates_and_complete_raw_sequences_verified':True,
      'only_pure_points_in_every_worker_and_scalar_separation_verified':True,'all_original_coarse_states_masses_initial_values_and_references_unchanged':True,
      'raw_material_projection_independently_bincount_checked':True,'fixed_common_native_H_indicators_independently_checked':True,
      'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,'maximum_primary_control_relative_difference':max_control,
      'maximum_fine_raw_mass_drift':max_raw,'maximum_independent_projection_mass_norm_difference':max_projection,
      'reference_replication_difference_relative_mass_norm':ref_replication,'comparison_decision':result['comparison_decision'],
      'geometry_workers_selected':workers,'aggregate_geometry_resources_verified':True,'new_software_checks':3,'recorded_checks_total':466,
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':86,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)

