"""Independent exact-rate32 path audit; same independently derived rate audit as prior case."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from compression_probe import make_compression_samples
from compression_time32_study import SOURCE,COPIES,load_previous,compare
from compression_voronoi_study import original_data
from shear_union64_study import arrays,classify


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');ex=read(out/'execution.json');status=read(out/'stage3-status.json')
    assert config['parent_manifest_entries_verified']==2500 and config['recorded_checks_total']==463
    assert read(out/'tests.json')=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    first=read(out/'tests-first-attempt.json');assert not first['successful'] and first['failures']==1 and first['errors']==0
    assert ex['main_first_attempt_passed'] and not ex['tests_first_attempt_passed']
    assert ex['elapsed_main_seconds']<1400 and ex['peak_RSS_MiB']<1024 and ex['numerical_norm_estimator_seed']==20261101
    lock=read(out/'implementation-lock.json');index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json')
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/compression_time32_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    assert len(index)==seal['graphs']==32 and seal['graph_index_sha256']==digest(out/'graph-index.json')
    assert seal['all32_coefficients_sealed_before_mass_c0_reference_or_old_state_loading'] and not seal['builder_received_mass_c0_reference_true_faces_or_F_C']
    assert seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==32 and geo['all32_pure_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_PDE_output_loaded'] and not geo['geometry_builder_rerun']
    for name,p in COPIES.items():assert digest(out/name)==digest(p),name
    ref,old=original_data(out);pure=arrays(out/'trajectory-source.npz');b=arrays(out/'builder-inputs.npz')
    assert digest(out/'trajectory-source.npz')==digest(SOURCE/'trajectory-source.npz')
    m,c0,ids=ref['mass'],ref['c0'],ref['ids'];n=len(m);assert n==128**2
    np.testing.assert_array_equal(pure['labels'],ref['material_label_centers']);np.testing.assert_array_equal(pure['ids'],ids)
    np.testing.assert_array_equal(pure['rho0_point'],old['rho0_point']);np.testing.assert_array_equal(ids,b['ids'])
    expected,_,_=make_compression_samples(pure['labels'],ids,m,c0,pure['rho0_point'],b['k32_time'])
    np.testing.assert_array_equal(expected.positions,b['k32_positions']);np.testing.assert_array_equal(expected.rho,b['k32_rho']);del expected
    previous=arrays(SOURCE/'builder-inputs.npz')
    for K in (4,8,16):
        for key in ('time','positions','rho'):np.testing.assert_array_equal(previous[f'k{K}_{key}'],b[f'k32_{key}'][::(32//K)])
    del previous
    E0=float(m@((c0-float(m@c0/m.sum()))**2));sigma=math.sqrt(E0);rel=lambda v:float(np.sqrt(m@(v*v)))/sigma
    H=lambda u:1-float(m@((u-float(m@u/m.sum()))**2))/E0
    runs,paths=load_previous(out);max_action=max_control=max_raw=0.
    for K in (32,):
        path=arrays(out/f'k{K}/trajectory-states.npz');run=read(out/f'k{K}/native-results.json');paths[K]=path;runs[K]=run
        np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
        assert run['initial_mass_fluctuation_norm']==sigma
        mm=tuple(F(float(v)) for v in m);dt=F(1,20*K);last=c0.copy();cum_machine=cum_tail=0.;previous_variance=E0
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];row=run['steps'][step]
            assert (item['N'],item['K'],item['step'],item['frame'])==(128,K,step,2*step+1)
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            assert item['midpoint_time_saved']==float(b[f'k{K}_midpoint_time'][step])==row['midpoint_time_saved']
            receipt=read(out/f'k{K}/geometry-only-audit-step{step:02d}.json')
            assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==seal['builder_packet_sha256']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py')
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_compression_time32_geometry.py')
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
    result,vectors=compare(ref,old,read(out/'original-kernel-results.json'),runs,paths)
    assert result==read(out/'results.json');saved=arrays(out/'comparison-vectors.npz')
    for key,v in vectors.items():np.testing.assert_array_equal(v,saved[key])
    original=read(out/'source-results.json')
    for K in (4,8,16):assert result['observations_by_intervals'][str(K)]==original['observations_by_intervals'][str(K)]
    for k,row in enumerate(result['comparison_observations']):
        r=ref['best_projected_states'][k];u16=paths[16]['states'][8*k];u32=paths[32]['states'][16*k];bg=old['k8_states'][4*k]
        assert row['FV32_error']==rel(u32-r) and row['saved_kernel8_error']==rel(bg-r)
        numerical=observed=observedH=local_abs=localH=0.;count=0
        for K in (16,32):
            i=k*K//2
            if i:
                step=runs[K]['steps'][i-1];numerical+=step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute']
                observed+=rel(paths[K]['states'][i]-paths[K]['independent_states'][i])
                observedH+=abs(H(paths[K]['states'][i])-H(paths[K]['independent_states'][i]))
        for K in (4,8):
            for field in ('direct_endpoints','two_half_endpoints'):
                for step in range(k*K//2):
                    v=old[f'k{K}_{field}'][step];u=old[f'k{K}_states'][step+1]
                    local_abs+=float(np.sqrt(m@((v-u)**2)));localH+=abs(H(v)-H(u));count+=1
        assert local_abs==row['saved_kernel_local_controls']['absolute_sum'] and count==len(row['saved_kernel_local_controls']['details'])
        state_parts=[rel(u16-u32),rel(old['k4_states'][2*k]-bg),2*rel(ref['space_difference'][k]),2*rel(ref['time_fine_difference'][k]),
          rel(ref['best_projected_states'][0]-c0),numerical/sigma,observed,local_abs/sigma]
        np.testing.assert_array_equal(state_parts,list(row['state_indicator_components'].values()))
        a=numerical+local_abs
        hparts=[abs(H(u16)-H(u32)),abs(H(old['k4_states'][2*k])-H(bg)),
          2*abs(H(ref['n512-k256_projected_states'][k])-H(ref['n1024-k256_projected_states'][k])),
          2*abs(H(ref['n1024-k512_projected_states'][k])-H(r)),abs(H(ref['best_projected_states'][0])-H(c0)),
          observedH,localH,(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*a+a*a)/E0]
        np.testing.assert_array_equal(hparts,list(row['H_indicator_components'].values()))
        for label in ('state','H'):
            assert all(v>=0 for v in row[label+'_indicator_components'].values())
            assert sum(row[label+'_indicator_components'].values())==row[label+'_indicator']
            assert classify(row[label+'_gain'],row[label+'_indicator'])==row[label+'_decision']
        time=result['time_controls'][k]
        assert time['FV16_32_state_difference']==rel(u16-u32) and time['FV8_16_state_difference']==original['time_controls'][k]['FV8_16_state_difference']
        assert time['FV4_8_state_difference']==original['time_controls'][k]['FV4_8_state_difference']
        assert time['state_difference_decreased']==(time['FV16_32_state_difference']<time['FV8_16_state_difference'])
        assert time['H_difference_decreased']==(abs(H(u16)-H(u32))<abs(H(paths[8]['states'][4*k])-H(u16)))
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==85 and status['recorded_checks_total']==463 and status['new_candidate_graphs']==32
    assert status['registered32_interval_control_executed'] and not status['fluid_orc_enabled'];save(out/'preservation.json',preserved)
    summary={'all32_geometries_exact_rates_and_complete_raw_sequence_verified':True,
      'old4_8_16_inputs_and_observations_reproduced_bitwise_without_candidate_reruns':True,
      'original_mass_c0_reference_and_old_solver_controls_unchanged':True,'fixed16_32_indicators_independently_recomputed':True,
      'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,'maximum_primary_control_relative_difference':max_control,
      'maximum_native_raw_mass_drift':max_raw,'comparison_decision':result['comparison_decision'],
      'new_software_checks':2,'recorded_checks_total':463,'initial_test_expectation_failure_retained':True,
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':85,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)

