"""Independent exact local rates, complete paths and fixed compression comparison."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from compression_probe import make_compression_samples
from compression_voronoi_study import SOURCE,COPIES,original_data,compare
from shear_union64_study import arrays,classify


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');ex=read(out/'execution.json');status=read(out/'stage3-status.json')
    lock=read(out/'implementation-lock.json');index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json')
    assert config['parent_manifest_entries_verified']==2331 and config['recorded_checks_total']==461
    assert read(out/'tests.json')=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed'] and ex['elapsed_main_seconds']<2000 and ex['peak_RSS_MiB']<1024
    assert ex['numerical_norm_estimator_seed']==20261101
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/compression_voronoi_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    assert seal['graphs']==len(index)==28 and seal['graph_index_sha256']==digest(out/'graph-index.json')
    assert seal['all28_coefficients_sealed_before_mass_c0_reference_or_kernel_state_loading'] and not seal['builder_received_mass_c0_reference_true_faces_or_F_C']
    assert seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==28 and geo['all28_pure_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_PDE_output_loaded'] and not geo['geometry_builder_rerun']
    for name,p in COPIES.items():assert digest(out/name)==digest(p),name
    ref,old=original_data(out);labels=arrays(out/'trajectory-source.npz');b=arrays(out/'builder-inputs.npz')
    m,c0,ids=ref['mass'],ref['c0'],ref['ids'];n=len(m);assert n==128**2
    for name,key in [('labels','material_label_centers'),('ids','ids'),('rho0_point','rho0_point')]:np.testing.assert_array_equal(labels[name],old[key])
    np.testing.assert_array_equal(labels['labels'],ref['material_label_centers']);np.testing.assert_array_equal(ids,b['ids'])
    np.testing.assert_array_equal(labels['rho0_point'],1+.25*np.cos(2*np.pi*labels['labels'][:,0]))
    for K in (4,8,16):
        expected,_,_=make_compression_samples(labels['labels'],ids,m,c0,labels['rho0_point'],b[f'k{K}_time'])
        np.testing.assert_array_equal(expected.positions,b[f'k{K}_positions']);np.testing.assert_array_equal(expected.rho,b[f'k{K}_rho'])
        del expected
        for key in ('time','positions','rho'):
            np.testing.assert_array_equal(b[f'k{K}_{key}'],b[f'k16_{key}'][::(16//K)])
            if K in (4,8):np.testing.assert_array_equal(b[f'k{K}_{key}'],old[f'k{K}_{key}'])
    E0=float(m@((c0-float(m@c0/m.sum()))**2));sigma=math.sqrt(E0);rel=lambda v:float(np.sqrt(m@(v*v)))/sigma
    H=lambda u:1-float(m@((u-float(m@u/m.sum()))**2))/E0
    runs={};paths={};max_action=max_control=max_raw=0.
    for K in (4,8,16):
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
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_compression_voronoi_geometry.py')
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
    old_result=read(out/'original-kernel-results.json');result,vectors=compare(ref,old,old_result,runs,paths)
    assert result==read(out/'results.json');saved=arrays(out/'comparison-vectors.npz')
    for key,v in vectors.items():np.testing.assert_array_equal(v,saved[key])
    for k,row in enumerate(result['comparison_observations']):
        r=ref['best_projected_states'][k];fv=paths[16]['states'][8*k];background=old['k8_states'][4*k]
        assert rel(fv-r)==row['FV16_error'] and rel(background-r)==row['saved_kernel8_error']
        direct=two_half=localH=0.;details=0
        for K in (4,8):
            for step in range(k*K//2):
                u=old[f'k{K}_states'][step+1]
                for field in ('direct_endpoints','two_half_endpoints'):
                    v=old[f'k{K}_{field}'][step];delta=rel(v-u)*sigma
                    if field=='direct_endpoints':direct+=delta
                    else:two_half+=delta
                    localH+=abs(H(v)-H(u));details+=1
        reported=row['saved_kernel_local_controls'];assert details==len(reported['details'])
        assert abs((direct+two_half)-reported['absolute_sum'])<64*np.finfo(float).eps*max(1.,reported['absolute_sum'])
        assert abs(localH-reported['local_H_absolute_sum'])<64*np.finfo(float).eps
        assert not reported['original_local_checks_are_independent_complete_paths']
        # Independently rebuild every state-indicator component in its physical norm.
        numerical=observed=observedH=0.
        for K in (8,16):
            i=k*K//2
            if i:
                step=runs[K]['steps'][i-1];numerical+=step['cumulative_machine_indicator_absolute']+step['cumulative_tail_indicator_absolute']
                observed+=rel(paths[K]['states'][i]-paths[K]['independent_states'][i])
                observedH+=abs(H(paths[K]['states'][i])-H(paths[K]['independent_states'][i]))
        independent=[rel(paths[8]['states'][4*k]-fv),rel(old['k4_states'][2*k]-background),
          2*rel(ref['space_difference'][k]),2*rel(ref['time_fine_difference'][k]),
          rel(ref['best_projected_states'][0]-c0),numerical/sigma,observed,reported['absolute_sum']/sigma]
        np.testing.assert_array_equal(independent,list(row['state_indicator_components'].values()))
        a=numerical+reported['absolute_sum']
        independentH=[abs(H(paths[8]['states'][4*k])-H(fv)),abs(H(old['k4_states'][2*k])-H(background)),
          2*abs(H(ref['n512-k256_projected_states'][k])-H(ref['n1024-k256_projected_states'][k])),
          2*abs(H(ref['n1024-k512_projected_states'][k])-H(r)),abs(H(ref['best_projected_states'][0])-H(c0)),
          observedH,localH,(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*a+a*a)/E0]
        np.testing.assert_allclose(independentH,list(row['H_indicator_components'].values()),rtol=64*np.finfo(float).eps,atol=1e-30)
        for label in ('state','H'):
            assert all(v>=0 for v in row[label+'_indicator_components'].values())
            assert sum(row[label+'_indicator_components'].values())==row[label+'_indicator']
            assert classify(row[label+'_gain'],row[label+'_indicator'])==row[label+'_decision']
        time=result['time_controls'][k]
        assert time['FV4_8_state_difference']==rel(paths[4]['states'][2*k]-paths[8]['states'][4*k])
        assert time['FV8_16_state_difference']==rel(paths[8]['states'][4*k]-fv)
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==84 and status['recorded_checks_total']==461 and status['new_candidate_graphs']==28
    assert status['registered_compression_time_control_executed'] and not status['fluid_orc_enabled'];save(out/'preservation.json',preserved)
    summary={'all28_geometries_exact_rates_and_complete_raw_sequences_verified':True,
      'original_producer_and_all_nested_K4_K8_inputs_reproduced_bitwise':True,'original_mass_c0_IDs_and_references_unchanged':True,
      'both_original_kernel_local_action_checks_retained_and_independently_recomputed':True,
      'fixed_positive_sum_comparison_and_time_controls_verified':True,'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,
      'maximum_primary_control_relative_difference':max_control,'maximum_native_raw_mass_drift':max_raw,
      'comparison_decision':result['comparison_decision'],'new_software_checks':4,'recorded_checks_total':461,
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':84,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
