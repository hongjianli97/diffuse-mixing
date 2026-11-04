"""Independent exact moving geometry/rates/action and coefficient-Duhamel audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from t2_translation_study import SOURCE,COPIES,inputs
from t2_translation_core import compare
from shear_union64_study import arrays


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');config=read(out/'config.json')
    assert ex['main_first_attempt_passed'] and not ex['tests_first_attempt_passed'] and ex['tests_failed_attempts_preserved']==1
    assert ex['elapsed_main_seconds']<300 and ex['peak_RSS_MiB']<1024 and config['parent_manifest_entries_verified']==3592
    assert read(out/'tests.json')=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    assert read(out/'tests-first-attempt.json')=={'successful':False,'checks_run':2,'failures':1,'errors':0,'skipped':0}
    assert status['recorded_checks_total']==472 and not status['fluid_orc_enabled'] and not status['general_T2_admitted']
    lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/t2_translation_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for name,p in COPIES.items():assert digest(out/name)==digest(p),name
    assert digest(out/'original-pure-inputs.npz')==digest(SOURCE/'builder-inputs.npz')
    seal=read(out/'coefficient-seal.json');index=read(out/'graph-index.json')
    assert seal['graphs']==len(index)==16 and seal['all16_coefficients_sealed_before_mass_c0_reference_static_geometry_or_state']
    assert not seal['builder_received_true_material_faces_F_C_or_scalar']
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==16 and geo['all16_pure_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_PDE_output_loaded'] and not geo['geometry_builder_rerun']
    b=arrays(out/'builder-inputs.npz');pure=arrays(out/'original-pure-inputs.npz');data,old_run,old_path=inputs(out)
    np.testing.assert_array_equal(b['labels'],pure['n32_positions']);np.testing.assert_array_equal(b['ids'],pure['n32_ids'])
    np.testing.assert_array_equal(b['time'],[float(F(j,640)) for j in range(33)])
    np.testing.assert_array_equal(b['rho'],np.broadcast_to(pure['n32_rho'],(33,1024)))
    for k,t in enumerate(b['time']):
        expected=np.column_stack(((pure['n32_positions'][:,0]+.2*t)%1.,(pure['n32_positions'][:,1]-.1*t)%1.))
        np.testing.assert_array_equal(expected,b['positions'][k])
    mp,cp=data['mass'],data['c0'];sigma=math.sqrt(float(mp@((cp-float(mp@cp/mp.sum()))**2)))
    paths={(32,16):arrays(out/'k16/trajectory-states.npz')};runs={(32,16):read(out/'k16/native-results.json')}
    max_action=max_control=max_raw=0.
    for K in (16,):
        path=paths[32,K];run=runs[32,K];mm=tuple(F(float(v)) for v in mp);dt=F(1,20*K)
        np.testing.assert_array_equal(path['states'][0],cp);np.testing.assert_array_equal(path['independent_states'][0],cp)
        last=cp.copy();cum_machine=cum_tail=0.;previous_variance=float(mp@((cp-float(mp@cp/mp.sum()))**2))
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];row=run['steps'][step]
            assert (item['N'],item['K'],item['step'],item['frame'])==(32,K,step,2*step+1)
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            assert item['midpoint_time_saved']==float(b['midpoint_time'][step])==row['midpoint_time_saved']
            receipt=read(out/f'k{K}/geometry-only-audit-step{step:02d}.json')
            assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==seal['builder_packet_sha256']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py')
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_t2_translation_geometry.py')
            assert receipt['pure_geometry_only_no_mass_c0_reference_loaded']
            assert receipt['certificate']['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
            packet=read(out/item['file']);edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
            np.testing.assert_array_equal(packet['ids'],b['ids']);del packet;gc.collect();degree=[F(0)]*1024
            for (i,j),w in zip(edges,g):degree[i]+=w;degree[j]+=w
            Lambda=math.ceil(max(a/v for a,v in zip(degree,mm)));assert Lambda==row['operator']['Lambda_exact_integer']
            adj=[{} for _ in mp]
            for (i,j),w in zip(edges,g):
                assert j not in adj[i] and i not in adj[j]
                adj[i][j]=float(w/(Lambda*mm[i]));adj[j][i]=float(w/(Lambda*mm[j]))
            for i,(w,v) in enumerate(zip(degree,mm)):
                diag=1-w/(Lambda*v);assert diag>=0;adj[i][i]=float(diag)
            ptr=[0];cols=[];vals=[]
            for a in adj:
                for j in sorted(a):cols.append(j);vals.append(a[j])
                ptr.append(len(cols))
            P=csr_matrix((vals,cols,ptr),shape=(1024,1024));act=arrays(out/f'k{K}/action-step{step:02d}.npz')
            for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(v,act[key])
            gf=np.array(list(map(float,g)));np.testing.assert_array_equal(gf,act['g_float']);weights=act['weights']
            A=csr_matrix((act['symmetric_data'],act['symmetric_indices'],act['symmetric_indptr']),shape=tuple(act['shape']))
            ze=Lambda*dt;M=len(weights)-1;assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(ze) and row['poisson']['M']==M
            tail=ze**(M+1)/math.factorial(M+1)/(1-ze/(M+2));assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            ratio=ze/(M+1);assert ratio>=1 or ze**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            degree_max=int(np.bincount(edges.ravel(),minlength=1024).max())
            machine=float(64*np.finfo(float).eps*(degree_max+1)*(M+1)*(1+float(ze))*max(1.,math.sqrt(float(mp.sum())))*max(1.,float(np.max(abs(cp)))))
            cum_machine+=machine;cum_tail+=float(tail)*math.sqrt(float(mp.sum()))*float(np.max(abs(cp)))
            assert machine==row['machine_indicator_absolute_this_interval'] and cum_machine==row['cumulative_machine_indicator_absolute'] and cum_tail==row['cumulative_tail_indicator_absolute']
            v=last.copy();w=math.exp(-float(ze));following=w*v;assert weights[0]==w
            for j in range(1,M+1):v=P@v;w*=float(ze)/j;assert weights[j]==w;following+=w*v
            np.testing.assert_array_equal(following,path['independent_states'][step+1]);last=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            delta=float(np.sqrt(mp@((path['states'][step+1]-following)**2)))
            assert delta==row['primary_control_mass_norm_difference'] and delta<=cum_machine+cum_tail;max_control=max(max_control,delta/sigma)
            u=path['states'][step];i,j=edges.T;net=np.zeros(1024);flux=gf*(u[j]-u[i]);np.add.at(net,i,flux);np.add.at(net,j,-flux);root=np.sqrt(mp)
            diff=float(np.sqrt(mp@((A@(root*u)/root-net/mp)**2)))
            assert diff<1e-10*max(1.,float(np.sqrt(mp@((net/mp)**2))));max_action=max(max_action,diff)
            for u,chi in zip((path['states'][step],path['exponential_half_states'][step],path['states'][step+1]),row['chi_start_half_end']):
                assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
                assert abs(-float((root*u)@(A@(root*u)))-chi)<1e-10*max(1.,chi)
            u=path['states'][step+1];drift=float(mp@(u-cp));assert drift==row['raw_mass_change_from_c0'];assert abs(drift)/float(mp@cp)<1e-10
            assert u.min()>=cp.min()-1e-10 and u.max()<=cp.max()+1e-10;max_raw=max(max_raw,abs(drift))
            variance=float(mp@((u-float(mp@u/mp.sum()))**2));assert variance<=previous_variance+1e-10*sigma*sigma;previous_variance=variance
            save(out/f'k{K}/action-audit-step{step:02d}.json',{'cache_sha256':item['sha256'],
              'exact_rates_whole_control_sequence_and_minimal_raw_tail_verified':True,'symmetric_vs_shared_action_mass_norm_difference':diff})
            print(f'independent action:k{K} step{step+1}/{K}',flush=True);del A,P,adj,g,degree,act;gc.collect()
    result=read(out/'results.json');saved=read(out/'coefficient-perturbations.json')
    initial_variance=sum((F(float(m))*F(float(c))**2 for m,c in zip(mp,cp)),F(0))-sum((F(float(m))*F(float(c)) for m,c in zip(mp,cp)),F(0))**2/sum(map(lambda v:F(float(v)),mp),F(0))
    assert initial_variance==F(result['initial_variance_exact'])
    upper=F(result['initial_fluctuation_float_upper']);assert upper**2>=initial_variance
    old=read(out/'static-geometry.json');base={tuple(ij):F(g) for ij,g in zip(old['edges'],old['conductances_exact'])}
    max_relative=F(0);changed_graphs=0;rates=[]
    for step,row in enumerate(saved):
        packet=read(out/index[f'k16-step{step:02d}']['file']);moving={tuple(ij):F(g) for ij,g in zip(packet['edges'],packet['conductances_exact'])}
        differences={edge:abs(moving.get(edge,F(0))-base.get(edge,F(0))) for edge in base.keys()|moving.keys()}
        absolute_rows=[F(0)]*1024
        for (i,j),delta in differences.items():absolute_rows[i]+=delta;absolute_rows[j]+=delta
        eta=2*max(d/F(float(m)) for d,m in zip(absolute_rows,mp));rates.append(eta)
        rel=max((delta/base[e] for e,delta in differences.items() if e in base and base[e]>0),default=F(0))
        same=set(base)==set(moving)
        assert row['delta_generator_mass_norm_upper_exact']==str(eta)
        assert row['maximum_relative_static_face_difference_exact']==str(rel)
        assert row['same_shared_edge_topology']==same
        assert row['original_algebra_scale_passed']==(same and rel<=F(float(1e-10)))
        assert row['changed_shared_edges']==sum(v!=0 for v in differences.values())
        assert row['introduced_shared_edges']==len(moving.keys()-base.keys()) and row['removed_shared_edges']==len(base.keys()-moving.keys())
        max_relative=max(max_relative,rel);changed_graphs+=int(any(differences.values()))
    computed=compare(data,old_run,old_path,runs[32,16],paths[32,16],saved);assert computed==result
    for k,row in enumerate(result['observations']):
        i=8*k;integral=sum(rates[:i],F(0))*F(1,320);bound=integral*upper
        assert F(row['Duhamel_integrated_rate_exact'])==integral and F(row['Duhamel_absolute_upper_exact'])==bound
        assert F(row['state_indicator_components_absolute']['exact_coefficient_Duhamel'])>=bound
        assert row['state_indicator_absolute']==sum(row['state_indicator_components_absolute'].values())
        a=old_path['states'][i];v=paths[32,16]['states'][i];delta=float(np.sqrt(mp@((a-v)**2)))
        assert delta==row['state_difference_absolute'] and row['state_within_indicator']==(delta<=row['state_indicator_absolute'])
        def H(c):return 1-float(mp@((c-float(mp@c/mp.sum()))**2))/(sigma*sigma)
        assert abs(abs(H(v)-H(a))-row['H_difference'])<=16*np.finfo(float).eps
        for name,path,run in [('static',old_path,old_run),('moving',paths[32,16],runs[32,16])]:
            observed=float(np.sqrt(mp@((path['states'][i]-path['independent_states'][i])**2)))
            allowance=0. if i==0 else run['steps'][i-1]['cumulative_machine_indicator_absolute']+run['steps'][i-1]['cumulative_tail_indicator_absolute']
            assert row[name+'_sequence_within_original_allowance']==(observed<=allowance)
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==89;save(out/'preservation.json',preserved)
    summary={'all16_geometries_exact_rates_and_complete_raw_intervals_verified':True,'moving_complete_uniformization_path_bitwise_verified':True,
      'coefficient_perturbations_and_upward_Duhamel_bound_independently_verified':True,'both_sequence_entry_guards_verified':True,
      'original_copies_bitwise_unchanged':True,'graphs_with_nonzero_exact_coefficient_perturbation':changed_graphs,
      'maximum_relative_face_coefficient_difference_exact':str(max_relative),'maximum_relative_face_coefficient_difference':float(max_relative),
      'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,'maximum_primary_control_relative_difference':max_control,
      'maximum_native_raw_mass_drift':max_raw,'comparison_decision':result['comparison_decision'],
      'T2_registered_translation_case_passed':result['T2_registered_translation_case_passed'],'new_software_checks':2,'recorded_checks_total':472,
      'test_first_attempt_failed_checks':1,'first_test_failure_preserved':True,'original_files_verified':782,'historical_attachments_verified':15,
      'prior_formal_manifests_verified':89,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'general_T2_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
