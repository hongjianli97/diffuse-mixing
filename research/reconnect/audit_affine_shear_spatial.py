"""Independent new-grid world geometry, raw sequences, references and fixed-parent budgets."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,math,resource,time
import mpmath as mp
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save as raw_save,digest,preservation
from affine_shear_spatial_study import SOURCE,NEW,LEVELS,load_all,check_references
from affine_shear_spatial_core import compare
from affine_shear_completion import plain_json
from audit_affine_shear import all_cell_world_halfplanes
from shear_union64_study import arrays
from t2_d0_core import bitwise_same
from voronoi_time_action import metrics
COUNTS=(4,8,16)


def save(path,obj):raw_save(path,plain_json(obj))


def independent_references(out,data,labels,world,N):
    m,c0=data['mass'],data['c0'];p=F(float(.02));saved=arrays(out/'analytic-references.npz');meta=read(out/'analytic-parameters.json')
    R=64*np.finfo(float).eps*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))));assert R==meta['reference_evaluation_indicator_absolute']
    times=(F(0),F(1,40),F(1,20));frozen={}
    for K in COUNTS:
        values=[]
        for stop in (0,K//2,K):
            U=V=F(0)
            for j in range(stop):
                t=F(2*j+1,40*K);s=5*t
                assert world['cases'][str(K)]['frames'][2*j+1]['shear_exact']==str(s)
                U+=F(1,20*K)*p*(1+(1-s)**2);V+=F(1,20*K)*p*s
            values.append((U,V))
        assert [list(map(str,pair)) for pair in values]==meta['frozen_coefficients_exact'][str(K)];frozen[K]=values
    clocks=[(p*(2*t-5*t*t+F(25,3)*t**3),p*(2*t+5*t*t+F(25,3)*t**3)) for t in times]
    grid=[(p*(2*t-5*t*t+F(25,3)*t**3),p*F(5,2)*t*t) for t in times]
    assert [list(map(str,v)) for v in clocks]==meta['physical_clocks_exact']
    assert [list(map(str,v)) for v in grid]==meta['continuous_grid_coefficients_exact']
    rounded={}
    for precision in (80,100):
        with mp.workdps(precision):
            exact=lambda q:mp.mpf(q.numerator)/q.denominator
            def sine(q):
                q=q%1
                if q in (F(0),F(1,2)):return mp.mpf(0)
                if q==F(1,4):return mp.mpf(1)
                if q==F(3,4):return mp.mpf(-1)
                return mp.sin(2*mp.pi*exact(q))
            w=[[sine(F(float(x))+eta*F(float(y))) for x,y in labels] for eta in (1,-1)]
            amplitude=mp.mpf(1)/10*(mp.sin(mp.pi/N)/(mp.pi/N))**2
            phi=[[amplitude*z for z in row] for row in w]
            u=mp.sin(mp.pi/N)**2;v=mp.sin(2*mp.pi/N)**2
            def state(a,b):return [float(mp.mpf('.5')+x*mp.exp(-a)+y*mp.exp(-b)) for x,y in zip(*phi)]
            def gridstate(pair):
                U,V=map(exact,pair);return state(4*N*N*u*U,4*N*N*(u*U+v*V))
            values={'weights':np.array([[float(x) for x in row] for row in w]),'modes':np.array([[float(x) for x in row] for row in phi]),
              'initial':np.array(state(0,0)),'reference':np.array([state(4*mp.pi**2*exact(a),4*mp.pi**2*exact(b)) for a,b in clocks]),
              'continuous_grid_symbol':np.array([gridstate(pair) for pair in grid])}
            for K in COUNTS:values[f'frozen_k{K}']=np.array([gridstate(pair) for pair in frozen[K]])
            rounded[precision]=values
    checks=[];norm=lambda z:float(np.sqrt(m@(z*z)))
    for key,value in rounded[80].items():
        assert bitwise_same(value,rounded[100][key]),key
        original=c0 if key=='initial' else saved[key]
        errors=[norm(a-b) for a,b in zip(np.atleast_2d(value),np.atleast_2d(original))];assert max(errors)<=R,(key,errors,R)
        checks.append({'array':key,'80_100_digit_roundings_bitwise_equal':True,'mass_norm_differences':errors,'fixed_evaluation_allowance_absolute':R})
    np.savez_compressed(out/'independent-analytic80.npz',**rounded[80])
    save(out/'analytic-reference-audit.json',{'independent_two_physical_clocks_finite_means_and_grid_symbols_verified':True,
      'production_reference_or_clock_function_called':False,'precisions_decimal_digits':[80,100],
      'analytic_rational_phase_zero_and_unit_values_exact':True,'actual_numpy_weights_modes_or_states_corrected':False,
      'all_evaluations_within_original_fixed_allowance':True,'checks':checks})


def verify_paths(out,N,d,world,index):
    mass,c0=d['mass'],d['c0'];sigma=math.sqrt(metrics(mass,c0,c0)['variance'])
    seal={'pure_inputs_sha256':digest(out/'pure-inputs.json')}
    runs={};paths={};residues={};max_action=max_control=max_raw=max_mode=0.
    for K in COUNTS:
        folder=out/f'k{K}';path=arrays(folder/'trajectory-states.npz');run=read(folder/'native-results.json')
        paths[K]=path;runs[K]=run;rr=read(folder/'modal-residues.json');residues[K]=[]
        mm=tuple(F(float(v)) for v in mass);dt=F(1,20*K)
        np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
        last=c0.copy();cum_machine=cum_tail=0.;previous_variance=sigma*sigma
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];row=run['steps'][step];frame=world['cases'][str(K)]['frames'][2*step+1]
            assert (item['N'],item['K'],item['step'],item['frame'])==(N,K,step,2*step+1)
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            assert item['midpoint_time_saved']==float(F(frame['time_exact']))==row['midpoint_time_saved']
            receipt=read(folder/f'world-geometry-audit-step{step:02d}.json')
            assert receipt['world_cache_sha256']==item['world_sha256']==digest(out/item['world_file'])
            assert receipt['pure_inputs_sha256']==seal['pure_inputs_sha256']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_sliding_world_geometry.py')
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_affine_shear_spatial_geometry.py')
            assert not receipt['mass_c0_reference_or_oldstates_loaded'] and not receipt['original_or_sliding_geometry_builder_called_by_auditor']
            assert receipt['certificate']['all_world_vertices_rebuilt_directly_from_world_bisectors']
            packet=read(out/item['file']);wp=read(out/item['world_file'])
            for field in ('ids','edges','image_shifts','cell_vertex_indices','cell_incoming_labels','cell_areas_exact','face_vertex_indices','face_over_distance_exact','conductances_exact','D_exact'):assert packet[field]==wp[field],field
            np.testing.assert_array_equal(packet['ids'],world['ids'])
            edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
            halfplanes=all_cell_world_halfplanes(wp,frame)
            save(folder/f'all-cell-halfplanes-step{step:02d}.json',halfplanes)
            cert=read(folder/f'boundary-step{step:02d}.json')
            assert cert['world_euclidean_geometry_preserved'] and not cert['s0_original_rectangle_dispatched']
            assert not cert['mass_c0_reference_internal_faces_or_F_C_received'] and not cert['points_wrapped_jittered_or_fitted']
            assert read(folder/f'direction-step{step:02d}.json')['all_geometry_controls_passed']
            del packet,wp;degree=[F(0)]*(N*N)
            for (i,j),w in zip(edges,g):degree[i]+=w;degree[j]+=w
            Lambda=math.ceil(max(v/m for v,m in zip(degree,mm)));assert Lambda==row['operator']['Lambda_exact_integer']
            adj=[{} for _ in mass]
            for (i,j),w in zip(edges,g):
                assert j not in adj[i] and i not in adj[j]
                adj[i][j]=float(w/(Lambda*mm[i]));adj[j][i]=float(w/(Lambda*mm[j]))
            for i,(w,m) in enumerate(zip(degree,mm)):
                diag=1-w/(Lambda*m);assert diag>=0;adj[i][i]=float(diag)
            ptr=[0];cols=[];vals=[]
            for neighbors in adj:
                for j in sorted(neighbors):cols.append(j);vals.append(neighbors[j])
                ptr.append(len(cols))
            P=csr_matrix((vals,cols,ptr),shape=(N*N,N*N));act=arrays(folder/f'action-step{step:02d}.npz')
            for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(v,act[key])
            gf=np.array(list(map(float,g)));np.testing.assert_array_equal(gf,act['g_float']);weights=act['weights']
            A=csr_matrix((act['symmetric_data'],act['symmetric_indices'],act['symmetric_indptr']),shape=tuple(act['shape']))
            ze=Lambda*dt;M=len(weights)-1;assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(ze) and row['poisson']['M']==M
            tail=ze**(M+1)/math.factorial(M+1)/(1-ze/(M+2));assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            ratio=ze/(M+1);assert ratio>=1 or ze**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            degree_max=int(np.bincount(edges.ravel(),minlength=N*N).max())
            machine=float(64*np.finfo(float).eps*(degree_max+1)*(M+1)*(1+float(ze))*max(1.,math.sqrt(float(mass.sum())))*max(1.,float(np.max(abs(c0)))))
            cum_machine+=machine;cum_tail+=float(tail)*math.sqrt(float(mass.sum()))*float(np.max(abs(c0)))
            assert machine==row['machine_indicator_absolute_this_interval'] and cum_machine==row['cumulative_machine_indicator_absolute'] and cum_tail==row['cumulative_tail_indicator_absolute']
            v=last.copy();w=math.exp(-float(ze));following=w*v;assert weights[0]==w
            for j in range(1,M+1):v=P@v;w*=float(ze)/j;assert weights[j]==w;following+=w*v
            np.testing.assert_array_equal(following,path['independent_states'][step+1]);last=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            norm=lambda v:float(np.sqrt(mass@(v*v)))
            delta=norm(path['states'][step+1]-following);assert delta==row['primary_control_mass_norm_difference'] and delta<=cum_machine+cum_tail;max_control=max(max_control,delta/sigma)
            u=path['states'][step];i,j=edges.T;net=np.zeros(N*N);flux=gf*(u[j]-u[i]);np.add.at(net,i,flux);np.add.at(net,j,-flux);root=np.sqrt(mass)
            diff=norm(A@(root*u)/root-net/mass);assert diff<1e-10*max(1.,norm(net/mass));max_action=max(max_action,diff)
            for u,chi in zip((path['states'][step],path['exponential_half_states'][step],path['states'][step+1]),row['chi_start_half_end']):
                assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
                assert abs(-float((root*u)@(A@(root*u)))-chi)<1e-10*max(1.,chi)
            u=path['states'][step+1];drift=float(mass@(u-c0));assert drift==row['raw_mass_change_from_c0']
            assert abs(drift)/float(mass@c0)<1e-10 and u.min()>=c0.min()-1e-10 and u.max()<=c0.max()+1e-10;max_raw=max(max_raw,abs(drift))
            variance=float(mass@((u-float(mass@u/mass.sum()))**2));assert variance<=previous_variance+1e-10*sigma*sigma;previous_variance=variance
            modes=arrays(out/'analytic-references.npz')['modes'];s=F(frame['shear_exact']);p=F(float(.02))
            u0=math.sin(math.pi/N)**2;v0=math.sin(2*math.pi/N)**2;U=float(p*(2-2*s+s*s));V=float(p*s)
            lambdas=(4*N*N*u0*U,4*N*N*(u0*U+v0*V))
            eigen=[norm(A@(root*phi)/root+lam*phi) for phi,lam in zip(modes,lambdas)];constant=norm(.5*(A@root)/root)
            limits=[1e-10*max(1.,lam*norm(phi)) for phi,lam in zip(modes,lambdas)]
            assert all(x<=y for x,y in zip(eigen,limits)) and constant<=1e-10
            assert rr[step]=={'step':step,'declared_frozen_eigenvalues':list(lambdas),'mode_residues_absolute':eigen,
              'constant_residue_absolute':constant,'original_mode_algebra_limits':limits,'all_original_residue_guards_passed':True}
            total_residue=sum(eigen)+constant;residues[K].append(total_residue);max_mode=max(max_mode,total_residue)
            save(folder/f'action-audit-step{step:02d}.json',{'cache_sha256':item['sha256'],
              'exact_rates_whole_control_sequence_and_minimal_raw_tail_verified':True,'symmetric_vs_shared_action_mass_norm_difference':diff,
              'raw_mass_change':drift,'mode_plus_constant_residue_absolute':total_residue})
            print(f'independent affine-shear action:k{K} step{step+1}/{K}',flush=True);del A,P,adj,g,degree,act;gc.collect()
    return {'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,
      'maximum_primary_control_relative_difference':max_control,'maximum_native_raw_mass_drift':max_raw,
      'maximum_mode_plus_constant_residue_absolute':max_mode}


def verify_pure(folder,N,pure):
    world=read(folder/'pure-inputs.json');labels=pure[f'n{N}_positions'];p=F(float(.02))
    np.testing.assert_array_equal(world['ids'],pure[f'n{N}_ids'])
    for K in COUNTS:
        case=world['cases'][str(K)];assert case['K']==K and len(case['frames'])==2*K+1
        for j,frame in enumerate(case['frames']):
            t=F(j,40*K);s=5*t
            assert frame['time_exact']==str(t) and frame['shear_exact']==str(s) and frame['rho']==[1.]*(N*N) and F(frame['D_float'])==p
            assert frame['boundary']=={'origin_exact':['0','0'],'shear_exact':str(s),'basis_columns_exact':[['1','0'],[str(s),'1']]}
            for actual,(x,y) in zip(frame['world_positions_exact'],labels):assert tuple(map(F,actual))==(F(float(x))+s*F(float(y)),F(float(y)))
    return world


def direct_material_audit(out,data,paths,bundles):
    """Flat ID groups and direct component integrals; no production observation function."""
    base=16;mp,cp=data[base]['mass'],data[base]['c0'];saved=arrays(out/'common-observations.npz')
    checks=probes_checked=0;maxdiff=0.;probes=[0,8,15,128,255]
    for N in LEVELS:
        m=data[N]['mass'];r=N//base;groups=[[] for _ in mp]
        for i in range(N*N):groups[(i//N//r)*base+(i%N//r)].append(i)
        capacities=np.array([math.fsum(float(m[i]) for i in ids) for ids in groups]);np.testing.assert_array_equal(capacities,mp)
        sources={f'n{N}_c0':data[N]['c0'][None,:],f'n{N}_reference':bundles[N]['reference'],
          f'n{N}_continuous_grid_symbol':bundles[N]['continuous_grid_symbol'],f'n{N}_frozen16':bundles[N]['frozen_symbols'][16]}
        for K in COUNTS:
            for field in ('states','independent_states'):sources[f'n{N}_k{K}_{field}']=paths[N,K][field]
        for key,values in sources.items():
            recorded=np.atleast_2d(saved[key])
            for row,(u,v) in enumerate(zip(values,recorded)):
                expected=np.array([math.fsum(float(m[i])*float(u[i]) for i in ids)/float(mp[j]) for j,ids in enumerate(groups)])
                diff=float(np.max(abs(expected-v)));maxdiff=max(maxdiff,diff);assert diff<=1e-10*max(1.,float(np.max(abs(u))))
                checks+=1
                for j in probes:
                    exact=sum((F(float(m[i]))*F(float(u[i])) for i in groups[j]),F(0))/F(float(mp[j]))
                    assert expected[j]==float(exact);probes_checked+=1
    return {'independent_parent_ID_groups_and_all_component_integrals_verified':True,'production_observe_path_or_group_sum_called_by_this_audit':False,
      'original_capacities_unchanged':True,'all_saved_common_snapshots_checked':checks,'fixed_parent_exact_integral_probes_checked':probes_checked,
      'maximum_original_vs_direct_fsum_component_mean_difference':maxdiff,'audit_scale_original_algebra_tolerance':1e-10}


def main(directory):
    start=time.monotonic();out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');config=read(out/'config.json')
    assert ex['main_first_attempt_passed'] and not ex['tests_first_attempt_passed'] and ex['tests_second_attempt_passed'] and ex['resource_caps_passed']
    assert ex['elapsed_main_seconds']<1500 and ex['peak_RSS_MiB']<1024 and config['parent_manifest_entries_verified']==5830
    assert read(out/'tests.json')=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert read(out/'ltn-affine-shear-spatial-tests-attempt1.json')=={'successful':False,'checks_run':3,'failures':1,'errors':0,'skipped':0}
    assert not read(out/'test-correction.json')['scientific_criteria_or_thresholds_changed']
    original_test=(out/'ltn-test-affine-shear-spatial-attempt1.py').read_text()
    corrected_test=original_test.replace("self.assertFalse(r['fine_grid_direction_response_resolved']);self.assertFalse(r['registered_three_space_level_affine_shear_control_passed'])",
      "self.assertTrue(r['fine_grid_direction_response_resolved']);self.assertTrue(r['registered_three_space_level_affine_shear_control_passed'])")
    assert corrected_test==(ROOT/'research/reconnect/test_affine_shear_spatial.py').read_text()
    assert status['recorded_checks_total']==490 and not status['fluid_orc_enabled'] and not status['original_backends_changed'] and not status['general_T3_admitted']
    lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/affine_shear_spatial_protocol.json')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==digest(ROOT/item['source'])==item['sha256'],item['copy']
    index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json');geo=read(out/'geometry-audit-execution.json')
    assert len(index)==seal['graphs']==56 and seal['all56_new_world_euclidean_graphs_sealed_before_any_material_reference_or_reused_state']
    assert not seal['builder_received_mass_c0_reference_true_internal_faces_F_C_or_precomputed_g'] and digest(out/'graph-index.json')==seal['graph_index_sha256']
    assert geo['graphs_verified']==56 and geo['all56_independent_sliding_world_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_oldstates_loaded'] and not geo['original_or_sliding_geometry_builder_called_by_auditor']
    pure=arrays(out/'original-pure-inputs.npz');per={}
    for N in LEVELS:
        np.testing.assert_array_equal(pure[f'n{N}_ids'],np.arange(N*N))
        for i,point in enumerate(pure[f'n{N}_positions']):assert tuple(F(float(x)) for x in point)==(F(2*(i%N)+1,2*N),F(2*(i//N)+1,2*N))
    for N in NEW:
        folder=out/f'n{N}';world=verify_pure(folder,N,pure);d=arrays(folder/'material-inputs.npz')
        with np.load(folder/'original-material-reference.npz',allow_pickle=False) as z:
            for name in ('mass','c0'):assert bitwise_same(z[name],d[name])
        np.testing.assert_array_equal(d['mass'],np.full(N*N,1/(N*N)));np.testing.assert_array_equal(d['ids'],world['ids'])
        independent_references(folder,d,pure[f'n{N}_positions'],world,N)
        localindex=read(folder/'graph-index.json');assert len(localindex)==28
        for key,item in localindex.items():assert index[f'n{N}-'+key]=={**item,'file':f'n{N}/'+item['file'],'world_file':f'n{N}/'+item['world_file']}
        per[str(N)]=verify_paths(folder,N,d,world,localindex);del world;gc.collect()
    args=load_all(out);computed,common,bundles=compare(*args);assert plain_json(computed)==read(out/'results.json')
    assert plain_json(computed['per_grid_controls']['32'])==read(out/'n32/old-case-results.json')==read(SOURCE/'results-completed.json')
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():assert bitwise_same(value,saved[key]),key
    for N in LEVELS:check_references(out/f'n{N}',bundles[N])
    material=direct_material_audit(out,args[0],args[4],bundles);save(out/'common-material-audit.json',material)
    sigma=computed['common_initial_fluctuation_norm']
    for row in computed['adjacent_spatial_comparisons']:
        for name,field,den in [('native','native_indicator_components_absolute',sigma),('common','common_indicator_components_absolute',sigma),('native_H','native_H_indicator_components',1.),('common_H','common_H_indicator_components',1.)]:
            budget=sum(row[field].values())/den;assert budget==row['fixed_indicators'][name] and min(row[field].values())>=0
            gain=row['gains'][name];decision='fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
            assert decision==row['decisions'][name]
        assert not row['signed_cancellation_or_extrapolated_state_used']
    preserved=preservation();known={item['manifest'] for item in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==98;save(out/'preservation.json',preserved)
    assert digest(ROOT/'exports/ltn-affine-shear.zip')=='470fd77d903b240f0752bc43238df967ccdb809b052487b4550de8c676f47c0e'
    summary={'all56_independent_world_geometries_exact_rates_and_full_raw_sequences_verified':True,'all56_all_cell_world_halfplane_searches_independently_verified':True,
      'new_N16_N64_independent80_100_digit_references_passed':True,'three_N32_saved_paths_reused_not_rerun':True,'old_N32_decision_byte_for_value_preserved':True,
      'all_common_material_integrals_and_positive_budgets_verified':True,'all_three_source_label_orders_exact_nested_material_centers_verified':True,
      'per_new_grid_controls':per,'common_material_audit':material,'new_software_checks':3,'recorded_checks_total':490,
      'tests_first_attempt_passed':False,'tests_second_attempt_passed':True,'test_expectation_correction_preserved':True,
      'comparison_decision':computed['comparison_decision'],'registered_three_space_level_affine_shear_control_passed':computed['registered_three_space_level_affine_shear_control_passed'],
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':98,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'general_T3_admitted':False,'general_nonorthogonal_domains_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    save(out/'numerical-audit.json',summary)
    save(out/'audit-execution.json',{'audit_exit_code':0,'audit_first_attempt_passed':True,'elapsed_audit_seconds':time.monotonic()-start,
      'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'audit_source_sha256':digest(Path(__file__))})
    print(json.dumps(plain_json(summary),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)


