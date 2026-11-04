"""Independent affine laws, high-precision references and complete raw path audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,hashlib,json,math
import mpmath as mp
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save as new_save,digest,preservation
from t3_affine_study import SOURCE,inputs
from t3_affine_core import evaluate
from t2_d0_core import bitwise_same
from shear_union64_study import arrays

MODELS=('constant_D','constant_Gamma');COUNTS=(4,8,16);N=32


def save(path,obj):
    if Path(path).exists():assert read(path)==obj,str(path)
    else:new_save(path,obj)


def independent_references(out,labels,m,c0,world):
    """No production clock, reference evaluator or graph coefficients are called."""
    saved=arrays(out/'analytic-references.npz');parameters=read(out/'analytic-parameters.json')
    p=F(float(.02));epsilon=np.finfo(float).eps
    R=64*epsilon*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))
    assert R==parameters['reference_evaluation_indicator_absolute']
    mass=tuple(F(float(x)) for x in m);scalar=tuple(F(float(x)) for x in c0)
    mu=sum((a*b for a,b in zip(mass,scalar)),F(0))/sum(mass,F(0))
    variance=sum((a*(b-mu)**2 for a,b in zip(mass,scalar)),F(0))
    upper=parameters['initial_fluctuation_upper'];assert F(upper)**2>=variance
    assert F(math.nextafter(upper,-math.inf))**2<variance
    clocks={};checks=[]
    for model in MODELS:
        for K in COUNTS:
            ww=[];nn=[];rr=[];case=world['models'][model][str(K)]
            for stop in (0,K//2,K):
                wire=nominal=bound=F(0)
                for step in range(stop):
                    frame=case['frames'][2*step+1];a=F(frame['a_exact']);rho=F(float(1/a))
                    D=p if model=='constant_D' else p*a
                    gamma=rho*F(float(D));ideal=p/a if model=='constant_D' else p
                    wire+=F(1,20*K)*gamma*(1/a+a)
                    nominal+=F(1,20*K)*ideal*(1/a+a)
                    bound+=F(1,20*K)*4*N*N*abs(gamma-ideal)*(1/a+a)
                ww.append(wire);nn.append(nominal);rr.append(bound)
            meta=parameters['meta'][model]['cases'][str(K)]
            assert list(map(str,ww))==meta['wire_midpoint_clocks_exact']
            assert list(map(str,nn))==meta['nominal_midpoint_clocks_exact']
            assert list(map(str,rr))==meta['input_rounding_integrated_generator_bound_exact']
            for q,v in zip(rr,meta['input_rounding_absolute_upper']):
                target=q*F(upper);assert F(v)>=target
                assert v==0 or F(math.nextafter(v,-math.inf))<target
            clocks[model,K]=ww,nn
    rounded={}
    for precision in (80,100):
        with mp.workdps(precision):
            exact=lambda q:mp.mpf(q.numerator)/q.denominator
            phase=[mp.sin(2*mp.pi*exact(F(float(x))))*mp.cos(2*mp.pi*exact(F(float(y)))) for x,y in labels]
            amp=mp.mpf(1)/5*(mp.sin(mp.pi/N)/(mp.pi/N))**2
            base=[amp*v for v in phase];P=exact(p);qh=4*N*N*mp.sin(mp.pi/N)**2;qc=4*mp.pi**2
            values={'initial':np.array([float(mp.mpf('.5')+v) for v in base])}
            for model in MODELS:
                clock=[]
                for tf in (F(0),F(1,40),F(1,20)):
                    t=exact(tf);a=1-5*t
                    clock.append(P*(t/a+t) if model=='constant_D' else P*(mp.log(1/a)/5+t-mp.mpf(5)/2*t*t))
                for suffix,q in (('reference',qc),('continuous_grid_symbol',qh)):
                    values[model+'_'+suffix]=np.array([[float(mp.mpf('.5')+v*mp.exp(-q*z)) for v in base] for z in clock])
                for K in COUNTS:
                    for suffix,cc in zip(('frozen_wire_symbol','frozen_nominal_symbol'),clocks[model,K]):
                        values[f'{model}_k{K}_{suffix}']=np.array([[float(mp.mpf('.5')+v*mp.exp(-qh*exact(z))) for v in base] for z in cc])
            rounded[precision]=values
    norm=lambda v:float(np.sqrt(m@(v*v)))
    for key,value in rounded[80].items():
        assert bitwise_same(value,rounded[100][key]),key
        original=c0 if key=='initial' else saved[key]
        errors=[norm(a-b) for a,b in zip(np.atleast_2d(value),np.atleast_2d(original))]
        assert max(errors)<=R,(key,errors,R)
        checks.append({'array':key,'80_100_digit_roundings_bitwise_equal':True,'mass_norm_differences':errors,'fixed_evaluator_allowance_absolute':R})
    target=out/'independent-analytic-80digit.npz'
    if target.exists():
        old=arrays(target)
        for key,value in rounded[80].items():assert bitwise_same(old[key],value),key
    else:np.savez_compressed(target,**rounded[80])
    save(out/'analytic-reference-audit.json',{'independent_initial_cell_mean_and_both_constitutive_clocks':True,
      'production_clock_or_reference_evaluator_called':False,'precisions_decimal_digits':[80,100],
      'all_80_100_digit_roundings_bitwise_equal':True,'all_saved_evaluations_within_fixed_allowance':True,
      'exact_initial_variance':str(variance),'upward_initial_fluctuation_bound_verified':True,
      'all_exact_wire_nominal_and_input_Duhamel_integrals_verified':True,'checks':checks})


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');config=read(out/'config.json')
    assert ex['main_first_attempt_passed'] and not ex['tests_first_attempt_passed'] and ex['tests_failed_attempts_preserved']==1
    assert ex['elapsed_main_seconds']<600 and ex['peak_RSS_MiB']<1024 and config['parent_manifest_entries_verified']==4015
    assert read(out/'tests.json')=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert read(out/'tests-first-attempt.json')=={'successful':False,'checks_run':3,'failures':1,'errors':0,'skipped':0}
    assert status['recorded_checks_total']==480 and not status['fluid_orc_enabled'] and not status['original_geometry_resistance_time_backends_changed']
    assert not status['general_T3_admitted'] and not status['general_smooth_flow_admitted']
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    lock=read(out/'implementation-lock.json');assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/t3_affine_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    gate=read(SOURCE/'next-gate-spec.json');assert read(out/'source-gate-spec.json')==gate
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']==digest(ROOT/item['source'])
    seal=read(out/'coefficient-seal.json');index=read(out/'graph-index.json');world=read(out/'pure-inputs.json')
    assert seal['graphs']==len(index)==56 and seal['all56_model_coefficients_and_world_graphs_sealed_before_material_or_reference']
    assert not seal['builder_received_mass_scalar_reference_internalfaces_or_F_C']
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and seal['pure_inputs_sha256']==digest(out/'pure-inputs.json')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==56 and geo['all56_independent_world_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_oldstates_loaded'] and not geo['originalbuildercalledbyauditor']
    pure=arrays(out/'original-pure-inputs.npz');labels=pure['n32_positions'];assert world['ids']==pure['n32_ids'].tolist()
    assert not world['material_mass_c0_reference_or_internal_faces_supplied'];p=F(float(.02))
    for model in MODELS:
        for K in COUNTS:
            case=world['models'][model][str(K)];assert case['K']==K and len(case['frames'])==2*K+1
            for j,frame in enumerate(case['frames']):
                t=F(j,40*K);a=F(8*K-j,8*K);rho=1/a;D=p if model=='constant_D' else p*a
                gamma=F(float(rho))*F(float(D));ideal=p/a if model=='constant_D' else p
                assert frame['time_exact']==str(t) and frame['a_exact']==str(a) and frame['model']==model
                assert frame['rho_exact']==str(rho) and frame['rho_float']==float(rho) and frame['rho']==[float(rho)]*N*N
                assert frame['D_nominal_exact']==str(D) and frame['D_float']==float(D)
                assert frame['Gamma_actual_exact']==str(gamma) and frame['Gamma_nominal_exact']==str(ideal) and frame['Gamma_difference_exact']==str(gamma-ideal)
                assert frame['boundary']=={'origin_exact':['0','0'],'periods_exact':[str(a),'1'],
                  'orientation_rows_exact':[['1','0'],['0','1']],'basis_columns_exact':[[str(a),'0'],['0','1']]}
                for point,(x,y) in zip(frame['world_positions_exact'],labels):assert tuple(map(F,point))==(a*F(float(x)),F(float(y)))
    d=inputs(out);mass,c0=d['mass'],d['c0'];sigma=math.sqrt(float(mass@((c0-float(mass@c0/mass.sum()))**2)))
    np.testing.assert_array_equal(mass,np.full(N*N,1/(N*N)));np.testing.assert_array_equal(d['ids'],world['ids'])
    independent_references(out,labels,mass,c0,world)
    runs={};paths={};residues={};max_action=max_control=max_raw=max_mode=0.
    for model in MODELS:
        for K in COUNTS:
            folder=out/model/f'k{K}';path=arrays(folder/'trajectory-states.npz');run=read(folder/'native-results.json')
            paths[model,K]=path;runs[model,K]=run;rr=read(folder/'modal-residues.json');residues[model,K]=[]
            mm=tuple(F(float(v)) for v in mass);dt=F(1,20*K)
            np.testing.assert_array_equal(path['states'][0],c0);np.testing.assert_array_equal(path['independent_states'][0],c0)
            last=c0.copy();cum_machine=cum_tail=0.;previous_variance=sigma*sigma
            for step in range(K):
                item=index[f'{model}-k{K}-step{step:02d}'];row=run['steps'][step];frame=world['models'][model][str(K)]['frames'][2*step+1]
                assert (item['N'],item['model'],item['K'],item['step'],item['frame'])==(N,model,K,step,2*step+1)
                assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
                assert item['midpoint_time_saved']==float(F(frame['time_exact']))==row['midpoint_time_saved']
                receipt=read(folder/f'world-geometry-audit-step{step:02d}.json')
                assert receipt['world_cache_sha256']==item['world_sha256']==digest(out/item['world_file'])
                assert receipt['pure_inputs_sha256']==seal['pure_inputs_sha256']
                assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_oriented_world_geometry.py')
                assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_t3_affine_geometry.py')
                assert not receipt['mass_c0_reference_or_oldstates_loaded'] and not receipt['originalbuildercalledbyauditor']
                assert receipt['certificate']['all_world_vertices_rebuilt_directly_from_world_bisectors']
                packet=read(out/item['file']);wp=read(out/item['world_file'])
                for field in ('ids','edges','image_shifts','cell_vertex_indices','cell_incoming_labels','cell_areas_exact','face_vertex_indices','face_over_distance_exact','conductances_exact','D_exact'):assert packet[field]==wp[field],field
                np.testing.assert_array_equal(packet['ids'],world['ids']);a=F(frame['a_exact']);gamma=F(frame['rho_float'])*F(frame['D_float'])
                edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']));points=[tuple(map(F,q)) for q in frame['world_positions_exact']];counts=[0,0]
                for (i,j),(sx,sy),value in zip(edges,packet['image_shifts'],g):
                    dx=points[j][0]+sx*a-points[i][0];dy=points[j][1]+sy-points[i][1]
                    if dy==0 and abs(dx)==a/N:assert value==gamma/a;counts[0]+=1
                    else:assert dx==0 and abs(dy)==F(1,N) and value==gamma*a;counts[1]+=1
                assert counts==[N*N,N*N] and set(map(F,packet['cell_areas_exact']))=={a/(N*N)}
                assert read(folder/f'direction-step{step:02d}.json')['all_directional_conductances_exact']
                cert=read(folder/f'isometry-step{step:02d}.json')
                assert cert['exact_proper_isometry_verified'] and cert['local_coordinates_float64_lossless'] and cert['period_lengths_float64_lossless']
                assert not cert['points_wrapped_jittered_or_fitted'] and not cert['provided_source_local_labels_or_mass_scalar_reference']
                raw=np.array([[float(v) for v in q] for q in points]);assert cert['local_coordinate_C_order_sha256']==hashlib.sha256(raw.tobytes(order='C')).hexdigest()
                del packet,wp,points;degree=[F(0)]*(N*N)
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
                lam=4*N*N*math.sin(math.pi/N)**2*float(gamma*(a+1/a));phi=c0-.5
                eigen=norm(A@(root*phi)/root+lam*phi);constant=norm(.5*(A@root)/root);limit=1e-10*max(1.,lam*norm(phi))
                assert eigen+constant<=limit and rr[step]=={'step':step,'declared_frozen_eigenvalue':lam,'mode_residue_absolute':eigen,
                  'constant_residue_absolute':constant,'original_algebra_scale':limit,'original_algebra_scale_passed':True}
                residues[model,K].append(eigen+constant);max_mode=max(max_mode,eigen+constant)
                save(folder/f'action-audit-step{step:02d}.json',{'cache_sha256':item['sha256'],
                  'exact_rates_whole_control_sequence_and_minimal_raw_tail_verified':True,'symmetric_vs_shared_action_mass_norm_difference':diff,
                  'raw_mass_change':drift,'mode_plus_constant_residue_absolute':eigen+constant})
                print(f'independent affine action:{model} k{K} step{step+1}/{K}',flush=True);del A,P,adj,g,degree,act;gc.collect()
    computed,bundle=evaluate(d,world,runs,paths,residues);assert computed==read(out/'results.json')
    # Recompute each comparison directly from saved states and the independently checked references.
    refs=arrays(out/'analytic-references.npz');norm=lambda v:float(np.sqrt(mass@(v*v)))
    E0=float(mass@((c0-float(mass@c0/mass.sum()))**2))
    H=lambda u:1-float(mass@((u-float(mass@u/mass.sum()))**2))/E0
    HB=lambda z:(2*math.sqrt(float(mass.sum()))*max(1.,float(np.max(abs(c0))))*z+z*z)/E0
    for k,row in enumerate(computed['model_comparisons']):
        c=paths['constant_D',16]['states'][8*k];g=paths['constant_Gamma',16]['states'][8*k]
        # Production deliberately adds model totals/observed differences sequentially.
        # Python3.12 sum(float) uses a different, compensated accumulation order.
        total=0.
        for value in row['indicator_components_absolute'].values():total+=value
        assert row['fixed_state_indicator_relative']==total/sigma and row['fixed_H_indicator']==HB(total)
        assert row['model_state_difference_relative']==norm(c-g)/sigma and row['model_H_difference']==H(c)-H(g)
        assert row['model_state_response_resolved']==(norm(c-g)>total) and row['model_H_response_resolved']==(H(c)-H(g)>HB(total))
        assert row['analytic_model_difference_consistent']==(norm((c-g)-(refs['constant_D_reference'][k]-refs['constant_Gamma_reference'][k]))<=total)
    for model,rows in computed['time_controls'].items():
        for k,row in enumerate(rows):
            a,b,c=(paths[model,K]['states'][k*K//2] for K in COUNTS)
            assert row['state4_8_relative']==norm(a-b)/sigma and row['state8_16_relative']==norm(b-c)/sigma
            assert row['H4_8']==abs(H(a)-H(b)) and row['H8_16']==abs(H(b)-H(c))
            assert row['state_time_decrease_resolved']==(norm(a-b)-norm(b-c)>row['fixed_state_decrease_indicator_relative']*sigma)
            assert row['H_time_decrease_resolved']==(abs(H(a)-H(b))-abs(H(b)-H(c))>row['fixed_H_decrease_indicator'])
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        pth=ROOT/item['manifest'];assert digest(pth)==item['manifest_sha256'];entries=read(pth);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==93;save(out/'preservation.json',preserved)
    summary={'all56_independent_world_geometries_exact_rates_and_full_raw_sequences_verified':True,
      'declared_affine_motion_density_and_both_constitutive_laws_verified_exactly':True,
      'all_directional_gx_gy_and_original_fixed_masses_verified':True,'independent_80_100_digit_references_passed':True,
      'all_input_Duhamel_and_original_modal_residue_guards_verified':True,'all_saved_comparison_fields_recomputed':True,
      'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,'maximum_primary_control_relative_difference':max_control,
      'maximum_native_raw_mass_drift':max_raw,'maximum_mode_plus_constant_residue_absolute':max_mode,
      'new_software_checks':3,'recorded_checks_total':480,'tests_first_attempt_passed':False,'failed_test_attempts_preserved':1,
      'comparison_decision':computed['comparison_decision'],'T3_registered_affine_case_passed':computed['T3_registered_affine_case_passed'],
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':93,
      'new_reference_PDE_solves':0,'old_candidate_reruns':0,'general_T3_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
