"""Independent affine-shear references, complete positive paths and world constraints."""
from pathlib import Path
from fractions import Fraction as F
import argparse,gc,json,math,resource,time
import mpmath as mp
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from affine_shear_study import SOURCE,load_all
from affine_shear_core import evaluate
from affine_shear_completion import plain_json
from shear_union64_study import arrays
from t2_d0_core import bitwise_same
from voronoi_time_action import metrics
COUNTS=(4,8,16)


def all_cell_world_halfplanes(packet,frame):
    """All cells, independent finer fractional bins, no clipping or original builder."""
    s=F(frame['shear_exact']);points=[tuple(map(F,p)) for p in frame['world_positions_exact']];n=len(points)
    fractions=[(x-s*y,y) for x,y in points];side=math.isqrt(n)
    if side*side<n:side+=1
    bins={}
    for j,(a,b) in enumerate(fractions):bins.setdefault((int(a*side),int(b*side)),[]).append(j)
    def vector(i,label):
        j,sx,sy=label;return points[j][0]+sx+s*sy-points[i][0],points[j][1]+sy-points[i][1]
    checked=0;worst=F(0)
    for i,labels in enumerate(packet['cell_incoming_labels']):
        poly=[]
        for k,label in enumerate(labels):
            a,b=vector(i,label),vector(i,labels[(k+1)%len(labels)])
            det=a[0]*b[1]-a[1]*b[0];assert det
            ca=(a[0]*a[0]+a[1]*a[1])/2;cb=(b[0]*b[0]+b[1]*b[1])/2
            poly.append(((ca*b[1]-a[1]*cb)/det,(a[0]*cb-ca*b[0])/det))
        r2=max(x*x+y*y for x,y in poly);r=max(abs(x)+abs(y) for x,y in poly)
        assert 4*r2*(1+s)**2<1 and F(packet['initial_search_radius_L1_exact'][i])>=r
        bound=2*r*(1+s);a0,b0=fractions[i]
        lower=(int((a0-bound)*side//1),int((b0-bound)*side//1));upper=(int((a0+bound)*side//1),int((b0+bound)*side//1))
        for a in range(lower[0],upper[0]+1):
            for b in range(lower[1],upper[1]+1):
                sx,sy=a//side,b//side
                for j in bins.get((a%side,b%side),()):
                    if j==i and sx==sy==0:continue
                    vx,vy=vector(i,(j,sx,sy));d2=vx*vx+vy*vy
                    if d2>4*r2:continue
                    residual=max(2*(vx*x+vy*y)-d2 for x,y in poly);assert residual<=0
                    worst=max(worst,residual);checked+=1
    return {'all_cells_world_halfplanes_verified':True,'independent_fractional_bin_side':side,
      'uses_final_rebuilt_world_radius_not_production_search_radius':True,'candidate_halfplanes_checked':checked,
      'maximum_signed_halfplane_residual_exact':str(worst),'original_or_sliding_builder_called':False}


def independent_references(out,data,labels,world):
    N=32;m,c0=data['mass'],data['c0'];p=F(float(.02));saved=arrays(out/'analytic-references.npz');meta=read(out/'analytic-parameters.json')
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
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_affine_shear_geometry.py')
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


def main(directory):
    start=time.monotonic();out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');config=read(out/'config.json')
    assert ex['original_study_exit_code']==1 and ex['completion_exit_code']==0 and not ex['main_first_attempt_passed']
    assert ex['scientific_caches_complete_before_reporting_failure'] and ex['all_pre_completion_files_unchanged']
    assert ex['no_scientific_path_rerun_during_completion'] and ex['criteria_and_thresholds_unchanged']
    assert ex['elapsed_main_to_reporting_failure_seconds']<900 and ex['peak_main_RSS_MiB']<1024
    assert config['parent_manifest_entries_verified']==5540 and status['recorded_checks_total']==487
    assert not status['fluid_orc_enabled'] and not status['general_T3_admitted'] and not status['original_backends_changed']
    assert read(out/'tests.json')=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert read(out/'implementation-lock.json')['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/affine_shear_protocol.json')
    completion=read(out/'completion-lock.json');assert completion['completion_code_sha256']==digest(ROOT/'research/reconnect/affine_shear_completion.py')
    assert completion['frozen_core_sha256']==digest(ROOT/'research/reconnect/affine_shear_core.py')
    for name,h in read(out/'pre-completion-files-sha256.json').items():assert digest(out/name)==h,name
    assert read(out/'failed-execution.json')['type']=='TypeError' and 'JSON serializable' in read(out/'failed-execution.json')['error']
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==digest(ROOT/item['source'])==item['sha256']
    index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json');geo=read(out/'geometry-audit-execution.json')
    assert len(index)==seal['graphs']==28 and seal['all28_world_euclidean_graphs_sealed_before_material_scalar_reference']
    assert not seal['builder_received_mass_c0_reference_true_internal_faces_F_C_or_precomputed_g']
    assert digest(out/'graph-index.json')==seal['graph_index_sha256'] and digest(out/'pure-inputs.json')==seal['pure_inputs_sha256']
    assert geo['graphs_verified']==28 and geo['all28_independent_sliding_world_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_oldstates_loaded'] and not geo['original_or_sliding_geometry_builder_called_by_auditor']
    args=load_all(out);data,labels,world,runs,paths,res=args;N=32;p=F(float(.02))
    np.testing.assert_array_equal(data['mass'],np.full(1024,1/1024));np.testing.assert_array_equal(data['ids'],world['ids'])
    for i,(x,y) in enumerate(labels):assert (F(float(x)),F(float(y)))==(F(2*(i%32)+1,64),F(2*(i//32)+1,64))
    for K in COUNTS:
        case=world['cases'][str(K)];assert case['K']==K and len(case['frames'])==2*K+1
        for j,frame in enumerate(case['frames']):
            t=F(j,40*K);s=5*t
            assert frame['time_exact']==str(t) and frame['shear_exact']==str(s) and frame['rho']==[1.]*1024 and F(frame['D_float'])==p
            assert frame['boundary']=={'origin_exact':['0','0'],'shear_exact':str(s),'basis_columns_exact':[['1','0'],[str(s),'1']]}
            for actual,(x,y) in zip(frame['world_positions_exact'],labels):assert tuple(map(F,actual))==(F(float(x))+s*F(float(y)),F(float(y)))
    independent_references(out,data,labels,world)
    summary=verify_paths(out,N,data,world,index)
    computed,b=evaluate(*args);assert plain_json(computed)==read(out/'results-completed.json')
    m,c0=data['mass'],data['c0'];mu0=float(m@c0/m.sum());w=arrays(out/'analytic-references.npz')['weights']
    I=lambda c:np.array([float(m@((c-mu0)*v)) for v in w]);initial=I(c0);ri=I(b['reference'][0]);norm=lambda v:float(np.sqrt(m@(v*v)))
    for k,row in enumerate(computed['direction_observables']):
        actual=I(paths[16]['states'][8*k]);ref=I(b['reference'][k]);a8=I(paths[8]['states'][4*k])
        assert row['raw_actual_observables']==actual.tolist() and row['raw_initial_observables']==initial.tolist()
        assert row['raw_reference_observables']==ref.tolist() and row['raw_reference_initial_observables']==ri.tolist()
        parts=computed['reference_error_components']['16'];B=parts[k]['total_absolute']+parts[0]['total_absolute']+2*b['reference_evaluation_indicator_absolute']
        indicator=[norm(v)*B+abs(x-y) for v,x,y in zip(w,actual,a8)]
        assert row['positive_observable_indicators']==indicator and row['fixed_positive_contrast_indicator']==sum(indicator)
        contrast=float((actual-initial)[0]-(actual-initial)[1]);reference=float((ref-ri)[0]-(ref-ri)[1])
        assert row['actual_direction_contrast']==contrast and row['reference_direction_contrast']==reference
        assert row['direction_contrast_resolved']==(contrast>sum(indicator))
        assert not row['state_or_observables_rescaled_or_projected'] and not row['signed_error_cancellation_used_for_acceptance']
    preserved=preservation();known={item['manifest'] for item in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
      'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256'];entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==97;save(out/'preservation.json',preserved)
    assert digest(ROOT/'exports/ltn-t3-coverage.zip')=='dac706856f97f44be0cb037b85b5476908baca6bb0fc473d9e64032a0588ace0'
    summary.update({'all28_independent_world_geometries_exact_rates_and_full_raw_sequences_verified':True,
      'all28_all_cell_world_halfplane_searches_independently_verified':True,'independent80_100_digit_two_mode_references_passed':True,
      'all_pre_reporting_failure_caches_and_frozen_sources_unchanged':True,'reporting_failure_and_incomplete_JSON_preserved':True,
      'completion_performed_scalar_serialization_only_no_scientific_rerun':True,'raw_observables_and_positive_indicators_verified':True,
      'registered_affine_shear_control_passed':bool(computed['registered_affine_shear_numerical_control_passed']),
      'comparison_decision':computed['comparison_decision'],'new_software_checks':4,'recorded_checks_total':487,
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':97,
      'new_reference_PDE_solves':0,'old_candidate_reruns':0,'general_T3_admitted':False,'general_nonorthogonal_domains_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False})
    save(out/'numerical-audit.json',summary)
    save(out/'audit-execution.json',{'audit_exit_code':0,'audit_first_attempt_passed':True,'elapsed_audit_seconds':time.monotonic()-start,
      'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'audit_source_sha256':digest(Path(__file__))})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
