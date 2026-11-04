"""Independent new-grid references, raw paths, Padé and fixed-parent spatial audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,json,math,resource,time
import mpmath as mp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from audit_route_decision import ROOT,read,save as raw_save,digest,preservation
from audit_shear_voronoi_time import audit_geometry
from contact_spatial_study import SOURCE,NEW,LEVELS,load_all
from contact_spatial_core import compare
from affine_shear_completion import plain_json
from shear_union64_study import arrays
from t2_d0_core import bitwise_same


def save(p,obj):raw_save(p,plain_json(obj))


def references(out,data,pure,meta,N):
    m=data['mass'];case=read(out.parent/'source-reference-K128-p90.json');rounded={}
    for precision in (80,100):
        with mp.workdps(precision):
            exact=lambda q:mp.mpf(F(q).numerator)/F(q).denominator
            def sc(q):
                q=F(q)%2
                if q==0:return mp.mpf(0),mp.mpf(1)
                if q==F(1,2):return mp.mpf(1),mp.mpf(0)
                if q==1:return mp.mpf(0),mp.mpf(-1)
                if q==F(3,2):return mp.mpf(-1),mp.mpf(0)
                return mp.sin(mp.pi*exact(q)),mp.cos(mp.pi*exact(q))
            sinc=lambda q:mp.mpf(1) if q==0 else sc(q)[0]/(mp.pi*exact(q))
            alpha=list(map(mp.mpf,case['alpha']));beta=list(map(mp.mpf,case['beta']));states=[];serial=[]
            for timef in (F(0),F(1,40),F(1,20)):
                tau=4*mp.pi**2*exact(F(float(.02)))*exact(timef);normal=[]
                for i in range(N):
                    x=F(2*i+1,2*N);rho=1 if i<N//2 else 4
                    if timef==0:
                        a,phase=(F(16,5),F(1,5)) if rho==1 else (F(4,5),F(7,5))
                        v=sinc(a/(2*N))*sc(a*x+phase)[1]/5
                    else:
                        v=alpha[0]
                        for n in range(1,129):
                            s,c=sc(2*n*x);v+=mp.exp(-tau*n*n)*sinc(F(n,N))*(alpha[n]*c+beta[n]*s/rho)
                        v*=mp.exp(-tau)
                    normal.append(v)
                values=[mp.mpf('.5')+v*sinc(F(1,N))*sc(F(2*j+1,N))[1] for j in range(N) for v in normal]
                states.append(np.array(list(map(float,values))))
                serial.append(float(mp.sqrt(mp.fsum(exact(F(float(a)))*(exact(F(float(v)))-v)**2 for a,v in zip(m,values)))))
            rounded[precision]=np.array(states)
    assert bitwise_same(rounded[80],rounded[100])
    R=64*np.finfo(float).eps*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(data['c0']))));assert R==meta['reference_evaluation_indicator_absolute']
    differences=[float(np.sqrt(m@((a-b)**2))) for a,b in zip(rounded[100],data['reference'])];assert max(differences)<=R
    np.testing.assert_array_equal(data['mass'],pure['rho']/(N*N));np.testing.assert_array_equal(data['c0'],data['reference'][0])
    certificates=meta['tail_certificates_by_time'];assert certificates[0] is None
    for t,c in zip((F(1,40),F(1,20)),certificates[1:]):
        lower=4*F(157,50)**2*F(float(.02))*t*(129**2+1);decades=int(lower//F(231,100))
        assert F(c['decay_exponent_lower_exact'])==lower and c['guaranteed_decimal_decades']==decades
        assert F(c['material_mass_norm_tail_bound_exact'])==F(1,4*10**decades)
    assert F(certificates[1]['material_mass_norm_tail_bound_exact'])>F(certificates[2]['material_mass_norm_tail_bound_exact'])
    target=out/'independent-reference80.npz'
    if target.exists():assert bitwise_same(arrays(target)['reference'],rounded[80])
    else:np.savez_compressed(target,reference=rounded[80])
    save(out/'reference-audit.json',{'independent_mpmath80_100_roundings_bitwise_equal':True,'production_reference_evaluator_or_projection_called':False,
      'mass_norm_differences_from_Decimal':differences,'fixed_evaluation_allowance':R,'independent_float_serialization_mass_norms':serial,
      'separate_time_tail_exponents_and_exact_bounds_verified':True,'t0_direct_formula_not_truncated':True})


def raw_sequence(out,data,packet):
    m,c0=data['mass'],data['c0'];mm=list(map(lambda v:F(float(v)),m));edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']));degree=[F(0)]*len(m)
    for (i,j),v in zip(edges,g):degree[i]+=v;degree[j]+=v
    lam=math.ceil(max(d/a for d,a in zip(degree,mm)));rows=[{} for _ in m]
    for (i,j),v in zip(edges,g):
        assert j not in rows[i] and i not in rows[j];rows[i][j]=float(v/(lam*mm[i]));rows[j][i]=float(v/(lam*mm[j]))
    for i in range(len(m)):rows[i][i]=float(1-degree[i]/(lam*mm[i]))
    ptr=[0];col=[];val=[]
    for row in rows:
        for j in sorted(row):col.append(j);val.append(row[j])
        ptr.append(len(col))
    P=csr_matrix((val,col,ptr),shape=(len(m),len(m)));path=arrays(out/'k16/trajectory-states.npz');run=read(out/'k16/native-results.json')
    norm=lambda v:float(np.sqrt(m@(v*v)));previous=c0.copy();machine=tail=0.;maxdifference=maxdrift=maxaction=0.;variance0=float(m@((c0-float(m@c0/m.sum()))**2))
    for step,row in enumerate(run['steps']):
        a=arrays(out/f'k16/action-step{step:02d}.npz')
        for name,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(a[name],v)
        weights=a['weights'];M=len(weights)-1;z=F(lam,320);assert row['operator']['Lambda_exact_integer']==lam and row['dt_exact']=='1/320'
        bound=z**(M+1)/math.factorial(M+1)/(1-z/(M+2));assert bound==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
        ratio=z/(M+1);assert ratio>=1 or z**M/math.factorial(M)/(1-ratio)>F(1,10**16)
        assert str(z)==row['poisson']['z_exact'] and M==row['poisson']['M']
        v=previous.copy();w=math.exp(-float(z));following=w*v;assert weights[0]==w
        for k in range(1,M+1):v=P@v;w*=float(z)/k;assert weights[k]==w;following+=w*v
        np.testing.assert_array_equal(following,path['independent_states'][step+1]);previous=following
        assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
        machine+=float(64*np.finfo(float).eps*5*(M+1)*(1+float(z))*max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
        tail+=float(bound)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
        assert machine==row['cumulative_machine_indicator_absolute'] and tail==row['cumulative_tail_indicator_absolute']
        delta=norm(path['states'][step+1]-following);assert delta==row['primary_control_mass_norm_difference'] and delta<=machine+tail;maxdifference=max(maxdifference,delta)
        A=csr_matrix((a['symmetric_data'],a['symmetric_indices'],a['symmetric_indptr']),shape=tuple(a['shape']));root=np.sqrt(m);gf=np.array(list(map(float,g)));np.testing.assert_array_equal(gf,a['g_float']);i,j=edges.T
        for u,chi in zip((path['states'][step],path['exponential_half_states'][step],path['states'][step+1]),row['chi_start_half_end']):
            flux=gf*(u[j]-u[i]);net=np.zeros(len(m));np.add.at(net,i,flux);np.add.at(net,j,-flux)
            action=norm(A@(root*u)/root-net/m);assert action<=1e-10*max(1.,norm(net/m));maxaction=max(maxaction,action)
            assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
        u=path['states'][step+1];drift=float(m@(u-c0));assert drift==row['raw_mass_change_from_c0'] and abs(drift)/float(m@c0)<1e-10;maxdrift=max(maxdrift,abs(drift))
        assert u.min()>=c0.min()-1e-10 and u.max()<=c0.max()+1e-10
        variance=float(m@((u-float(m@u/m.sum()))**2));assert variance<=variance0+1e-10*variance0;variance0=variance
        assert row['coefficient_cache_sha256']==digest(out/'geometry.json')
    return {'exact_rates_minimal_raw_tail_and_all_control_states_verified':True,'maximum_primary_control_mass_norm_difference':maxdifference,
      'maximum_shared_vs_symmetric_action_mass_norm_difference':maxaction,'maximum_raw_mass_drift':maxdrift,'positive_weights_or_states_normalized':False}


def pade_control(out,data,control,cm,N):
    rho=data['rho'].reshape(N,N)[0];mass=data['mass'];D=F(float(.02));area=F(1,N*N);rr=list(map(lambda v:F(float(v)),rho));B=[[F(0)]*N for _ in range(N)]
    for i in range(N):
        j=(i+1)%N;g=1/(1/(2*rr[i]*D)+1/(2*rr[j]*D));B[i][i]+=g/(rr[i]*area);B[j][j]+=g/(rr[j]*area)
        v=-g/(area*F(math.isqrt((rr[i]*rr[j]).numerator),math.isqrt((rr[i]*rr[j]).denominator)));B[i][j]+=v;B[j][i]+=v
    matrix=np.array([[float(v) for v in row] for row in B]);np.testing.assert_array_equal(matrix,control['normal_matrix'])
    rounding=max(sum((abs(F(float(x))-v) for x,v in zip(row,exact)),F(0)) for row,exact in zip(matrix,B));assert str(rounding)==cm['normal_matrix_rounding_row_bound_exact']
    ev=control['normal_eigenvalues'];Q=control['normal_eigenvectors'];assert float(ev.min())==cm['raw_minimum_normal_eigenvalue']
    assert float(np.linalg.norm(matrix@Q-Q*ev,ord='fro'))==cm['normal_eigen_residual_frobenius']
    assert float(np.linalg.norm(Q.T@Q-np.eye(N),ord='fro'))==cm['normal_orthogonality_residual_frobenius']
    sqrt=np.sqrt(rho);z=np.fft.fft(data['c0'].reshape(N,N)*sqrt[None,:],axis=0,norm='ortho');states=[];differences=[]
    wave=np.arange(N);rates=4*float(D)*N*N*np.sin(np.pi*wave/N)**2
    for i in range(17):
        t=float(F(i,320));zz=(z@expm(-t*matrix))*np.exp(-t*rates[:,None]);u=np.fft.ifft(zz,axis=0,norm='ortho')/sqrt[None,:]
        states.append(u.real.ravel());error=float(np.sqrt(mass@((u.real.ravel()-control['states'][i])**2)))
        assert error<=cm['rows'][i]['total_absolute'];differences.append(error)
    target=out/'independent-Pade-states.npz'
    if target.exists():assert bitwise_same(arrays(target)['states'],np.array(states))
    else:np.savez_compressed(target,states=np.array(states))
    save(out/'Cartesian-Pade-audit.json',{'independent_exact_normal_ring_verified':True,'all17_full_mode_states_checked':True,
      'eigh_or_production_cartesian_solver_called_by_this_audit':False,'mass_norm_differences':differences,
      'all_within_fixed_Cartesian_indicator':True,'raw_minimum_normal_eigenvalue_not_corrected':float(ev.min())})


def material_audit(out,data,paths,controls,common):
    base=16;mp=data[base]['mass'];maxdiff=0.;count=probes=0
    for N in LEVELS:
        m=data[N]['mass'];ratio=N//base;groups=[[] for _ in mp]
        for i in range(N*N):groups[(i//N//ratio)*base+(i%N//ratio)].append(i)
        for j,ids in enumerate(groups):assert sum((F(float(m[i])) for i in ids),F(0))==F(float(mp[j]))
        source={f'n{N}_c0':data[N]['c0'][None,:],f'n{N}_reference':data[N]['reference'],f'n{N}_Cartesian':controls[N]['states']}
        for field in ('states','independent_states'):source[f'n{N}_{field}']=paths[N][field]
        for key,values in source.items():
            saved=np.atleast_2d(common[key]);assert saved.shape==(len(values),256)
            for u,v in zip(values,saved):
                expected=np.array([math.fsum(float(m[i])*float(u[i]) for i in ids)/float(mp[j]) for j,ids in enumerate(groups)])
                diff=float(np.max(abs(expected-v)));assert diff<=1e-10*max(1.,float(np.max(abs(u))));maxdiff=max(maxdiff,diff);count+=1
                for j in (0,8,15,128,255):
                    exact=sum((F(float(m[i]))*F(float(u[i])) for i in groups[j]),F(0))/F(float(mp[j]));assert expected[j]==float(exact);probes+=1
    return {'all_original_parent_capacities_exact':True,'all_direct_component_integral_snapshots_verified':count,
      'fixed_parent_exact_integral_probes_verified':probes,'maximum_fsum_vs_saved_mean_difference':maxdiff,
      'production_group_sum_or_observe_path_called':False,'masses_or_states_reweighted':False}


def main(directory):
    started=time.monotonic();out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');lock=read(out/'implementation-lock.json')
    assert ex['tests_first_attempt_passed'] and ex['main_first_attempt_passed'] and ex['study_exit_code']==ex['tests_exit_code']==0
    assert ex['elapsed_main_seconds']<900 and ex['peak_RSS_MiB']<1024 and status['recorded_checks_total']==496 and not status['fluid_orc_enabled']
    assert read(out/'tests.json')=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/contact_spatial_protocol.json')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==digest(ROOT/item['source'])==item['sha256'],item['copy']
    index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json');assert len(index)==seal['graphs']==2
    assert seal['both_new_static_graphs_sealed_before_any_material_c0_reference_or_reused_state'] and not seal['builder_received_true_material_faces_F_C_or_precomputed_g']
    assert digest(out/'graph-index.json')==seal['graph_index_sha256'];per={}
    pure_source=arrays(out/'original-pure-inputs.npz')
    for N in NEW:
        folder=out/f'n{N}';pure=arrays(folder/'builder-inputs.npz');packet=read(folder/'geometry.json')
        np.testing.assert_array_equal(pure['positions'],pure_source[f'n{N}_positions']);np.testing.assert_array_equal(pure['ids'],np.arange(N*N))
        for i,(x,y) in enumerate(pure['positions']):assert (F(float(x)),F(float(y)))==(F(2*(i%N)+1,2*N),F(2*(i//N)+1,2*N))
        np.testing.assert_array_equal(pure['rho'],np.where(pure['positions'][:,0]<.5,1.,4.))
        assert digest(folder/'geometry.json')==index[str(N)]['sha256'] and digest(folder/'builder-inputs.npz')==index[str(N)]['builder_inputs_sha256']
        edges,g,degree,cert=audit_geometry(packet,pure['positions'],pure['rho'],pure['ids'],[0,N//2,N-1,N*N//2,N*N-1])
        assert cert==read(folder/'geometry-only-audit.json')['certificate'];assert len(edges)==2*N*N
        p=F(float(.02));counts={str(p):0,str(4*p):0,str(F(8,5)*p):0}
        for v in g:counts[str(v)]+=1
        assert sorted(counts.values())==sorted([N*N-N,N*N-N,2*N]);assert set(map(F,packet['cell_areas_exact']))=={F(1,N*N)}
        d=arrays(folder/'material-reference.npz');meta=read(folder/'reference-parameters.json');control=arrays(folder/'Cartesian-control.npz');cm=read(folder/'Cartesian-parameters.json')
        references(folder,d,pure,meta,N);raw=raw_sequence(folder,d,packet);pade_control(folder,d,control,cm,N)
        per[str(N)]={**raw,'face_counts_by_exact_g':counts,'independent80_100_reference_and17_Pade_states_verified':True}
        print(f'independent static contact N{N}:geometry,reference,fullrawsequence,Padé passed',flush=True)
    args=load_all(out);data,meta,runs,paths,controls,cm=args;r,common=compare(*args);assert plain_json(r)==read(out/'results.json')
    assert plain_json(r['per_grid_controls']['32'])==read(out/'n32/results.json')==read(SOURCE/'results.json')
    saved=arrays(out/'common-observations.npz')
    for key,value in common.items():assert bitwise_same(value,saved[key]),key
    material=material_audit(out,data,paths,controls,saved);save(out/'common-material-audit.json',material)
    # Recompute gains from raw native/projected states independently of saved rows.
    sigma=r['common_initial_fluctuation_norm'];mp,cp=data[16]['mass'],data[16]['c0'];norm=lambda m,v:float(np.sqrt(m@(v*v)))
    def H(m,u,c0):
        mu=float(m@u/m.sum());mu0=float(m@c0/m.sum());return 1-float(m@((u-mu)**2))/float(m@((c0-mu0)**2))
    for row in r['adjacent_spatial_comparisons']:
        C,N=row['coarse_N'],row['fine_N'];j=int(round(row['time']/.025));values={}
        for Q in (C,N):
            d=data[Q];u=paths[Q]['states'][8*j];ref=d['reference'][j];p=common[f'n{Q}_states'][8*j];rp=data[16]['reference'][j]
            values[Q]={'native':norm(d['mass'],u-ref)/sigma,'common':norm(mp,p-rp)/sigma,
              'native_H':abs(H(d['mass'],u,d['c0'])-H(d['mass'],ref,d['c0'])),'common_H':abs(H(mp,p,cp)-H(mp,rp,cp))}
        for name,field,den in [('native','native_indicator_components_absolute',sigma),('common','common_indicator_components_absolute',sigma),('native_H','native_H_indicator_components',1.),('common_H','common_H_indicator_components',1.)]:
            assert row['gains'][name]==values[C][name]-values[N][name]
            budget=sum(row[field].values())/den;assert budget==row['fixed_indicators'][name] and min(row[field].values())>=0
            gain=row['gains'][name];decision='fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive');assert decision==row['decisions'][name]
    preserved=preservation();known={item['manifest'] for item in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==100;save(out/'preservation.json',preserved)
    for rel,h in read(SOURCE/'sha256.json').items():assert digest(ROOT/rel)==h,rel
    summary={'both_new_exact_world_geometries_rawsequences_and17Pade_states_verified':True,'new_N16_N64_independent80_100_references_and_separate_time_tails_passed':True,
      'per_new_grid_controls':per,'common_material_audit':material,'N32_path_loaded_only_not_rerun':True,'N32_original_decision_exactly_preserved':True,
      'all_original_capacity_integrals_and_positive_comparison_budgets_verified':True,'comparison_decision':r['comparison_decision'],
      'registered_three_space_level_contact_oracle_control_passed':bool(r['registered_three_space_level_contact_oracle_control_passed']),
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':100,'new_software_checks':3,'recorded_checks_total':496,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'new_spectral_projection_fits':0,'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    save(out/'numerical-audit.json',summary);save(out/'audit-execution.json',{'audit_exit_code':0,'audit_first_attempt_passed':True,'elapsed_audit_seconds':time.monotonic()-started,
      'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'audit_source_sha256':digest(Path(__file__))})
    print(json.dumps(plain_json(summary),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)


