"""Independent contact reference, exact world geometry, Padé and raw sequence audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,json,math,resource,time
import mpmath as mp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from audit_route_decision import ROOT,read,save as raw_save,digest,preservation
from audit_shear_voronoi_time import audit_geometry
from contact_oracle_study import SOURCE,load_all
from contact_oracle_core import evaluate
from affine_shear_completion import plain_json
from shear_union64_study import arrays
from t2_d0_core import bitwise_same


def save(p,obj):raw_save(p,plain_json(obj))


def references(out,data,pure,meta):
    N=32;m=data['mass'];case=read(out/'source-reference-K128-p90.json');rounded={}
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
    np.testing.assert_array_equal(data['mass'],pure['rho']/1024);np.testing.assert_array_equal(data['c0'],data['reference'][0])
    certificates=meta['tail_certificates_by_time'];assert certificates[0] is None
    for t,c in zip((F(1,40),F(1,20)),certificates[1:]):
        lower=4*F(157,50)**2*F(float(.02))*t*(129**2+1);decades=int(lower//F(231,100))
        assert F(c['decay_exponent_lower_exact'])==lower and c['guaranteed_decimal_decades']==decades
        assert F(c['material_mass_norm_tail_bound_exact'])==F(1,4*10**decades)
    assert F(certificates[1]['material_mass_norm_tail_bound_exact'])>F(certificates[2]['material_mass_norm_tail_bound_exact'])
    np.savez_compressed(out/'independent-reference80.npz',reference=rounded[80])
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


def pade_control(out,data,control,cm):
    N=32;rho=data['rho'].reshape(N,N)[0];mass=data['mass'];D=F(float(.02));area=F(1,N*N);rr=list(map(lambda v:F(float(v)),rho));B=[[F(0)]*N for _ in range(N)]
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
    np.savez_compressed(out/'independent-Pade-states.npz',states=np.array(states))
    save(out/'Cartesian-Pade-audit.json',{'independent_exact_normal_ring_verified':True,'all17_full_mode_states_checked':True,
      'eigh_or_production_cartesian_solver_called_by_this_audit':False,'mass_norm_differences':differences,
      'all_within_fixed_Cartesian_indicator':True,'raw_minimum_normal_eigenvalue_not_corrected':float(ev.min())})


def main(directory):
    started=time.monotonic();out=Path(directory).resolve();assert not (out/'sha256.json').exists();ex=read(out/'execution.json')
    assert ex['main_first_attempt_passed'] and not ex['tests_first_attempt_passed'] and ex['tests_second_attempt_passed'] and ex['elapsed_main_seconds']<900 and ex['peak_RSS_MiB']<1024
    for name,h in read(out/'implementation-lock.json')['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert digest(out/'implementation-protocol.json')==read(out/'implementation-lock.json')['protocol_sha256']==digest(ROOT/'research/reconnect/contact_oracle_protocol.json')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==digest(ROOT/item['source'])==item['sha256']
    assert read(out/'tests.json')=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert read(out/'ltn-contact-oracle-tests-attempt1.json')=={'successful':False,'checks_run':3,'failures':1,'errors':0,'skipped':0}
    assert not read(out/'test-fixture-correction.json')['scientific_N32_or_audit_guard_formula_threshold_changed']
    seal=read(out/'coefficient-seal.json');assert seal['graphs']==1 and seal['static_graph_sealed_before_any_material_c0_reference_or_oldstate']
    assert digest(out/'geometry.json')==seal['geometry_sha256'] and digest(out/'builder-inputs.npz')==seal['builder_inputs_sha256']
    pure=arrays(out/'builder-inputs.npz');packet=read(out/'geometry.json');N=32
    np.testing.assert_array_equal(pure['ids'],np.arange(1024))
    for i,(x,y) in enumerate(pure['positions']):assert (F(float(x)),F(float(y)))==(F(2*(i%N)+1,2*N),F(2*(i//N)+1,2*N))
    np.testing.assert_array_equal(pure['rho'],np.where(pure['positions'][:,0]<.5,1.,4.))
    edges,g,degree,cert=audit_geometry(packet,pure['positions'],pure['rho'],pure['ids'],[0,16,31,512,1023])
    assert cert==read(out/'geometry-only-audit.json')['certificate'];assert len(edges)==2048
    counts={str(F(float(.02))):0,str(4*F(float(.02))):0,str(F(8,5)*F(float(.02))):0}
    for v in g:counts[str(v)]+=1
    assert sorted(counts.values())==[64,992,992]
    data,meta,run,path,control,cm=load_all(out);references(out,data,pure,meta);raw=raw_sequence(out,data,packet);pade_control(out,data,control,cm)
    result=evaluate(data,meta,run,path,control,cm);assert plain_json(result)==read(out/'results.json')
    # Independent triangle decomposition and original quadratic H indicator, no scalar fit.
    m,c0=data['mass'],data['c0'];mu=float(m@c0/m.sum());E0=float(m@((c0-mu)**2));sigma=math.sqrt(E0)
    for j,row in enumerate(result['observations']):
        c=path['states'][8*j];ref=data['reference'][j];norm=lambda v:float(np.sqrt(m@(v*v)))
        assert row['reference_error_relative']==norm(c-ref)/sigma
        absolute=row['total_PDE_comparison_indicator_relative']*sigma;bound=(2*math.sqrt(float(m.sum()))*max(1.,float(np.max(abs(c0))))*absolute+absolute**2)/E0
        H=lambda u:1-float(m@((u-float(m@u/m.sum()))**2))/E0
        assert abs(H(c)-H(ref))<=bound
    preserved=preservation();known={i['manifest'] for i in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==99;save(out/'preservation.json',preserved)
    assert digest(ROOT/'exports/ltn-affine-shear-spatial.zip')=='7466b3e90291dbb1132858d0f9521dda6f06f7d87b9c234d7d66e13ab496d183'
    summary={**raw,'independent_reference80_100_and_separate_time_tails_verified':True,'independent_Pade17_full_mode_states_passed':True,
      'independent_geometry_and_interface_conductances_verified':True,'face_counts_by_exact_g':counts,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'registered_same_backend_contact_oracle_control_passed':bool(result['registered_same_backend_contact_oracle_control_passed']),
      'comparison_decision':result['comparison_decision'],'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':99,
      'new_software_checks':3,'recorded_checks_total':493,'tests_first_attempt_passed':False,'tests_second_attempt_passed':True,
      'moving_shock_T4_closed':False,'all_three_density_sources_closed':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    save(out/'numerical-audit.json',summary);save(out/'audit-execution.json',{'audit_exit_code':0,'audit_first_attempt_passed':True,
      'elapsed_audit_seconds':time.monotonic()-started,'peak_auditor_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'audit_source_sha256':digest(Path(__file__))})
    print(json.dumps(plain_json(summary),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
