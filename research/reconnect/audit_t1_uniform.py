"""Independent exact static geometry/rates, high precision references and84 action intervals."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
import mpmath as mpf
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from t1_uniform_study import SOURCE,state_data
from t1_uniform_core import compare,TIMES
from shear_union64_study import arrays,classify


def reference(N,digits):
    with mpf.workdps(digits):
        pi=mpf.pi;D=mpf.mpf(1)/50;A=mpf.mpf(1)/5
        axis=[(mpf.mpf(i)+mpf.mpf(1)/2)/N for i in range(N)]
        sine=[mpf.sin(2*pi*x) for x in axis];cosine=[mpf.cos(2*pi*y) for y in axis]
        amp=A*(mpf.sin(pi/N)/(pi/N))**2
        mode=[amp*s*c for c in cosine for s in sine]
        lam=8*pi*pi*D;lamh=8*D*N*N*mpf.sin(pi/N)**2
        initial=np.array([float(mpf.mpf(1)/2+q) for q in mode])
        continuum=np.array([[float(mpf.mpf(1)/2+q*mpf.exp(-lam*t)) for q in mode] for t in (0,mpf.mpf(1)/40,mpf.mpf(1)/20)])
        symbol=np.array([[float(mpf.mpf(1)/2+q*mpf.exp(-lamh*t)) for q in mode] for t in (0,mpf.mpf(1)/40,mpf.mpf(1)/20)])
        return initial,continuum,symbol,float(lam),float(lamh)


def main(directory):
    runroot=Path(directory).resolve();out=runroot
    assert not (out/'sha256.json').exists()
    execution=read(out/'execution.json');config=read(out/'config.json');status=read(out/'stage3-status.json')
    assert execution['main_first_attempt_passed'] and execution['tests_first_attempt_passed']
    assert execution['elapsed_main_seconds']<600 and execution['peak_RSS_MiB']<1024
    assert config['parent_manifest_entries_verified']==3342 and status['recorded_checks_total']==470
    assert read(out/'tests.json')=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/t1_uniform_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    seal=read(out/'coefficient-seal.json');assert seal['graphs']==3 and seal['all3_coefficients_sealed_before_mass_c0_reference_symbol']
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    assert not seal['builder_received_mass_c0_reference_true_faces_symbol_or_F_C']
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==3 and geo['all3_pure_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_symbol_loaded'] and not geo['geometry_builder_rerun']
    b=arrays(out/'builder-inputs.npz');data={};runs={};paths={};precision={}
    max_action=max_control=max_raw=max_group=0.
    for N in (16,32,64):
        out=runroot/f'n{N}';n=N*N;index=read(out/'graph-index.json');d=state_data(runroot,N);data[N]=d
        mp,cp=d['mass'],d['c0'];sigma=math.sqrt(float(mp@((cp-float(mp@cp/mp.sum()))**2)))
        np.testing.assert_array_equal(mp,np.full(n,1/n))
        axis=(np.arange(N)+.5)/N
        expected=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
        np.testing.assert_array_equal(b[f'n{N}_positions'],expected)
        np.testing.assert_array_equal(b[f'n{N}_ids'],np.arange(n,dtype=np.int64))
        np.testing.assert_array_equal(b[f'n{N}_rho'],np.ones(n))
        packet=read(out/'geometry.json');g=list(map(F,packet['conductances_exact']))
        assert all(w==F(float(.02)) for w in g)
        cardinal=set()
        for row in range(N):
            for col in range(N):
                i=row*N+col
                for j in (row*N+(col+1)%N,((row+1)%N)*N+col):cardinal.add(tuple(sorted((i,j))))
        assert set(map(tuple,packet['edges']))==cardinal and len(g)==2*n
        # Exact graph checks precede using the independent Fourier symbol.
        c80,r80,s80,l80,lh80=reference(N,80);c100,r100,s100,l100,lh100=reference(N,100)
        for x,y in [(c80,c100),(r80,r100),(s80,s100)]:np.testing.assert_array_equal(x,y)
        norm=lambda v:float(np.sqrt(mp@(v*v)))
        diffs=[norm(cp-c80)]+[norm(x-y) for x,y in zip(d['reference'],r80)]+[norm(x-y) for x,y in zip(d['symbol'],s80)]
        assert max(diffs)<=d['reference_evaluation_indicator_absolute']
        assert abs(d['continuum_lambda']-l80)<=64*np.finfo(float).eps*max(1.,abs(l80))
        assert abs(d['discrete_lambda']-lh80)<=64*np.finfo(float).eps*max(1.,abs(lh80))
        np.savez_compressed(out/'independent-analytic-check.npz',c0_80digits=c80,reference_80digits=r80,symbol_80digits=s80)
        precision[str(N)]={'80_and100_digit_rounded_arrays_bitwise_equal':True,'maximum_main_high_precision_mass_norm_difference':max(diffs),
          'predeclared_reference_evaluation_indicator_absolute':d['reference_evaluation_indicator_absolute'],'D_in_high_precision_reference':'1/50',
          'geometry_D_exact_binary_value':str(F(float(.02))),'binary_parameter_rounding_included_in_evaluation_check':True}
        del packet,g;gc.collect()
        for K in (4,8,16):
            paths[N,K]=arrays(out/f'k{K}/trajectory-states.npz');runs[N,K]=read(out/f'k{K}/native-results.json')
            path=paths[N,K];run=runs[N,K];mm=tuple(F(float(v)) for v in mp);dt=F(1,20*K)
            np.testing.assert_array_equal(path['states'][0],cp);np.testing.assert_array_equal(path['independent_states'][0],cp)
            last=cp.copy();cum_machine=cum_tail=0.;previous_variance=float(mp@((cp-float(mp@cp/mp.sum()))**2))
            for step in range(K):
                item=index[f'k{K}-step{step:02d}'];row=run['steps'][step]
                assert (item['N'],item['K'],item['step'])==(N,K,step)
                assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
                assert item['midpoint_time_saved']==float((2*step+1)*.05/(2*K))==row['midpoint_time_saved']
                receipt=read(out/'geometry-only-audit.json')
                assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==seal['builder_packet_sha256']
                assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py')
                assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_t1_uniform_geometry.py')
                assert receipt['pure_geometry_only_no_mass_c0_reference_loaded']
                assert receipt['certificate']['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
                packet=read(out/item['file']);edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
                np.testing.assert_array_equal(packet['ids'],b[f'n{N}_ids']);del packet;gc.collect();degree=[F(0)]*n
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
                P=csr_matrix((vals,cols,ptr),shape=(n,n));act=arrays(out/f'k{K}/action-step{step:02d}.npz')
                for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(v,act[key])
                gf=np.array(list(map(float,g)));np.testing.assert_array_equal(gf,act['g_float']);weights=act['weights']
                A=csr_matrix((act['symmetric_data'],act['symmetric_indices'],act['symmetric_indptr']),shape=tuple(act['shape']))
                ze=Lambda*dt;M=len(weights)-1;assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(ze) and row['poisson']['M']==M
                tail=ze**(M+1)/math.factorial(M+1)/(1-ze/(M+2));assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
                ratio=ze/(M+1);assert ratio>=1 or ze**M/math.factorial(M)/(1-ratio)>F(1,10**16)
                degree_max=int(np.bincount(edges.ravel(),minlength=n).max())
                machine=float(64*np.finfo(float).eps*(degree_max+1)*(M+1)*(1+float(ze))*max(1.,math.sqrt(float(mp.sum())))*max(1.,float(np.max(abs(cp)))))
                cum_machine+=machine;cum_tail+=float(tail)*math.sqrt(float(mp.sum()))*float(np.max(abs(cp)))
                assert machine==row['machine_indicator_absolute_this_interval'] and cum_machine==row['cumulative_machine_indicator_absolute'] and cum_tail==row['cumulative_tail_indicator_absolute']
                v=last.copy();w=math.exp(-float(ze));following=w*v;assert weights[0]==w
                for j in range(1,M+1):v=P@v;w*=float(ze)/j;assert weights[j]==w;following+=w*v
                np.testing.assert_array_equal(following,path['independent_states'][step+1]);last=following
                assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
                delta=float(np.sqrt(mp@((path['states'][step+1]-following)**2)))
                assert delta==row['primary_control_mass_norm_difference'] and delta<=cum_machine+cum_tail;max_control=max(max_control,delta/sigma)
                u=path['states'][step];i,j=edges.T;net=np.zeros(n);flux=gf*(u[j]-u[i]);np.add.at(net,i,flux);np.add.at(net,j,-flux);root=np.sqrt(mp)
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
    out=runroot;result,computed=compare(data,runs,paths);assert result==read(out/'results.json')
    common=arrays(out/'common-observations.npz');target=data[16]['mass'];sigma0=result['common_initial_fluctuation_norm']
    for key,v in computed.items():np.testing.assert_array_equal(v,common[key])
    for N in (16,32,64):
        rr,cc=np.indices((N,N));ratio=N//16;idx=((rr//ratio)*16+cc//ratio).ravel()
        for K in (4,8,16):
            for field in ('states','independent_states'):
                for raw,stored in zip(paths[N,K][field],common[f'n{N}_k{K}_{field}']):
                    independent=np.bincount(idx,weights=data[N]['mass']*raw,minlength=256)/target
                    delta=float(np.sqrt(target@((independent-stored)**2)))
                    assert delta<64*np.finfo(float).eps;max_group=max(max_group,delta)
        for raw,stored in zip(data[N]['reference'],common[f'n{N}_reference']):
            independent=np.bincount(idx,weights=data[N]['mass']*raw,minlength=256)/target
            delta=float(np.sqrt(target@((independent-stored)**2)));assert delta<64*np.finfo(float).eps;max_group=max(max_group,delta)
    for row in result['adjacent_comparisons']:
        k=round(row['time']/.025);C,N=row['coarse_N'],row['fine_N'];ref=data[16]['reference'][k]
        for label,Q in [('coarse',C),('fine',N)]:
            u=common[f'n{Q}_k16_states'][8*k];e=float(np.sqrt(target@((u-ref)**2)))/sigma0
            assert e==row[label+'_common_error']
        for label in ('common','native','H'):
            assert all(v>=0 for v in row[label+'_indicator_components'].values())
            assert sum(row[label+'_indicator_components'].values())==row[label+'_indicator']
            assert classify(row[label+'_gain'],row[label+'_indicator'])==row[label+'_decision']
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256'];entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==88;save(out/'preservation.json',preserved)
    save(out/'independent-reference-audit.json',precision)
    summary={'all3_geometries_exact_rates_and_all84_complete_raw_intervals_verified':True,
      'all9_independent_uniformization_paths_bitwise_verified':True,'analytic_means_references_symbols_independently_checked_at80_and100_digits':True,
      'postsealed_graph_cardinal_edges_and_exact_D_verified':True,'direct_material_observations_independently_checked':True,
      'maximum_independent_grouping_mass_norm_difference':max_group,'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,
      'maximum_primary_control_native_relative_difference':max_control,'maximum_native_raw_mass_drift':max_raw,
      'comparison_decision':result['comparison_decision'],'T1_registered_case_passed':result['T1_registered_case_passed'],
      'new_software_checks':2,'recorded_checks_total':470,'original_files_verified':782,'historical_attachments_verified':15,
      'prior_formal_manifests_verified':88,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
