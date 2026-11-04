"""Independent physical-union grouping, exact-rate and complete action-sequence audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from shear_union64_study import SOURCE,arrays,native_data,load_old,compare,classify,COPIES
from shear_probe import shear_map


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');execution=read(out/'execution.json');status=read(out/'stage3-status.json')
    lock=read(out/'implementation-lock.json');index=read(out/'graph-index.json');seal=read(out/'coefficient-seal.json')
    assert config['parent_manifest_entries_verified']==2174 and config['recorded_checks_total']==457
    assert read(out/'tests.json')=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    assert execution['main_first_attempt_passed'] and execution['tests_first_attempt_passed']
    assert execution['elapsed_main_seconds']<600 and execution['peak_RSS_MiB']<1024
    assert execution['numerical_norm_estimator_seed']==20261101
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/union64_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    assert seal['graphs']==len(index)==24 and seal['graph_index_sha256']==digest(out/'graph-index.json')
    assert seal['all24_coefficients_sealed_before_mass_c0_reference_loading'] and not seal['builder_received_mass_c0_reference_true_faces_or_F_C']
    assert seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==24 and geo['all24_pure_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_PDE_output_loaded'] and not geo['geometry_builder_rerun']
    for name,p in COPIES.items():assert digest(out/name)==digest(p),name
    d=arrays(out/'original-comparison.npz');data=native_data(d);saved=arrays(out/'union-definition.npz')
    labels=arrays(out/'trajectory-source.npz');b=arrays(out/'builder-inputs.npz');oldpoints=arrays(SOURCE/'trajectory-source.npz')
    for name,v in oldpoints.items():np.testing.assert_array_equal(v,labels[name])
    n=64;axis=(np.arange(n)+.5)/n;expected=np.stack(np.meshgrid(axis,axis,indexing='xy'),axis=-1).reshape(-1,2)
    np.testing.assert_array_equal(expected,labels['labels']);np.testing.assert_array_equal(np.arange(n*n,dtype=np.int64),labels['ids'])
    for N in (64,128,256):
        a=labels['labels' if N==64 else f'n{N}_labels'];r=1+.25*np.cos(2*np.pi*a[:,0])
        axisN=(np.arange(N)+.5)/N
        np.testing.assert_array_equal(a,np.stack(np.meshgrid(axisN,axisN,indexing='xy'),axis=-1).reshape(-1,2))
        np.testing.assert_array_equal(r,labels['rho0_point' if N==64 else f'n{N}_rho0_point'])
        if N!=64:np.testing.assert_array_equal(d['parent_ids' if N==128 else 'ids'],labels[f'n{N}_ids'])
    for K in (8,16):
        times=b[f'k{K}_time'];s=.25*times/.05
        x,y=shear_map(expected[None,:,0],expected[None,:,1],s[:,None],times[:,None])
        pos=np.stack([x,np.broadcast_to(y,x.shape)],axis=-1)%1.
        np.testing.assert_array_equal(pos,b[f'k{K}_positions'])
        np.testing.assert_array_equal(np.broadcast_to(labels['rho0_point'],(2*K+1,n*n)),b[f'k{K}_rho'])
    for name,v in data[64].items():np.testing.assert_array_equal(saved[name],v)
    mp,cp=data[64]['mass'],data[64]['c0'];sigma=math.sqrt(float(mp@((cp-float(mp@cp/mp.sum()))**2)))
    max_union_rounding=0.;max_group=0.
    indices={}
    for N in (128,256):
        row,col=np.indices((N,N));idx=((row//(N//64))*64+col//(N//64)).ravel();indices[N]=idx
        np.testing.assert_array_equal(idx,labels[f'n{N}_to_n64'])
    # Independently sum each original physical integral as exact rational arithmetic.
    m,c=d['parent_mass'],d['parent_c0'];refs=d['parent_reference_states'];idx=indices[128]
    exact_mass=[F(0)]*4096;exact_component=[[F(0)]*4096 for _ in range(4)]
    for i,j in enumerate(idx):
        mi=F(float(m[i]));exact_mass[j]+=mi
        for k,u in enumerate((c,*refs)):exact_component[k][j]+=mi*F(float(u[i]))
    for j,mj in enumerate(exact_mass):
        delta=abs(F(float(mp[j]))-mj);max_union_rounding=max(max_union_rounding,float(delta))
        assert float(delta)<=16*np.finfo(float).eps*float(mj)
        for u,q in zip((cp,*data[64]['reference']),exact_component):
            err=abs(F(float(u[j]))-q[j]/mj);assert float(err)<64*np.finfo(float).eps
    runs,paths=load_old(out);max_action=max_control=max_raw=0.
    for K in (8,16):
        paths[64,K]=arrays(out/f'k{K}/trajectory-states.npz');runs[64,K]=read(out/f'k{K}/native-results.json')
        path=paths[64,K];run=runs[64,K];mm=tuple(F(float(v)) for v in mp);dt=F(1,20*K)
        np.testing.assert_array_equal(path['states'][0],cp);np.testing.assert_array_equal(path['independent_states'][0],cp)
        last=cp.copy();cum_machine=cum_tail=0.;previous_variance=float(mp@((cp-float(mp@cp/mp.sum()))**2))
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];row=run['steps'][step]
            assert (item['N'],item['K'],item['step'],item['frame'])==(64,K,step,2*step+1)
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            assert item['midpoint_time_saved']==float(b[f'k{K}_midpoint_time'][step])==row['midpoint_time_saved']
            receipt=read(out/f'k{K}/geometry-only-audit-step{step:02d}.json')
            assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==seal['builder_packet_sha256']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py')
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_union64_geometry.py')
            assert receipt['pure_geometry_only_no_mass_c0_reference_loaded']
            assert receipt['certificate']['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
            packet=read(out/item['file']);edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
            np.testing.assert_array_equal(packet['ids'],labels['ids']);del packet;gc.collect();degree=[F(0)]*4096
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
            P=csr_matrix((vals,cols,ptr),shape=(4096,4096));act=arrays(out/f'k{K}/action-step{step:02d}.npz')
            for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:np.testing.assert_array_equal(v,act[key])
            gf=np.array(list(map(float,g)));np.testing.assert_array_equal(gf,act['g_float']);weights=act['weights']
            A=csr_matrix((act['symmetric_data'],act['symmetric_indices'],act['symmetric_indptr']),shape=tuple(act['shape']))
            ze=Lambda*dt;M=len(weights)-1;assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(ze) and row['poisson']['M']==M
            tail=ze**(M+1)/math.factorial(M+1)/(1-ze/(M+2));assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            ratio=ze/(M+1);assert ratio>=1 or ze**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            degree_max=int(np.bincount(edges.ravel(),minlength=4096).max())
            machine=float(64*np.finfo(float).eps*(degree_max+1)*(M+1)*(1+float(ze))*max(1.,math.sqrt(float(mp.sum())))*max(1.,float(np.max(abs(cp)))))
            cum_machine+=machine;cum_tail+=float(tail)*math.sqrt(float(mp.sum()))*float(np.max(abs(cp)))
            assert machine==row['machine_indicator_absolute_this_interval'] and cum_machine==row['cumulative_machine_indicator_absolute'] and cum_tail==row['cumulative_tail_indicator_absolute']
            v=last.copy();w=math.exp(-float(ze));following=w*v;assert weights[0]==w
            for j in range(1,M+1):v=P@v;w*=float(ze)/j;assert weights[j]==w;following+=w*v
            np.testing.assert_array_equal(following,path['independent_states'][step+1]);last=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            delta=float(np.sqrt(mp@((path['states'][step+1]-following)**2)))
            assert delta==row['primary_control_mass_norm_difference'] and delta<=cum_machine+cum_tail;max_control=max(max_control,delta/sigma)
            u=path['states'][step];i,j=edges.T;net=np.zeros(4096);flux=gf*(u[j]-u[i]);np.add.at(net,i,flux);np.add.at(net,j,-flux);root=np.sqrt(mp)
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
    common=arrays(out/'common-observations.npz')
    for N in (128,256):
        idx=indices[N];m=data[N]['mass']
        for K in (8,16):
            for field in ('states','independent_states'):
                projected=common[f'n{N}_k{K}_{field}']
                for raw,stored in zip(paths[N,K][field],projected):
                    independent=np.bincount(idx,weights=m*raw,minlength=4096)/mp
                    delta=float(np.sqrt(mp@((independent-stored)**2)))
                    assert delta<64*np.finfo(float).eps*max(1.,float(np.sqrt(m@(raw*raw))));max_group=max(max_group,delta)
        for raw,stored in zip(data[N]['reference'],common[f'n{N}_reference']):
            independent=np.bincount(idx,weights=m*raw,minlength=4096)/mp
            delta=float(np.sqrt(mp@((independent-stored)**2)));assert delta<64*np.finfo(float).eps;max_group=max(max_group,delta)
    result,computed=compare(data,runs,paths);assert result==read(out/'results.json')
    for key,v in computed.items():np.testing.assert_array_equal(v,common[key])
    assert sigma==result['common_initial_fluctuation_norm']
    H=lambda u:1-float(mp@((u-float(mp@u/mp.sum()))**2))/(sigma*sigma)
    for row in result['adjacent_comparisons']:
        C,N,k=row['coarse_N'],row['fine_N'],round(row['time']/.025)
        ref=data[64]['reference'][k];coarse=common[f'n{C}_k16_states'][8*k];fine=common[f'n{N}_k16_states'][8*k]
        for name,u in [('coarse',coarse),('fine',fine)]:
            e=float(np.sqrt(mp@((u-ref)**2)))/sigma;assert e==row[name+'_common_error']
        assert abs(H(coarse)-row['coarse_H_common'])<16*np.finfo(float).eps
        for label in ('common','native','H'):
            assert all(v>=0 for v in row[label+'_indicator_components'].values())
            assert sum(row[label+'_indicator_components'].values())==row[label+'_indicator']
            assert classify(row[label+'_gain'],row[label+'_indicator'])==row[label+'_decision']
    assert read(out/'old-results.json')['comparison_decision']=='spatial_improvement_resolved_after_time_control'
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==83 and status['recorded_checks_total']==457 and status['new_candidate_graphs']==24
    assert status['registered_N64_union_control_executed'] and not status['fluid_orc_enabled'];save(out/'preservation.json',preserved)
    raw_mass_delta=saved['n256_mass_sum']-mp
    summary={'all24_geometries_exact_rates_and_complete_raw_sequences_verified':True,
      'original_material_integrals_independently_checked_with_Fractions':True,'direct_ratio2_and4_observation_independently_checked':True,
      'original_source_copies_bitwise_unchanged':True,'maximum_union_mass_rounding_absolute':max_union_rounding,
      'native256_child_sum_vs_union_mass_nonzero_regions':int(np.count_nonzero(raw_mass_delta)),
      'native256_child_sum_vs_union_mass_maximum_relative_difference':float(np.max(abs(raw_mass_delta)/mp)),
      'native256_child_sum_vs_union_mass_raw_total_difference':float(raw_mass_delta.sum()),
      'maximum_independent_grouping_mass_norm_difference':max_group,'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,
      'maximum_primary_control_relative_difference':max_control,'maximum_native_raw_mass_drift':max_raw,
      'fixed_common_native_H_comparison_verified':True,'comparison_decision':result['comparison_decision'],
      'new_software_checks':4,'recorded_checks_total':457,'original_files_verified':782,'historical_attachments_verified':15,
      'prior_formal_manifests_verified':83,'old_candidate_reruns':0,'new_reference_PDE_solves':0,
      'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
