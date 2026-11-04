"""Independent original-producer, exact-rate, complete-sequence and grouping audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from shear_probe import make_shear_samples
from shear_voronoi_time16_study import compare

SOURCE=ROOT/'runs/reconnect-shear-voronoi-spatial'


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    spec=read(out/'source-gate-spec.json');proto=read(out/'implementation-protocol.json');lock=read(out/'implementation-lock.json')
    run=read(out/'execution.json');status=read(out/'stage3-status.json');geo_run=read(out/'geometry-audit-execution.json')
    assert spec==read(SOURCE/'next-gate-spec.json') and not spec['already_executed']
    assert read(out/'tests.json')=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    assert run['main_first_attempt_passed'] and run['tests_first_attempt_passed']
    assert run['elapsed_main_seconds']<4000 and run['peak_RSS_MiB']<2048 and run['numerical_norm_estimator_seed']==20261101
    assert geo_run['graphs_verified']==32 and geo_run['all32_pure_geometry_audits_passed'] and not geo_run['mass_c0_reference_or_PDE_output_loaded']
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/voronoi_time16_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    seal=read(out/'coefficient-seal.json');index=read(out/'graph-index.json')
    assert seal['graphs']==len(index)==32 and seal['graph_index_sha256']==digest(out/'graph-index.json')
    assert seal['all32_coefficients_sealed_before_mass_c0_reference_loading'] and not seal['builder_received_mass_c0_reference_true_faces_or_F_C']
    assert seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    for name,p in [('original-comparison.npz',SOURCE/'original-comparison.npz'),('source-spatial-states.npz',SOURCE/'trajectory-states.npz'),
        ('source-coarse-results.json',SOURCE/'source-coarse-results.json'),('source-fine-results.json',SOURCE/'native-results.json'),('source-spatial-results.json',SOURCE/'results.json')]:
        assert digest(out/name)==digest(p),name
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    with np.load(out/'trajectory-source.npz',allow_pickle=False) as z:labels={k:z[k].copy() for k in z.files}
    with np.load(out/'original-comparison.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    with np.load(out/'source-spatial-states.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    with np.load(out/'common-observations.npz',allow_pickle=False) as z:common={k:z[k].copy() for k in z.files}
    matching=read(out/'input-matching.json');assert matching['all_old4_8_records_and_new_nested_subsets_match_by_bytes']
    mp,cp0=d['parent_mass'],d['parent_c0'];pE0=float(mp@((cp0-float(mp@cp0/mp.sum()))**2));psigma=math.sqrt(pE0)
    prel=lambda v:float(np.sqrt(mp@(v*v))/psigma)
    H=lambda u:1-float(mp@((u-float(mp@u/mp.sum()))**2))/pE0
    native={};paths={};max_action=max_projection=max_control=max_raw=0.
    for N in (128,256):
        prefix='parent_' if N==128 else '';m,c0,ids=d[prefix+'mass'],d[prefix+'c0'],d[prefix+'ids'];n=len(m)
        assert np.array_equal(ids,b[f'n{N}_ids']) and np.array_equal(ids,labels[f'n{N}_ids'])
        source=matching['grids'][str(N)]['label_source'];assert digest(ROOT/source)==matching['grids'][str(N)]['label_source_sha256']
        with np.load(ROOT/source,allow_pickle=False) as z:
            for a,k in [('labels','material_label_centers'),('rho0_point','rho0_point'),('ids','ids')]:assert np.array_equal(labels[f'n{N}_{a}'],z[k])
        data,_,_=make_shear_samples(labels[f'n{N}_labels'],ids,m,c0,labels[f'n{N}_rho0_point'],b[f'n{N}_time'])
        assert np.array_equal(data.positions,b[f'n{N}_positions']) and np.array_equal(data.rho,b[f'n{N}_rho'])
        del data
        old_point=ROOT/matching['grids'][str(N)]['old_point_source'];assert digest(old_point)==matching['grids'][str(N)]['old_point_source_sha256']
        with np.load(old_point,allow_pickle=False) as z:
            for K,stride in [(4,4),(8,2)]:
                for key in ['time','positions','rho']:assert np.array_equal(b[f'n{N}_{key}'][::stride],z[f'k{K}_{key}'])
        native[str(N)]=read(out/f'n{N}/native-results.json')
        with np.load(out/f'n{N}/trajectory-states.npz',allow_pickle=False) as z:paths[str(N)]={k:z[k].copy() for k in z.files}
        p=paths[str(N)];primary=p['states'];control=p['independent_states'];mm=tuple(F(float(v)) for v in m)
        E0=float(m@((c0-float(m@c0/m.sum()))**2));sigma=math.sqrt(E0);rel=lambda v:float(np.sqrt(m@(v*v))/sigma)
        assert sigma==native[str(N)]['initial_mass_fluctuation_norm'];np.testing.assert_array_equal(primary[0],c0);np.testing.assert_array_equal(control[0],c0)
        last=c0.copy();dt=F(1,320);cum_machine=cum_tail=0.
        for step in range(16):
            item=index[f'n{N}-step{step:02d}'];row=native[str(N)]['steps'][step]
            assert (item['N'],item['K'],item['step'],item['frame'])==(N,16,step,2*step+1)
            assert item['midpoint_time_saved']==float(b[f'n{N}_midpoint_time'][step])==row['midpoint_time_saved']
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            receipt=read(out/f'n{N}/geometry-only-audit-step{step:02d}.json')
            assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==seal['builder_packet_sha256']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py')
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time16_geometry.py')
            assert receipt['pure_geometry_only_no_mass_c0_reference_loaded'] and receipt['certificate']['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
            cache=read(out/item['file']);edges=np.array(cache['edges'],dtype=np.int64);g=list(map(F,cache['conductances_exact']));assert np.array_equal(cache['ids'],ids)
            del cache;gc.collect();degree=[F(0)]*n
            for (i,j),w in zip(edges,g):degree[i]+=w;degree[j]+=w
            Lambda=math.ceil(max(a/v for a,v in zip(degree,mm)));assert Lambda==row['operator']['Lambda_exact_integer']
            adj=[{} for _ in m]
            for (i,j),w in zip(edges,g):
                assert j not in adj[i] and i not in adj[j]
                adj[i][j]=float(w/(Lambda*mm[i]));adj[j][i]=float(w/(Lambda*mm[j]))
            for i,(w,v) in enumerate(zip(degree,mm)):
                diag=1-w/(Lambda*v);assert diag>=0;adj[i][i]=float(diag)
            ptr=[0];columns=[];values=[]
            for a in adj:
                for j in sorted(a):columns.append(j);values.append(a[j])
                ptr.append(len(columns))
            P=csr_matrix((values,columns,ptr),shape=(n,n))
            with np.load(out/f'n{N}/action-step{step:02d}.npz',allow_pickle=False) as z:
                for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:assert np.array_equal(z[key],v),key
                gf=np.array(list(map(float,g)));assert np.array_equal(gf,z['g_float']);weights=z['weights'].copy()
                A=csr_matrix((z['symmetric_data'],z['symmetric_indices'],z['symmetric_indptr']),shape=tuple(z['shape']))
            zexact=Lambda*dt;M=len(weights)-1;assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(zexact) and row['poisson']['M']==M
            tail=zexact**(M+1)/math.factorial(M+1)/(1-zexact/(M+2));assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            ratio=zexact/(M+1);assert ratio>=1 or zexact**M/math.factorial(M)/(1-ratio)>F(1,10**16)
            degree_max=int(np.bincount(edges.ravel(),minlength=n).max())
            machine=float(64*np.finfo(float).eps*(degree_max+1)*(M+1)*(1+float(zexact))*max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
            cum_machine+=machine;cum_tail+=float(tail)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
            assert machine==row['machine_indicator_absolute_this_interval'] and cum_machine==row['cumulative_machine_indicator_absolute'] and cum_tail==row['cumulative_tail_indicator_absolute']
            v=last.copy();w=math.exp(-float(zexact));following=w*v;assert weights[0]==w
            for k in range(1,M+1):v=P@v;w*=float(zexact)/k;assert weights[k]==w;following+=w*v
            np.testing.assert_array_equal(following,control[step+1]);last=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            delta=rel(primary[step+1]-following);assert delta==row['primary_control_relative_difference'];assert delta*sigma<=cum_machine+cum_tail;max_control=max(max_control,delta)
            u=primary[step];i,j=edges.T;net=np.zeros(n);flux=gf*(u[j]-u[i]);np.add.at(net,i,flux);np.add.at(net,j,-flux);root=np.sqrt(m)
            difference=float(np.sqrt(m@((A@(root*u)/root-net/m)**2)));assert difference<1e-10*max(1.,float(np.sqrt(m@((net/m)**2))));max_action=max(max_action,difference)
            for u,chi in zip((primary[step],p['exponential_half_states'][step],primary[step+1]),row['chi_start_half_end']):
                assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
                assert abs(-float((root*u)@(A@(root*u)))-chi)<1e-10*max(1.,chi)
            drift=float(m@(primary[step+1]-c0));assert drift==row['raw_mass_change_from_c0'];assert abs(drift)/float(m@c0)<1e-10;max_raw=max(max_raw,abs(drift))
            assert primary[step+1].min()>=c0.min()-1e-10 and primary[step+1].max()<=c0.max()+1e-10
            save(out/f'n{N}/geometry-action-audit-step{step:02d}.json',{'coefficient_sha256':item['sha256'],'geometry_receipt_sha256':digest(out/f'n{N}/geometry-only-audit-step{step:02d}.json'),
                'exact_rates_and_whole_control_sequence_verified':True,'exact_minimal_tail_and_raw_weights_verified':True,'shared_vs_symmetric_action_mass_norm_difference':difference})
            print(f'independent action/whole-sequence audit:n{N} step{step+1}/16',flush=True)
            del A,P,adj,g,degree,edges;gc.collect()
        for k,row in enumerate(native[str(N)]['observations']):
            assert rel(primary[8*k]-d[prefix+'reference_states'][k])==row['relative_concentration_error']
    for key,state in [('fine_parent_states16',paths['256']['states']),('fine_parent_independent_states16',paths['256']['independent_states'])]:
        for a,p in zip(state,common[key]):
            independent=np.bincount(d['parent_index'],weights=d['mass']*a,minlength=len(mp))/mp
            delta=float(np.sqrt(mp@((p-independent)**2)));assert delta<64*np.finfo(float).eps*max(1.,float(np.sqrt(d['mass']@(a*a))));max_projection=max(max_projection,delta)
    result,projected=compare(d,native,paths,read(out/'source-fine-results.json'),read(out/'source-coarse-results.json'),old)
    assert result==read(out/'results.json')
    for key,v in projected.items():assert np.array_equal(v,common[key]),key
    for k,row in enumerate(result['observations']):
        coarse=paths['128']['states'][8*k];fine=common['fine_parent_states16'][8*k];ref=d['parent_reference_states'][k]
        assert prel(coarse-ref)==row['coarse16_error_common'] and prel(fine-ref)==row['fine16_error_common']
        assert H(coarse)==row['coarse16_H'] and H(fine)==row['fine16_H_common']
        assert all(v>=0 for v in row['state_indicator_components'].values()) and sum(row['state_indicator_components'].values())==row['combined_state_indicator']
        if k:
            gain=prel(coarse-ref)-prel(fine-ref);budget=row['combined_state_indicator'];expected='fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
            assert expected==row['state_decision']
            gainH=abs(H(coarse)-H(ref))-abs(H(fine)-H(ref));budgetH=row['combined_H_indicator'];expectedH='fine_better_resolvably' if gainH>budgetH else ('fine_worse_resolvably' if gainH < -budgetH else 'comparison_inconclusive')
            assert expectedH==row['H_decision']
    assert read(out/'source-spatial-results.json')['comparison_decision']=='spatial_comparison_inconclusive'
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
        {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==82 and status['recorded_checks_total']==449+4==453
    assert status['new_candidate_graphs']==32 and status['registered16_interval_control_executed'] and not status['fluid_orc_enabled']
    save(out/'preservation.json',preserved)
    summary={'all32_geometries_and_exact_rates_verified':True,'old_producer_new_time_inputs_reproduced_by_bytes':True,
        'all_complete_uniformization_sequences_and_raw_weights_bitwise_verified':True,'original_parent_projection_independently_grouped':True,
        'fixed8_16_indicators_and_old4_8_inconclusive_result_verified':True,'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,
        'maximum_independent_projection_mass_norm_difference':max_projection,'maximum_primary_control_relative_difference':max_control,
        'maximum_native_raw_mass_drift':max_raw,'comparison_decision':result['comparison_decision'],
        'new_software_checks':4,'recorded_checks_total':453,'original_files_verified':782,'historical_attachments_verified':15,
        'prior_formal_manifests_verified':82,'new_reference_PDE_solves':0,'old4_8_candidate_reruns':0,
        'general_smooth_flow_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
