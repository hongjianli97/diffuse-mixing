"""Independent bisectors, exact rates, whole control sequences and raw grouping."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from audit_shear_voronoi_time import audit_geometry
from shear_voronoi_spatial_study import compare

SOURCE=ROOT/'runs/reconnect-shear-voronoi-time'


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    spec=read(out/'source-gate-spec.json');protocol=read(out/'implementation-protocol.json');lock=read(out/'implementation-lock.json')
    run=read(out/'execution.json');status=read(out/'stage3-status.json')
    assert spec==read(SOURCE/'next-gate-spec.json') and not spec['already_executed']
    assert read(out/'tests.json')=={'successful':True,'checks_run':4,'failures':0,'errors':0,'skipped':0}
    assert run['scientific_main_first_attempt_passed'] and run['tests_first_attempt_passed']
    assert run['elapsed_main_seconds']<2400 and run['peak_RSS_MiB']<2048 and run['norm_estimator_seed']==20261101
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert digest(out/'implementation-protocol.json')==lock['protocol_sha256']
    assert digest(out/'control-at-lock.md')==lock['control_note_sha256']
    assert digest(out/'builder-inputs.npz')==spec['builder_packet_sha256']
    assert digest(out/'original-comparison.npz')==spec['comparison_packet_sha256']
    assert digest(out/'source-coarse-results.json')==digest(SOURCE/'results.json')
    seal=read(out/'coefficient-seal.json');index=read(out/'graph-index.json')
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and seal['graphs']==len(index)==12
    assert seal['all12_coefficients_sealed_before_mass_c0_reference_loading'] and not seal['builder_received_mass_state_reference_true_faces_or_F_C']
    with np.load(out/'builder-inputs.npz',allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    with np.load(out/'original-comparison.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    with np.load(out/'trajectory-states.npz',allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
    native=read(out/'native-results.json');r=read(out/'results.json');coarse=read(out/'source-coarse-results.json')
    m,c0,mp,cp0=d['mass'],d['c0'],d['parent_mass'],d['parent_c0'];n=len(m)
    mm=tuple(F(float(v)) for v in m);mu0=float(m@c0/m.sum());E0=float(m@((c0-mu0)**2));sigma=math.sqrt(E0)
    pmean=float(mp@cp0/mp.sum());pE0=float(mp@((cp0-pmean)**2));psigma=math.sqrt(pE0)
    rel=lambda v:float(np.sqrt(m@(v*v))/sigma)
    prel=lambda v:float(np.sqrt(mp@(v*v))/psigma)
    H=lambda u:1-float(mp@((u-float(mp@u/mp.sum()))**2))/pE0
    assert np.array_equal(d['parent_id_per_child'],d['parent_ids'][d['parent_index']])
    assert np.all(np.bincount(d['parent_index'],minlength=len(mp))==4)
    with np.load(SOURCE/'trajectory-states.npz',allow_pickle=False) as z:
        for K in (4,8):assert np.array_equal(d[f'parent_FV_k{K}_states'],z[f'k{K}_states'])
    with np.load(SOURCE/'original-comparison.npz',allow_pickle=False) as z:
        for key in ['mass','c0','ids','reference_times','reference_states','reference_space_difference','reference_time_difference']:
            assert np.array_equal(d['parent_'+key],z[key]),key
    audits=[];max_action_delta=max_projection_delta=max_control_delta=max_raw_mass=0.
    for K in (4,8):
        primary=s[f'k{K}_states'];control=s[f'k{K}_independent_states'];np.testing.assert_array_equal(primary[0],c0)
        np.testing.assert_array_equal(control[0],c0);last=c0.copy();dt=F(1,20*K);cum_machine=cum_tail=0.
        for step in range(K):
            item=index[f'k{K}-step{step:02d}'];row=native['cases'][str(K)]['steps'][step]
            assert (item['K'],item['step'],item['frame'])==(K,step,2*step+1)
            assert item['midpoint_time_saved']==float(b[f'k{K}_midpoint_time'][step])==row['midpoint_time_saved']
            assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
            cache=read(out/item['file']);frame=item['frame']
            receipt_path=out/f'geometry-only-audit-k{K}-step{step:02d}.json'
            if receipt_path.exists():
                receipt=read(receipt_path)
                assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==spec['builder_packet_sha256']
                assert receipt['independent_audit_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_time.py')
                assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_shear_voronoi_spatial_geometry.py')
                assert receipt['pure_geometry_only_no_mass_c0_reference_loaded']
                certificate=receipt['certificate'];edges=np.array(cache['edges'],dtype=np.int64);g=list(map(F,cache['conductances_exact']))
                degree=[F(0)]*n
                for (i,j),w in zip(edges,g):degree[i]+=w;degree[j]+=w
            else:
                edges,g,degree,certificate=audit_geometry(cache,b[f'k{K}_positions'][frame],b[f'k{K}_rho'][frame],b['ids'],protocol['fine_geometry_probe_ids'])
            assert certificate['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
            del cache;gc.collect()
            Lambda=math.ceil(max(a/v for a,v in zip(degree,mm)));assert Lambda==row['operator']['Lambda_exact_integer']
            adj=[{} for _ in m]
            for (i,j),w in zip(edges,g):
                assert j not in adj[i] and i not in adj[j]
                adj[i][j]=float(w/(Lambda*mm[i]));adj[j][i]=float(w/(Lambda*mm[j]))
            for i,(w,v) in enumerate(zip(degree,mm)):
                pd=1-w/(Lambda*v);assert pd>=0;adj[i][i]=float(pd)
            ptr=[0];columns=[];values=[]
            for entries in adj:
                for j in sorted(entries):columns.append(j);values.append(entries[j])
                ptr.append(len(columns))
            P=csr_matrix((values,columns,ptr),shape=(n,n))
            with np.load(out/f'action-k{K}-step{step:02d}.npz',allow_pickle=False) as z:
                for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr),('edges',edges)]:
                    assert np.array_equal(z[key],v),key
                gf=np.array(list(map(float,g)));assert np.array_equal(z['g_float'],gf)
                A=csr_matrix((z['symmetric_data'],z['symmetric_indices'],z['symmetric_indptr']),shape=tuple(z['shape']))
                weights=z['weights'].copy()
            zexact=Lambda*dt;M=len(weights)-1;assert row['dt_exact']==str(dt) and row['poisson']['z_exact']==str(zexact)
            assert M==row['poisson']['M'] and zexact/(M+2)<1
            tail=zexact**(M+1)/math.factorial(M+1)/(1-zexact/(M+2))
            assert tail==F(row['poisson']['tail_upper_exact'])<=F(1,10**16)
            if M:
                previous_ratio=zexact/(M+1)
                assert previous_ratio>=1 or zexact**M/math.factorial(M)/(1-previous_ratio)>F(1,10**16)
            degree_max=int(np.bincount(edges.ravel(),minlength=n).max())
            machine=float(64*np.finfo(float).eps*(degree_max+1)*(M+1)*(1+float(zexact))
                *max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
            cum_machine+=machine;cum_tail+=float(tail)*math.sqrt(float(m.sum()))*float(np.max(abs(c0)))
            assert machine==row['machine_indicator_absolute_this_interval'] and cum_machine==row['cumulative_machine_indicator_absolute']
            assert cum_tail==row['cumulative_tail_indicator_absolute']
            v=last.copy();w=math.exp(-float(zexact));following=w*v;assert weights[0]==w
            for k in range(1,M+1):
                v=P@v;w*=float(zexact)/k;assert w==weights[k];following+=w*v
            np.testing.assert_array_equal(following,control[step+1]);last=following
            assert math.fsum(weights)==row['poisson']['raw_weight_sum'] and not row['poisson']['weights_or_state_normalized']
            delta=rel(primary[step+1]-following);assert delta==row['primary_control_relative_difference']
            assert delta*sigma<=cum_machine+cum_tail;max_control_delta=max(max_control_delta,delta)
            u=primary[step];i,j=edges.T;flux=gf*(u[j]-u[i]);net=np.zeros(n)
            np.add.at(net,i,flux);np.add.at(net,j,-flux);root=np.sqrt(m)
            action_delta=float(np.sqrt(m@((A@(root*u)/root-net/m)**2)))
            assert action_delta<1e-10*max(1.,float(np.sqrt(m@((net/m)**2))));max_action_delta=max(max_action_delta,action_delta)
            for u,chi in zip((primary[step],s[f'k{K}_exponential_half_states'][step],primary[step+1]),row['chi_start_half_end']):
                assert float(gf@((u[j]-u[i])**2))==chi and chi>=0
                assert abs(-float((root*u)@(A@(root*u)))-chi)<1e-10*max(1.,chi)
            drift=float(m@(primary[step+1]-c0));assert drift==row['raw_mass_change_from_c0']
            assert abs(drift)/float(m@c0)<1e-10;max_raw_mass=max(max_raw_mass,abs(drift))
            certificate.update({'K':K,'step':step,'coefficient_sha256':item['sha256'],
                'whole_uniformization_sequence_bitwise_verified':True,'exact_minimal_tail_degree_verified':True,
                'symmetric_vs_shared_action_mass_norm_difference':action_delta})
            save(out/f'geometry-action-audit-k{K}-step{step:02d}.json',certificate);audits.append(certificate)
            print(f'independent N256 geometry/action/whole-control audit:k{K} step{step+1}/{K}',flush=True)
            del A,P,adj,g,degree,edges;gc.collect()
        for suffix in ['states','independent_states']:
            actual=s[f'k{K}_{suffix}'];projected=s[f'k{K}_parent_'+suffix]
            for a,p in zip(actual,projected):
                separate=np.bincount(d['parent_index'],weights=m*a,minlength=len(mp))/mp
                delta=float(np.sqrt(mp@((p-separate)**2)))
                assert delta<64*np.finfo(float).eps*max(1.,float(np.sqrt(m@(a*a))))
                max_projection_delta=max(max_projection_delta,delta)
    derived={'k4_states':s['k4_states'],'k8_states':s['k8_states'],
             'k4_independent_states':s['k4_independent_states'],'k8_independent_states':s['k8_independent_states']}
    recomputed,projected=compare(d,native['cases'],derived,coarse)
    assert recomputed==r
    for key,v in projected.items():assert np.array_equal(v,s[key]),key
    for k,row in enumerate(r['observations']):
        a=s['k8_parent_states'][4*k];a4=s['k4_parent_states'][2*k]
        c=d['parent_FV_k8_states'][4*k];c4=d['parent_FV_k4_states'][2*k];ref=d['parent_reference_states'][k]
        assert prel(a-ref)==row['fine_error_common'] and prel(c-ref)==row['coarse_error_common']
        assert H(a)==row['fine_H_common'] and H(c)==row['coarse_H'] and H(ref)==row['reference_H']
        parts=row['state_indicator_components'];assert all(v>=0 for v in parts.values())
        assert parts['coarse_4_8']==prel(c4-c) and parts['fine_4_8_common']==prel(a4-a)
        assert parts['twice_parent_reference_space']==2*prel(d['parent_reference_space_difference'][k])
        assert parts['twice_parent_reference_time']==2*prel(d['parent_reference_time_difference'][k])
        assert parts['reference_parent_projection']==prel(s['projected_reference_states'][k]-ref)
        assert sum(parts.values())==row['combined_state_indicator']
        if k:
            gain=prel(c-ref)-prel(a-ref);budget=row['combined_state_indicator']
            decision='fine_better_resolvably' if gain>budget else ('fine_worse_resolvably' if gain < -budget else 'comparison_inconclusive')
            assert decision==row['state_decision']
            gain_H=abs(H(c)-H(ref))-abs(H(a)-H(ref));budget_H=row['combined_H_indicator']
            decision_H='fine_better_resolvably' if gain_H>budget_H else ('fine_worse_resolvably' if gain_H < -budget_H else 'comparison_inconclusive')
            assert decision_H==row['H_decision']
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
        {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),
         'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==81 and status['recorded_checks_total']==445+4==449
    assert status['registered_N256_spatial_gate_executed'] and status['new_candidate_graphs']==12 and status['new_candidate_transient_runs']==2
    assert status['new_PDE_reference_solves']==0 and not status['fluid_orc_enabled'] and status['stage3_physical_acceptance']=='incomplete'
    save(out/'preservation.json',preserved)
    summary={'all12_geometries_and_whole_control_sequences_verified':True,'parent_projection_independently_grouped':True,
        'old_coarse_states_inputs_and_reference_unchanged':True,'comparison_recomputed_with_fixed_indicators':True,
        'maximum_independent_projection_mass_norm_difference':max_projection_delta,'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action_delta,
        'maximum_primary_control_relative_difference':max_control_delta,'maximum_native_raw_mass_change':max_raw_mass,
        'comparison_decision':r['comparison_decision'],'new_software_checks':4,'recorded_checks_total':449,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':81,
        'new_PDE_reference_solves':0,'coarse_candidate_rerun':False,'general_smooth_flow_admitted':False,
        'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
