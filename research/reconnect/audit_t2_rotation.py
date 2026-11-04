"""Independent world-law, original sparse action and exact covariance audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,hashlib,json,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,save,digest,preservation
from t2_rotation_study import SOURCE,inputs,evidence
from t2_rotation_core import compare
from t2_d0_core import bitwise_same
from shear_union64_study import arrays


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');config=read(out/'config.json')
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed']
    assert ex['elapsed_main_seconds']<300 and ex['peak_RSS_MiB']<1024 and config['parent_manifest_entries_verified']==3849
    assert read(out/'tests.json')=={'successful':True,'checks_run':3,'failures':0,'errors':0,'skipped':0}
    assert status['recorded_checks_total']==477 and status['exact_boundary_adapter_implemented'] and not status['original_geometry_resistance_time_backends_changed']
    assert not status['fluid_orc_enabled'] and not status['arbitrary_float_world_inputs_or_nonorthogonal_lattices_admitted']
    lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/t2_rotation_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    gate=read(SOURCE/'next-gate-spec.json');assert read(out/'source-gate-spec.json')==gate
    assert digest(ROOT/'research/reconnect/periodic_voronoi_fv.py')==gate['original_backend_sha256']
    for item in read(out/'source-copies.json'):assert digest(out/item['copy'])==item['sha256']==digest(ROOT/item['source']),item['copy']
    assert digest(out/'original-pure-inputs.npz')==gate['sources']['pure_points']['sha256']
    seal=read(out/'coefficient-seal.json');index=read(out/'graph-index.json')
    assert seal['graphs']==len(index)==16 and seal['all16_fresh_local_and_world_graphs_sealed_before_material_or_static_data']
    assert not seal['adapter_received_source_local_labels_mass_c0_reference_true_internal_faces_or_F_C']
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and seal['world_trajectory_sha256']==digest(out/'world-trajectory.json')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==16 and geo['all16_independent_world_geometry_audits_passed']
    assert not geo['mass_c0_reference_staticgeometry_or_localcloud_loaded'] and not geo['original_builder_rerun_by_auditor']
    world=read(out/'world-trajectory.json');pure=arrays(out/'original-pure-inputs.npz');a=pure['n32_positions']
    assert world['ids']==pure['n32_ids'].tolist() and len(world['frames'])==33
    local_hash=hashlib.sha256(a.tobytes(order='C')).hexdigest()
    for j,frame in enumerate(world['frames']):
        assert F(frame['time_exact'])==F(j,640);den=4096+j*j;c=F(4096-j*j,den);s=F(128*j,den)
        assert c*c+s*s==1;E=((c,-s),(s,c));B=((c,s),(-s,c));boundary=frame['boundary']
        assert tuple(tuple(map(F,row)) for row in boundary['orientation_rows_exact'])==E
        assert tuple(tuple(map(F,row)) for row in boundary['basis_columns_exact'])==B
        assert list(map(F,boundary['origin_exact']))==[0,0] and list(map(F,boundary['periods_exact']))==[1,1]
        assert frame['rho']==pure['n32_rho'].tolist()
        for p,(x,y) in zip(frame['world_positions_exact'],a):
            x,y=F(float(x)),F(float(y));assert tuple(map(F,p))==(c*x-s*y,s*x+c*y)
    assert world['frames'][-1]['boundary']['orientation_rows_exact']==[['3/5','-4/5'],['4/5','3/5']]
    for step in range(16):
        cert=read(out/f'k16/isometry-step{step:02d}.json')
        assert cert['exact_proper_isometry_verified'] and cert['local_coordinates_float64_lossless'] and cert['period_lengths_float64_lossless']
        assert not cert['points_wrapped_jittered_or_fitted'] and not cert['provided_source_local_labels_or_mass_scalar_reference']
        assert cert['local_coordinate_C_order_sha256']==local_hash
    d,old_run,old_path,old_packet=inputs(out);mp,cp=d['mass'],d['c0'];sigma=math.sqrt(float(mp@((cp-float(mp@cp/mp.sum()))**2)))
    b={'ids':np.array(world['ids'],dtype=np.int64),'midpoint_time':[float(F(world['frames'][2*i+1]['time_exact'])) for i in range(16)]}
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
            receipt=read(out/f'k16/world-geometry-audit-step{step:02d}.json')
            assert receipt['world_cache_sha256']==item['world_sha256']==digest(out/item['world_file'])
            assert receipt['world_trajectory_sha256']==seal['world_trajectory_sha256']
            assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_oriented_world_geometry.py')
            assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_t2_rotation_geometry.py')
            assert not receipt['mass_c0_reference_staticgeometry_or_localcloud_loaded']
            assert receipt['certificate']['all_world_vertices_rebuilt_directly_from_world_bisectors']
            packet=read(out/item['file']);edges=np.array(packet['edges'],dtype=np.int64);g=list(map(F,packet['conductances_exact']))
            wp=read(out/item['world_file'])
            for field in ('ids','edges','image_shifts','cell_vertex_indices','cell_incoming_labels','cell_areas_exact','face_vertex_indices','face_over_distance_exact','conductances_exact','D_exact'):
                assert packet[field]==wp[field],field
            del wp
            baseline_action=arrays(out/f'static-actions/action-step{step:02d}.npz');moving_action=arrays(out/f'k16/action-step{step:02d}.npz')
            for key,v in baseline_action.items():assert bitwise_same(v,moving_action[key]),key
            del baseline_action,moving_action
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
    packets,actions,static=evidence(out,index)
    computed=compare(d,old_run,old_path,runs[32,16],paths[32,16],packets,old_packet,actions,static);assert computed==read(out/'results.json')
    for key,v in paths[32,16].items():assert bitwise_same(v,old_path[key]),key
    assert computed['positive_diffusion_and_nonzero_mixing_preserved']
    assert all(v['state_difference_relative']==v['H_difference']==0 for v in computed['observations'])
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==92;save(out/'preservation.json',preserved)
    summary={'all16_independent_world_geometries_exact_rates_and_full_raw_sequences_verified':True,
      'declared_continuous_world_motion_and_rotating_lattice_verified_exactly':True,'world_local_coefficients_and_static_actions_linked':True,
      'all17_actual_independent_and16_halfstates_bitwise_static':True,'original_source_copies_bitwise_unchanged':True,
      'maximum_shared_vs_symmetric_action_mass_norm_difference':max_action,'maximum_primary_control_relative_difference':max_control,
      'maximum_native_raw_mass_drift':max_raw,'new_software_checks':3,'recorded_checks_total':477,
      'comparison_decision':computed['comparison_decision'],'T2_registered_rotation_case_passed':computed['T2_registered_rotation_case_passed'],
      'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':92,
      'new_reference_PDE_solves':0,'old_candidate_reruns':0,'general_T2_admitted':False,'general_smooth_flow_admitted':False,
      'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False,'kappa_computed':False}
    save(out/'numerical-audit.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
