"""Independent all-interval D0 identity and original sparse zero-branch audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse,gc,json,math
import numpy as np
from scipy.sparse import csr_matrix,eye
from audit_route_decision import ROOT,read,save,digest,preservation
from t2_d0_study import SOURCE,inputs,cached_evidence
from t2_d0_core import evaluate
from shear_union64_study import arrays


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    ex=read(out/'execution.json');status=read(out/'stage3-status.json');config=read(out/'config.json')
    assert ex['main_first_attempt_passed'] and ex['tests_first_attempt_passed']
    assert ex['elapsed_main_seconds']<300 and ex['peak_RSS_MiB']<1024 and config['parent_manifest_entries_verified']==3711
    assert read(out/'tests.json')=={'successful':True,'checks_run':2,'failures':0,'errors':0,'skipped':0}
    assert status['recorded_checks_total']==474 and not status['fluid_orc_enabled'] and not status['general_T2_admitted']
    lock=read(out/'implementation-lock.json')
    for name,h in lock['files'].items():assert digest(ROOT/'research/reconnect'/name)==h,name
    assert lock['protocol_sha256']==digest(out/'implementation-protocol.json')==digest(ROOT/'research/reconnect/t2_d0_protocol.json')
    assert lock['control_note_sha256']==digest(out/'control-at-lock.md')
    assert read(out/'source-gate-spec.json')==read(SOURCE/'next-gate-spec.json')
    for name in ('original-material-reference.npz','original-pure-inputs.npz'):assert digest(out/name)==digest(SOURCE/name),name
    seal=read(out/'coefficient-seal.json');index=read(out/'graph-index.json')
    assert seal['graphs']==len(index)==16 and seal['all16_coefficients_sealed_before_mass_c0_or_old_states']
    assert not seal['builder_received_true_material_faces_F_C_or_scalar']
    assert seal['graph_index_sha256']==digest(out/'graph-index.json') and seal['builder_packet_sha256']==digest(out/'builder-inputs.npz')
    geo=read(out/'geometry-audit-execution.json');assert geo['graphs_verified']==16 and geo['all16_pure_geometry_audits_passed']
    assert not geo['mass_c0_reference_or_PDE_output_loaded'] and not geo['geometry_builder_rerun']
    b=arrays(out/'builder-inputs.npz');pure=arrays(out/'original-pure-inputs.npz');d=inputs(out);m,c0=d['mass'],d['c0'];n=len(m)
    np.testing.assert_array_equal(b['labels'],pure['n32_positions']);np.testing.assert_array_equal(b['ids'],pure['n32_ids'])
    np.testing.assert_array_equal(b['rho0'],np.ones(n));np.testing.assert_array_equal(b['time'],[float(F(j,640)) for j in range(33)])
    a=pure['n32_positions'];alpha=.25*b['time']/.05
    expected=np.empty_like(b['positions'])
    expected[:,:,0]=(a[None,:,0]+alpha[:,None]/(2*np.pi)*np.sin(2*np.pi*a[None,:,0])+.2*b['time'][:,None])%1.
    expected[:,:,1]=(a[None,:,1]-.1*b['time'][:,None])%1.
    np.testing.assert_array_equal(expected,b['positions'])
    rho=1/(1+alpha[:,None]*np.cos(2*np.pi*a[:,0])[None,:]);np.testing.assert_array_equal(rho,b['rho'])
    assert not np.array_equal(b['positions'][0],b['positions'][-1]) and not np.array_equal(b['rho'][0],b['rho'][-1])
    target=arrays(out/'identity-reference.npz');np.testing.assert_array_equal(target['time'],[float(F(j,320)) for j in range(17)])
    for c in target['states']:assert c.dtype==c0.dtype and c.tobytes()==c0.tobytes()
    with np.load(out/'original-material-reference.npz',allow_pickle=False) as z:assert not np.array_equal(z['reference'][-1],target['states'][-1])
    path=arrays(out/'k16/trajectory-states.npz');run=read(out/'k16/native-results.json')
    for key,count in [('states',17),('independent_states',17),('exponential_half_states',16)]:
        assert path[key].shape==(count,n) and path[key].dtype==c0.dtype
        for c in path[key]:assert c.tobytes()==c0.tobytes()
    exact_mass=sum((F(float(v)) for v in m),F(0));exact_component=sum((F(float(w))*F(float(c)) for w,c in zip(m,c0)),F(0))
    exact_variance=sum((F(float(w))*(F(float(c))-exact_component/exact_mass)**2 for w,c in zip(m,c0)),F(0))
    assert exact_variance>0;machine=0.;sym_storage=[];maximum_degree=0
    for step in range(16):
        item=index[f'k16-step{step:02d}'];row=run['steps'][step];frame=2*step+1
        assert (item['N'],item['K'],item['step'],item['frame'])==(32,16,step,frame)
        assert digest(out/item['file'])==item['sha256']==row['coefficient_cache_sha256']
        assert row['midpoint_time_saved']==item['midpoint_time_saved']==float(b['midpoint_time'][step]) and row['dt_exact']=='1/320'
        receipt=read(out/f'k16/geometry-only-audit-step{step:02d}.json')
        assert receipt['cache_sha256']==item['sha256'] and receipt['builder_packet_sha256']==seal['builder_packet_sha256']
        assert receipt['audit_source_sha256']==digest(ROOT/'research/reconnect/audit_d0_geometry_core.py')
        assert receipt['driver_source_sha256']==digest(ROOT/'research/reconnect/audit_t2_d0_geometry.py')
        assert receipt['pure_geometry_only_no_mass_c0_reference_loaded'] and receipt['certificate']['zero_diffusion_limit_without_division_by_zero']
        assert receipt['certificate']['fixed_all_point_halfplane_probes']==item['independent_fixed_probes']
        packet=read(out/item['file']);assert F(packet['D_exact'])==0 and packet['ids']==b['ids'].tolist()
        assert all(F(v)==0 for v in packet['conductances_exact']);edges=np.array(packet['edges'],dtype=np.int64)
        action=arrays(out/f'k16/action-step{step:02d}.npz');np.testing.assert_array_equal(edges,action['edges'])
        assert tuple(action['shape'])==(n,n) and np.count_nonzero(action['g_float'])==0
        A=csr_matrix((action['symmetric_data'],action['symmetric_indices'],action['symmetric_indptr']),shape=(n,n))
        P=eye(n,format='csr');sym_storage.append(len(A.data));assert np.count_nonzero(A.data)==0 and np.count_nonzero(A@np.sqrt(m))==0
        for key,v in [('uniformization_data',P.data),('uniformization_indices',P.indices),('uniformization_indptr',P.indptr)]:np.testing.assert_array_equal(action[key],v)
        assert action['weights'].dtype==np.dtype(float) and action['weights'].tobytes()==np.array([1.]).tobytes()
        assert row['operator']['Lambda_exact_integer']==0 and F(row['operator']['minimum_exact_uniformization_diagonal'])==1
        assert row['poisson']=={'M':0,'z_exact':'0','tail_upper_exact':'0','raw_weight_sum':1.,'weights_or_state_normalized':False}
        degree=int(np.bincount(edges.ravel(),minlength=n).max());maximum_degree=max(maximum_degree,degree)
        allowance=float(64*np.finfo(float).eps*(degree+1)*max(1.,math.sqrt(float(m.sum())))*max(1.,float(np.max(abs(c0)))))
        machine+=allowance;assert allowance==row['machine_indicator_absolute_this_interval'] and machine==row['cumulative_machine_indicator_absolute']
        assert row['cumulative_tail_indicator_absolute']==0 and row['primary_control_mass_norm_difference']==row['primary_control_relative_difference']==0
        assert row['raw_mass_change_from_c0']==row['independent_raw_mass_change_from_c0']==0
        assert row['chi_start_half_end']==[0.,0.,0.] and row['half_variance_loss']==row['Simpson_chi_integral']==row['raw_energy_balance_difference']==0
        assert row['minimum']==float(c0.min()) and row['maximum']==float(c0.max())
        for field in ('state_metrics','independent_metrics'):
            assert row[field]['H']==0 and row[field]['I']==1 and row[field]['mean']==float(m@c0/m.sum())
            assert row[field]['variance']==float(m@((c0-float(m@c0/m.sum()))**2))
        save(out/f'k16/action-audit-step{step:02d}.json',{'cache_sha256':item['sha256'],'exact_D0_rates_A_zero_P_identity_and_raw_weight_one_verified':True,
          'actual_independent_and_halfstates_bitwise_initial':True,'stored_explicit_zero_A_entries_retained':len(A.data)})
        print(f'independent D0 action:{step+1}/16 passed',flush=True);del packet,action,A,P;gc.collect()
    packets,actions=cached_evidence(out);result=evaluate(d,run,path,packets,actions);assert result==read(out/'results.json')
    assert all(v['raw_mass_change']==0 and v['H']==0 and v['I']==1 for v in result['observations'])
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    previous=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
      {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in previous:
        if item['manifest'] in known:continue
        p=ROOT/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=read(p);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==90;save(out/'preservation.json',preserved)
    summary={'all16_zero_geometry_audits_and_original_zero_action_branches_verified':True,'all17_actual_independent_and16_halfstates_bitwise_original':True,
      'independent_identity_reference_verified_not_positiveD_reference':True,'sparse_identity_and_raw_weight_one_verified':True,
      'exact_initial_mass':str(exact_mass),'exact_initial_component_mass':str(exact_component),'exact_initial_centered_variance':str(exact_variance),
      'maximum_reference_error_relative':max(v['state_reference_difference_relative'] for v in result['observations']),
      'maximum_raw_mass_drift':max(abs(v['raw_mass_change']) for v in result['observations']),'maximum_H':max(v['H'] for v in result['observations']),
      'maximum_geometric_degree':maximum_degree,'original_positive_machine_indicator_final':machine,'original_zero_A_stored_entries_per_interval':sym_storage,
      'comparison_decision':result['comparison_decision'],'T2_registered_D0_case_passed':result['T2_registered_D0_case_passed'],
      'new_software_checks':2,'recorded_checks_total':474,'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':90,
      'old_candidate_reruns':0,'new_reference_PDE_solves':0,'kappa_computed':False,'general_T2_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'numerical-audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if not k.startswith('exact_')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
