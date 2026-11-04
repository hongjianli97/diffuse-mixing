"""Audit one saved-graph endpoint without new weights or PDE evaluation."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import numpy as np

from audit_route_decision import ROOT,read,save,digest,preservation
from point_fixed_time_study import SOURCE,SIGMA,record,norm,metrics
from saved_exchange_time import load_stencil,stencil_action,poisson_tail_bound


def main(directory):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists()
    cfg=read(out/'config.json');spec=read(SOURCE/'next-gate-spec.json')
    assert cfg['protocol']==spec and cfg['parent_gate_sha256']==digest(SOURCE/'next-gate-spec.json')
    assert (out/'source-gate-spec.json').read_bytes()==(SOURCE/'next-gate-spec.json').read_bytes()
    assert not spec['already_executed'] and spec['T_exact']=='1/20'
    assert read(out/'tests.json')=={'successful':True,'checks_run':5,'failures':0,'errors':0,'skipped':0}
    status=read(out/'stage3-status.json');execution=read(out/'execution.json');p=read(out/'input-provenance.json')
    assert status['recorded_checks_total']==423+5==428 and status['previous_checks_rerun']==0
    assert execution['tests_exit_code']==execution['study_exit_code']==0
    assert execution['elapsed_candidate_and_control_seconds']<600 and execution['peak_RSS_MiB']<512
    assert digest(out/'original-inputs.npz')==p['input_artifact_sha256']
    assert digest(out/'source-coefficient-cache.json')==p['copied_cache_sha256']==p['source_cache_sha256']==digest(ROOT/spec['source_coefficient_cache'])
    assert p['parent_matched_inputs_sha256']==digest(SOURCE/'matched-time-reference-inputs.npz')
    with np.load(out/'original-inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(SOURCE/'matched-time-reference-inputs.npz',allow_pickle=False) as z:
        for k in z.files:assert record(z[k])==record(data[k])==p['original_array_records'][k],k
    with np.load(ROOT/'runs/reconnect-reconstructed-global-frozen/global-action.npz',allow_pickle=False) as z:
        assert record(z['rate'])==record(data['saved_initial_rate'])
    with np.load(out/'point-T0.05.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    with np.load(out/'modal-blocks.npz',allow_pickle=False) as z:blocks={k:z[k] for k in z.files}
    controls=read(out/'solver-controls.json')
    s=load_stencil(read(out/'source-coefficient-cache.json'),data['mass'],data['ids'],data['positions'])
    root=np.array(list(map(float,s.sqrt_mass)))
    expected_fft=np.fft.fft(data['c0'].reshape(128,128)*root[None,:],axis=0,norm='ortho')
    assert np.array_equal(expected_fft,arrays['initial_full_Fourier_coefficients'])
    assert arrays['initial_full_eigen_coefficients'].shape==(128,128)
    assert blocks['blocks'].shape==blocks['eigenvectors'].shape==(65,128,128)
    assert blocks['eigenvalues'].shape==(65,128)
    z0=np.empty_like(expected_fft);zT=np.empty_like(expected_fft);T=float(F(spec['T_exact']))
    for k in range(128):
        b=min(k,128-k);V=blocks['eigenvectors'][b];lam=blocks['eigenvalues'][b]
        coefficients=expected_fft[k]@V
        assert np.array_equal(coefficients,arrays['initial_full_eigen_coefficients'][k])
        z0[k]=coefficients@V.T;zT[k]=(coefficients*np.exp(T*lam))@V.T
    reinitial=np.fft.ifft(z0,axis=0,norm='ortho')/root[None,:]
    endpoint=np.fft.ifft(zT,axis=0,norm='ortho')/root[None,:]
    endpoint_delta=norm(data['mass'],endpoint.real.ravel()-arrays['point_semidiscrete_T'])
    assert endpoint_delta<=controls['transform_machine_bound']
    assert norm(data['mass'],reinitial.real.ravel()-data['c0'])==controls['initial_transform_roundtrip_mass_norm']
    for b,H in enumerate(blocks['blocks']):
        assert np.array_equal(H,H.T)
        V=blocks['eigenvectors'][b];lam=blocks['eigenvalues'][b]
        assert float(np.linalg.norm(H@V-V*lam,ord='fro'))==blocks['eigen_residuals'][b]
        assert float(np.linalg.norm(V.T@V-np.eye(128),ord='fro'))==blocks['orthogonality_residuals'][b]
    # The original zero-mode raw eigensystem is checked; no zero snapping.
    eigen0=np.linalg.eigh(blocks['blocks'][0])[0]
    assert np.array_equal(eigen0,blocks['eigenvalues'][0])
    assert float(eigen0.max())==controls['raw_zero_mode_largest_eigenvalue']
    direct_initial=stencil_action(s,data['c0']);direct_final=stencil_action(s,arrays['point_semidiscrete_T'])
    assert np.array_equal(direct_final,arrays['graph_action_at_T'])
    assert norm(data['mass'],arrays['modal_initial_action']-data['saved_initial_rate'])==controls['initial_action_minus_saved_exact_rate_mass_norm']
    assert norm(data['mass'],direct_initial-data['saved_initial_rate'])<=controls['initial_action_machine_bound']
    assert controls['initial_action_minus_saved_exact_rate_mass_norm']<=controls['initial_action_machine_bound']
    Lam=spec['uniformization']['uniformization_lambda_integer'];M=spec['uniformization']['Poisson_max_degree_predeclared']
    assert Lam==168 and M==47
    z=Lam*T;weights=[math.exp(-z)];alternate_weights=weights.copy()
    for n in range(1,M+1):
        # Reproduce the saved implementation's operation order exactly, while
        # retaining a separately parenthesized recurrence as a rounding check.
        weights.append(weights[-1]*(z/n))
        alternate_weights.append(alternate_weights[-1]*z/n)
    assert np.array_equal(weights,arrays['uniformization_weights'])
    recurrence_rounding_difference=float(np.max(abs(np.array(weights)-alternate_weights)))
    assert recurrence_rounding_difference<=M*np.finfo(float).eps*max(weights)
    assert controls['uniformization_raw_weight_sum']==math.fsum(weights)
    tail=poisson_tail_bound(F(Lam)*F(spec['T_exact']),M)
    assert str(tail)==controls['Poisson_tail_bound_exact']==spec['uniformization']['Poisson_tail_bound_exact']
    assert controls['minimum_raw_embedded_diagonal']>=0
    assert norm(data['mass'],arrays['point_semidiscrete_T']-arrays['independent_uniformization_T'])==controls['modal_minus_uniformization_mass_norm']
    assert controls['modal_minus_uniformization_mass_norm']<=controls['independent_action_numerical_allowance_mass_norm']
    result=metrics(data,arrays,controls,read(out/'source-FV-results.json'))
    assert result==read(out/'results.json')
    assert result['initial_fluctuation_norm_locked']==SIGMA
    assert result['comparison_decision']=='point_endpoint_worse_than_saved_FV_resolvably'
    assert result['point_minus_saved_FV_error']>result['candidate_plus_FV_comparison_indicator']
    assert result['outputs']['point']['inventory_drift_display']!=0
    assert result['outputs']['point']['inside_original_initial_extrema']
    assert result['no_old_endpoint_harmonic_T_baseline_no_improvement_claim_over_it']
    for obj in [result,status]:
        assert not obj['finite_node_candidate_admitted'] and not obj['fluid_orc_enabled']
        assert not obj['contact_transient_physically_admitted'] and obj['stage3_physical_acceptance']=='incomplete'
    for k in ['new_candidate_weights','new_kernel_builds','new_density_path_builds','new_global_2D_matrices','new_full_propagator_matrices','new_PDE_reference_solves','new_exact_algebra_certificate_groups']:
        assert status[k]==0,k
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']
    prior=prior+[{'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'manifest_sha256':digest(SOURCE/'sha256.json'),'entries_verified':len(read(SOURCE/'sha256.json'))}]
    for item in prior:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==73
    save(out/'preservation.json',preserved)
    audit={'all_original_state_and_existing_time_reference_hashes_verified':True,
        'saved_electrical_coefficients_copied_without_rebuilding_or_modification':True,
        'all_128_modes_and_raw_zero_eigenvalue_verified':True,
        'saved_modal_action_reconstruction_difference_mass_norm':endpoint_delta,
        'all_raw_blocks_symmetry_eigen_residuals_and_initial_action_verified':True,
        'unrenormalized_uniformization_weights_and_fixed_tail_verified':True,
        'separately_parenthesized_weight_recurrence_maximum_difference':recurrence_rounding_difference,
        'endpoint_I_H_raw_inventory_and_error_metrics_replay_exactly':True,
        'point_worse_than_FV_by_more_than_saved_reference_solver_indicators':True,
        'unfavorable_result_preserved_no_retuning_or_physical_admission':True,
        'one_fixed_endpoint_case_not_a_time_parameter_scan':True,
        'new_software_checks':5,'recorded_checks_total':428,'previous_checks_rerun':0,
        'new_candidate_weights':0,'new_PDE_reference_solves':0,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':73,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'time-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    main(parser.parse_args().directory)
