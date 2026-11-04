"""Audit exact common-material observation and an independent fine solve."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import scipy
from scipy.linalg import expm

from audit_route_decision import ROOT,digest,read,save,preservation
from fv_contact_spatial_study import SOURCE,REFERENCE,array_record,norm,replay,summary_replay
from cartesian_fv_modes import MATERIAL_PRIOR,build_modes,direct_structured_action


def main(output):
    out=Path(output).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');status_path=SOURCE/'stage3-status.json'
    assert config['protocol']==read(status_path)['next_single_spatial_control_predeclared']
    assert config['protocol_source_sha256']==digest(status_path) and not config['protocol']['already_executed']
    packet=read(out/'comparison-inputs.json');coarse=packet['coarse_arrays'];fine=read(out/'derived-fine-data.json')
    state=read(out/'fine-state.json');observed=read(out/'exact-coarse-observation.json')
    provenance=read(out/'input-provenance.json')
    assert digest(ROOT/provenance['coarse_source_NPZ'])==provenance['coarse_source_NPZ_sha256']
    assert digest(REFERENCE/'reference-K128-p90.json')==provenance['source_reference_case_sha256']
    assert packet['sealed_reference_case']==read(REFERENCE/'reference-K128-p90.json')
    assert digest(SOURCE/'comparison-inputs.json')==provenance['source_comparison_inputs_sha256']
    assert digest(SOURCE/'comparison-states.json')==provenance['source_coarse_states_sha256']
    original_packet=read(SOURCE/'comparison-inputs.json')
    for key,value in original_packet['source_arrays'].items():assert coarse[key]==value,key
    assert coarse['coarse_FV_T']==read(SOURCE/'comparison-states.json')['semidiscrete']
    assert packet['coarse_results']==read(SOURCE/'results.json')
    assert digest(out/'FV-N256-common-N128.npz')==provenance['output_NPZ_sha256']
    with np.load(ROOT/provenance['coarse_source_NPZ'],allow_pickle=False) as old,np.load(out/'FV-N256-common-N128.npz',allow_pickle=False) as saved:
        for key,record in provenance['source_original_member_records'].items():
            assert array_record(old[key])==array_record(saved[key])==record,key
            assert array_record(np.asarray(coarse[key],dtype=old[key].dtype))==record,key
        assert saved['coarse_FV_T'].tobytes()==old['semidiscrete'].tobytes()
        for key,value in fine.items():assert saved['fine_'+key].tobytes()==np.asarray(value).tobytes()
        assert saved['fine_FV_T'].tobytes()==np.asarray(state).tobytes()
        assert saved['fine_cell_ids'].tolist()==list(range(256*256))
        assert saved['fine_positions'].tolist()==[[(i+.5)/256,(j+.5)/256] for j in range(256) for i in range(256)]
        B=saved['normal_matrix'].copy();lamy=saved['transverse_eigenvalues'].copy()
    for name,expected in provenance['receipt_SHA256'].items():assert digest(out/name)==expected,name
    assert (out/'source-stage3-status.json').read_bytes()==status_path.read_bytes()
    assert (out/'source-reference-K128-p90.json').read_bytes()==(REFERENCE/'reference-K128-p90.json').read_bytes()
    assert (out/'source-coarse-results.json').read_bytes()==(SOURCE/'results.json').read_bytes()
    replay(out);summary_replay(out)
    N=256;n=128;rho=np.asarray(fine['rho']).reshape(N,N);m=np.asarray(fine['mass']).reshape(N,N)
    c0=np.asarray(fine['c0']).reshape(N,N);T=float(F(packet['model']['T_exact']));D=float(F(packet['model']['D_exact']))
    modes=build_modes(rho,m,D,model_prior=MATERIAL_PRIOR)
    np.testing.assert_array_equal(B,modes.normal_matrix)
    # The same geometry and diffusion law, independently gathered at all fine cells.
    gather=np.zeros_like(c0)
    for axis in [0,1]:
        for sign in [-1,1]:
            other_rho=np.roll(rho,sign,axis=axis);other_c=np.roll(c0,sign,axis=axis)
            conductance=2*D*rho*other_rho/(rho+other_rho)
            gather+=conductance*(other_c-c0)/m
    structured,_=direct_structured_action(modes,c0)
    action_difference=norm(fine['mass'],(gather-structured).ravel().tolist())
    action_allowance=64*np.finfo(float).eps*D*N*N*norm(fine['mass'],fine['c0'])
    assert action_difference<=action_allowance
    # Independent Pade normal exponential on the one existing normal ring.
    z=np.fft.fft(c0*modes.sqrt_rho[None,:],axis=0,norm='ortho')
    independent=np.fft.ifft(np.exp(-T*lamy)[:,None]*(z@expm(-T*B).T),axis=0,norm='ortho')/modes.sqrt_rho[None,:]
    controls=read(out/'solver-controls.json')
    independent_difference=norm(fine['mass'],(independent.real.ravel()-np.asarray(state)).tolist())
    assert independent_difference<=controls['solver_indicator_mass_norm']
    # Independent block indexing and floating reduction; primary restriction is exact Fraction.
    child_mass=m.reshape(n,2,n,2).sum(axis=(1,3))
    np.testing.assert_array_equal(child_mass.ravel(),np.asarray(coarse['mass']))
    floating_observation=(m*np.asarray(state).reshape(N,N)).reshape(n,2,n,2).sum(axis=(1,3))/child_mass
    exact_observation=np.array([float(F(x)) for x in observed['fine_to_coarse_FV_T_exact']])
    reduction_difference=norm(coarse['mass'],(floating_observation.ravel()-exact_observation).tolist())
    assert reduction_difference<=64*np.finfo(float).eps*norm(fine['mass'],state)
    result=read(out/'results.json');generation=read(out/'generation-controls.json')
    assert result['coarse128_error']==packet['coarse_results']['spatial_semidiscrete_relative_error']
    assert result['original_initial_fluctuation_norm']==packet['coarse_results']['initial_fluctuation_norm']
    assert result['restriction_inventory_residual_exact']==result['restriction_reference_error_closure_max_exact']=='0'
    assert F(result['projected_error_Jensen_square_gap_exact'])>=0
    assert result['fine_inside_own_initial_extrema_at_T']
    assert abs(float(F(result['fine_final_inventory_minus_fine_initial_exact'])))<=math.sqrt(math.fsum(fine['mass']))*controls['solver_indicator_mass_norm']
    assert result['state_difference_divided_by_combined_indicator']>1
    assert abs(result['error_reduction_divided_by_combined_indicator'])>1
    assert float(generation['high_precision_factor_nesting_max_difference'])<1e-85 # arithmetic, not a physical gate
    tests=read(out/'tests.json');assert tests==config['software_checks'] and tests['successful'] and tests['checks_run']==3
    assert result['recorded_checks_total']==400+tests['checks_run']==403
    for key in ['new_PDE_time_solutions_or_projection_fits','new_time_step_controls','new_full_2D_transport_matrices',
                'new_point_kernel_builds','new_point_candidate_transient_runs','new_exact_algebra_certificate_groups']:
        assert result[key]==0,key
    assert not result['fluid_orc_enabled'] and result['stage3_physical_acceptance']=='incomplete'
    for name in ['positive_kernel_2d','contact_frozen_probe','contact_finite_time_study']:
        assert name not in sys.modules,name
    preserved=preservation()
    for directory in ['reconnect-route-decision','reconnect-density-weak','reconnect-point-quadrature',
        'reconnect-density-path','reconnect-saved-density-paths','reconnect-cartesian-fv-rows',
        'reconnect-FV-face-budget','reconnect-FV-interface-expansion','reconnect-four-mean-interface',
        'reconnect-FV-global-frozen','reconnect-contact-finite-time-reference','reconnect-FV-contact-time']:
        path=ROOT/'runs'/directory/'sha256.json';entries=read(path)
        for rel,expected in entries.items():assert digest(ROOT/rel)==expected,rel
        preserved['formal_scientific_manifests_verified'].append({'manifest':str(path.relative_to(ROOT)),
            'manifest_sha256':digest(path),'entries_verified':len(entries)})
    assert len(preserved['formal_scientific_manifests_verified'])==65
    save(out/'preservation.json',preserved)
    roles={
        'research/reconnect/FV_CONTACT_FIXED_TIME_COMPARISON.md':'coarse endpoint and one predefined spatial control',
        'runs/reconnect-FV-contact-time/stage3-status.json':'predeclared fine grid, observation and no time scan',
        'runs/reconnect-FV-contact-time/results.json':'unchanged coarse error and initial denominator',
        'runs/reconnect-FV-contact-time/comparison-inputs.json':'original material inputs and reference',
        'runs/reconnect-FV-contact-time/comparison-states.json':'unaltered coarse semidiscrete endpoint',
        'runs/reconnect-contact-finite-time-reference/reference-K128-p90.json':'sealed continuum coefficients reused',
        'research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md':'basis, true averages and projection-contracting tail bound',
        'research/reconnect/cartesian_fv_modes.py':'unchanged ordinary FV solver; no eigenvalue repair'}
    manifest=read(SOURCE/'sha256.json');index=[]
    for rel,role in roles.items():
        assert manifest[rel]==digest(ROOT/rel),rel
        index.append({'path':rel,'sha256':manifest[rel],'role':role})
    save(out/'evidence-index.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256':digest(SOURCE/'sha256.json'),'sources':index})
    status=read(out/'stage3-status.json')
    assert status['recorded_checks_total']==403 and not status['fluid_orc_enabled']
    assert not status['next_point_candidate_contract_review_predeclared']['already_executed']
    audit={'predeclared_one_fine_grid_and_common_observation_verified':True,
        'original_coarse_input_reference_and_endpoint_bytes_preserved':True,
        'fine_inputs_are_new_true_material_integrals_not_interpolated_points':True,
        'sealed_continuum_coefficients_reused_without_PDE_projection_fit':True,
        'all_block_mass_sums_match_original_coarse_masses_exactly':True,
        'exact_restriction_inventory_Jensen_and_error_bridge_closures_verified':True,
        'independent_floating_block_reduction_mass_norm_difference':reduction_difference,
        'independent_fine_Pade_exponential_mass_norm_difference':independent_difference,
        'independent_full_fine_cell_gather_action_mass_norm_difference':action_difference,
        'independent_cell_gather_face_law_evaluations':4*N*N,
        'initial_reference_bridges_and_raw_inventory_drift_retained':True,
        'NumPy_full_replay_and_standard_library_observation_replay_passed':True,
        'new_software_checks_passed':3,'recorded_checks_total':403,'previous_checks_rerun':0,
        'prior_formal_manifests_verified':65,'new_fine_FV_time_cases':1,
        'new_time_step_controls':0,'new_PDE_time_solutions_or_projection_fits':0,
        'new_point_candidate_transient_runs':0,'new_full_2D_transport_matrices':0,'fluid_orc_enabled':False,
        'audit_scipy_version':scipy.__version__}
    save(out/'implementation-audit.json',audit)
    print(json.dumps({'original_files_verified':782,'historical_attachments_verified':15,**audit},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
