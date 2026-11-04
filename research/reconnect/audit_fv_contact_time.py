"""Fixed-time FV source and independent algebraic-solver controls."""
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
from fv_contact_time_study import SOURCE,array_record,norm,replay,summary_replay
from cartesian_fv_modes import MATERIAL_PRIOR,build_modes,coefficients,reconstruct,direct_structured_action


def main(output):
    out=Path(output).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');source_status=SOURCE/'stage3-status.json'
    assert config['protocol']==read(source_status)['next_ordinary_FV_fixed_time_predeclared']
    assert config['protocol_source_sha256']==digest(source_status) and not config['protocol']['already_executed']
    packet=read(out/'comparison-inputs.json');source=packet['source_arrays'];model=packet['model']
    provenance=read(out/'input-provenance.json')
    assert digest(ROOT/provenance['original_snapshot'])==provenance['original_snapshot_sha256']
    assert digest(ROOT/provenance['reference_source'])==provenance['reference_source_sha256']
    assert digest(out/'FV-T0.05.npz')==provenance['output_NPZ_sha256']
    with np.load(ROOT/provenance['reference_source'],allow_pickle=False) as previous,np.load(out/'FV-T0.05.npz',allow_pickle=False) as saved:
        for key,record in provenance['original_source_member_records'].items():
            assert array_record(previous[key])==array_record(saved[key])==record,key
            assert array_record(np.asarray(source[key],dtype=previous[key].dtype))==record,key
        states=read(out/'comparison-states.json')
        for label,value in states.items():assert saved[label].tobytes()==np.asarray(value).tobytes()
        B=saved['normal_matrix'].copy();Q=saved['normal_eigenvectors'].copy();lamx=saved['normal_eigenvalues'].copy()
        lamy=saved['transverse_eigenvalues'].copy();coeff_saved=saved['initial_full_coefficients'].copy()
    for name,expected in provenance['receipt_SHA256'].items():assert digest(out/name)==expected,name
    assert (out/'source-stage3-status.json').read_bytes()==source_status.read_bytes()
    assert (out/'source-reference-results.json').read_bytes()==(SOURCE/'results.json').read_bytes()
    frozen=ROOT/'runs/reconnect-FV-global-frozen'
    assert digest(frozen/'exact-action.json')==provenance['frozen_exact_action_sha256']
    assert packet['saved_frozen_exact_rates']==read(frozen/'exact-action.json')['rates_per_s']
    assert digest(frozen/'frozen-inputs.json')==provenance['original_masks_source_sha256']
    old=read(frozen/'frozen-inputs.json')
    for key,value in packet['original_masks'].items():assert value==old['saved_arrays'][key]
    replay(out);summary_replay(out)
    N=128;T=float(F(model['T_exact']));rho=np.asarray(source['rho']).reshape(N,N)
    mass=np.asarray(source['mass']).reshape(N,N);c0=np.asarray(source['c0']).reshape(N,N)
    modes=build_modes(rho,mass,float(F(model['D_exact'])),model['periods'],model_prior=MATERIAL_PRIOR)
    np.testing.assert_array_equal(B,modes.normal_matrix)
    # Independent face identities for the original seven saved conductances.
    stencil=read(ROOT/'runs/reconnect-FV-interface-expansion/source-stencil.json')
    D=F(model['D_exact'])
    for edge in stencil['edges']:
        a,b=edge['ids'];i=a%N
        if edge['axis']==0:
            left=i if (i+1)%N==b%N else b%N
            assert modes.normal_g_exact[left]==F(edge['conductance_exact'])
        else:assert F(float(rho.ravel()[a]))*D==F(edge['conductance_exact'])
    coeff=coefficients(modes,c0)
    np.testing.assert_array_equal(coeff,coeff_saved)
    assert coeff.shape==(128,128)
    controls=read(out/'solver-controls.json');result=read(out/'results.json')
    assert float(lamx.min())==controls['raw_normal_minimum_eigenvalue']
    # Pade scaling-and-squaring normal exponential, independent of normal eigh.
    zhat=np.fft.fft(c0*modes.sqrt_rho[None,:],axis=0,norm='ortho')
    normal_exponential=expm(-T*B)
    independent=np.fft.ifft(np.exp(-T*lamy)[:,None]*(zhat@normal_exponential.T),axis=0,norm='ortho')/modes.sqrt_rho[None,:]
    exp_difference=norm(source['mass'],(independent.real.ravel()-np.asarray(states['semidiscrete'])).tolist())
    assert exp_difference<=controls['solver_indicator_mass_norm']
    endpoint_residuals={}
    lam=lamy[:,None]+lamx[None,:]
    for label,k in [('BE512',512),('BE1024',1024)]:
        dt=T/k
        previous,_=reconstruct(modes,coeff*np.exp(-(k-1)*np.log1p(dt*lam)))
        value=np.asarray(states[label]).reshape(N,N)
        action,_=direct_structured_action(modes,value)
        residue=value-previous-dt*action
        endpoint_residuals[label]=norm(source['mass'],residue.ravel().tolist())
        assert endpoint_residuals[label]<=controls['solver_indicator_mass_norm']
    # Frobenius residual controls plus exact matrix construction are audited,
    # not repaired to make global inventory or the zero eigenvalue exact.
    assert np.array_equal(B,B.T)
    assert np.all(B[~np.eye(N,dtype=bool)]<=0)
    assert result['all_original_binary_c0_discrete_modes_retained'] and not result['clipping_or_inventory_renormalization']
    assert result['spatial_semidiscrete_relative_error']>result['combined_reference_and_solver_indicator_over_initial_fluctuation']
    for label,row in result['outputs'].items():
        assert row['inside_original_c0_extrema_at_T']
        assert abs(float(F(row['inventory_minus_original_exact_kg'])))<=math.sqrt(math.fsum(source['mass']))*controls['solver_indicator_mass_norm']
    tests=read(out/'tests.json');assert tests==config['software_checks'] and tests['successful'] and tests['checks_run']==4
    assert result['recorded_checks_total']==396+tests['checks_run']==400
    for key in ['new_point_kernel_builds','new_point_candidate_transient_runs','new_reference_time_solutions',
                'new_full_2D_transport_matrices','new_exact_algebra_certificate_groups']:
        assert result[key]==0,key
    assert not result['kernel_finite_time_comparison_performed'] and not result['fluid_orc_enabled']
    assert result['stage3_physical_acceptance']=='incomplete'
    for name in ['positive_kernel_2d','contact_frozen_probe','contact_finite_time_study']:
        assert name not in sys.modules,name
    preserved=preservation()
    for directory in ['reconnect-route-decision','reconnect-density-weak','reconnect-point-quadrature',
        'reconnect-density-path','reconnect-saved-density-paths','reconnect-cartesian-fv-rows',
        'reconnect-FV-face-budget','reconnect-FV-interface-expansion','reconnect-four-mean-interface',
        'reconnect-FV-global-frozen','reconnect-contact-finite-time-reference']:
        path=ROOT/'runs'/directory/'sha256.json';entries=read(path)
        for rel,expected in entries.items():assert digest(ROOT/rel)==expected,rel
        preserved['formal_scientific_manifests_verified'].append({'manifest':str(path.relative_to(ROOT)),
            'manifest_sha256':digest(path),'entries_verified':len(entries)})
    assert len(preserved['formal_scientific_manifests_verified'])==64
    save(out/'preservation.json',preserved)
    roles={
        'research/reconnect/CONTACT_FINITE_TIME_REFERENCE.md':'controlled reference and fixed-time FV protocol',
        'runs/reconnect-contact-finite-time-reference/stage3-status.json':'predeclared original-state solver and time controls',
        'runs/reconnect-contact-finite-time-reference/reference-T0.05.npz':'sealed material-average PDE reference and original c0',
        'runs/reconnect-contact-finite-time-reference/results.json':'retained reference and initial-bridge controls',
        'runs/reconnect-FV-global-frozen/exact-action.json':'sealed full-cell exact FV rate control',
        'runs/reconnect-FV-global-frozen/frozen-inputs.json':'original interface masks and original inputs',
        'runs/reconnect-FV-interface-expansion/source-stencil.json':'seven saved central-resistance face controls',
        'research/reconnect/CONTACT_PERIODIC_SPECTRAL_BASIS.md':'independent PDE reference proof',
        'research/reconnect/cartesian_fv_stream.py':'independent shared-face source operator',
        provenance['original_snapshot']:'original immutable full snapshot'}
    manifest=read(SOURCE/'sha256.json');index=[]
    for rel,role in roles.items():
        assert manifest[rel]==digest(ROOT/rel),rel
        index.append({'path':rel,'sha256':manifest[rel],'role':role})
    save(out/'evidence-index.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256':digest(SOURCE/'sha256.json'),'sources':index})
    status=read(out/'stage3-status.json')
    assert status['recorded_checks_total']==400 and not status['fluid_orc_enabled']
    assert not status['next_single_spatial_control_predeclared']['already_executed']
    audit={'predeclared_one_case_same_c0_time_protocol_verified':True,
        'original_c0_mass_rho_ID_position_and_reference_bytes_verified':True,
        'all_16384_discrete_modes_retained_without_source_projection':True,
        'normal_ring_uses_original_seven_g_and_full_saved_FV_rate_control':True,
        'raw_negative_eigenvalue_and_inventory_drift_not_repaired':True,
        'independent_Pade_exponential_mass_norm_difference':exp_difference,
        'implicit_Euler_last_step_residual_mass_norms':endpoint_residuals,
        'signed_space_time_error_budgets_and_original_masks_verified':True,
        'NumPy_solver_replay_and_standard_library_metric_replay_passed':True,
        'new_software_checks_passed':4,'recorded_checks_total':400,'previous_checks_rerun':0,
        'prior_formal_manifests_verified':64,'new_FV_baseline_time_cases':1,
        'new_reference_time_solutions':0,'new_point_candidate_transient_runs':0,
        'new_full_2D_transport_matrices':0,'fluid_orc_enabled':False,
        'audit_scipy_version':scipy.__version__}
    save(out/'implementation-audit.json',audit)
    print(json.dumps({'original_files_verified':782,'historical_attachments_verified':15,**audit},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
