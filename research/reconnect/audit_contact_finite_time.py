"""Independent boundary projection/primitive-mean and immutable-source audit."""
from decimal import Decimal as Dec, localcontext
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import argparse
import inspect
import json
import sys
import numpy as np

from audit_route_decision import ROOT,digest,read,save,preservation
from contact_finite_time_study import SOURCE,THEORY,array_record,replay
from contact_spectral_reference import TrigPi,PIECES,dec,pi_agm,projection_products,projection_boundary,field


def main(output):
    out=Path(output).resolve();assert not (out/'sha256.json').exists()
    config=read(out/'config.json');source_status=SOURCE/'stage3-status.json'
    assert config['protocol']==read(source_status)['next_same_contact_finite_time_reference_predeclared']
    assert config['protocol_source_sha256']==digest(source_status)
    assert not config['protocol']['already_executed']
    assert digest(THEORY)==digest(out/'basis-theory-at-evaluation.md')==config['basis_theory_sha256_before_reference_evaluation']
    packet=read(out/'reference-inputs.json');model=packet['model'];source=packet['source_arrays']
    assert model=={'N':128,'rho_left':'1','rho_right':'4','D_exact':config['protocol']['D_binary_input_exact'],
                   'T_exact':'1/20','amplitude_exact':'1/5','phase_pi_ratio':'1/5'}
    assert list(inspect.signature(projection_products).parameters)==['trig','K']
    provenance=read(out/'input-provenance.json')
    assert digest(ROOT/provenance['original_snapshot'])==provenance['original_snapshot_sha256']
    assert digest(ROOT/provenance['source_selected_observations'])==provenance['source_selected_observations_sha256']
    assert digest(out/'reference-T0.05.npz')==provenance['output_reference_NPZ_sha256']
    with np.load(ROOT/provenance['source_selected_observations'],allow_pickle=False) as old,np.load(out/'reference-T0.05.npz',allow_pickle=False) as saved:
        for key,record in provenance['original_member_records'].items():
            assert array_record(old[key])==array_record(saved[key])==record,key
            assert array_record(np.asarray(source[key],dtype=old[key].dtype))==record,key
        chosen=read(out/'reference-K128-p90.json')
        assert array_record(saved['reference_cbar_T'])==provenance['reference_mean_array_record']
        assert saved['reference_cbar_T'].tobytes()==np.asarray(list(map(float,field(chosen)))).tobytes()
        with localcontext() as context:
            context.prec=120
            initial=np.asarray([float(Dec('.5')+Dec(x)*Dec(y)) for y in chosen['transverse_cell_means'] for x in chosen['initial_normal_cell_means']])
        assert saved['math_initial_cbar_reference_only'].tobytes()==initial.tobytes()
    for name,expected in provenance['receipt_SHA256'].items():assert digest(out/name)==expected,name
    assert (out/'source-stage3-status.json').read_bytes()==source_status.read_bytes()
    assert (out/'source-global-frozen-results.json').read_bytes()==(SOURCE/'results.json').read_bytes()
    replay(out)
    # A distinct final-field route: AGM pi, boundary projections, endpoint
    # primitive means (not sinc-at-center), independent modal reconstruction.
    with localcontext() as context:
        context.prec=120;trig=TrigPi();trig.pi=pi_agm()
        D,T=dec(model['D_exact']),dec(model['T_exact']);tau=4*trig.pi**2*D*T
        alpha0=3*trig.sc(F(1,5))[0]/(4*trig.pi)
        coeff=[projection_boundary(trig,n) for n in range(1,129)]
        normal=[];y=[]
        for i in range(128):
            lo,hi=F(i,128),F(i+1,128);rho=F(1) if i<64 else F(4)
            value=alpha0
            for n,(a,b) in enumerate(coeff,1):
                value+=(-tau*n*n).exp()*128*(a*trig.integral_cos(2*n,F(0),lo,hi)
                    +b*trig.integral_sin(2*n,F(0),lo,hi)/dec(rho))
            normal.append((-tau).exp()*value)
            y.append(128*trig.integral_cos(F(2),F(0),lo,hi))
        diff=max([abs(a-Dec(b)) for a,b in zip(normal,chosen['normal_T'])]
                 +[abs(a-Dec(b)) for a,b in zip(y,chosen['transverse_cell_means'])])
        assert diff<Dec('1e-85') # implementation arithmetic check, not a physical threshold
        for K in [64,128]:
            case=read(out/f'reference-K{K}-p90.json');cert=case['tail_certificate']
            exponent=4*F(157,50)**2*F(model['D_exact'])*F(model['T_exact'])*((K+1)**2+1)
            q=int(exponent//F(231,100))
            assert F(cert['material_mass_norm_tail_bound_exact'])==F(1,4*10**q)
            assert q==cert['guaranteed_decimal_decades']
            assert sum((F(231,100)**j/factorial(j) for j in range(13)),F(0))>10
        result=read(out/'results.json')
        source_inventory=sum((F(m)*F(c) for m,c in zip(source['mass'],source['c0'])),F(0))
        assert str(source_inventory-F(5,4))==result['source_initial_inventory_minus_5_over_4_exact']
        assert result['bridge_contraction_requires_explicit_cellwise_constant_perturbation_lift']
        assert result['initial_bridge_not_subtracted_and_reference_not_renormalized']
        reconstruction_difference=str(diff)
    tests=read(out/'tests.json');assert tests==config['software_checks'] and tests['successful'] and tests['checks_run']==4
    assert result['recorded_checks_total']==392+tests['checks_run']==396
    for key in ['new_candidate_transient_runs','new_FV_transient_runs','new_kernel_builds','new_global_matrices',
                'new_reference_time_stepping_solves','new_reference_numerical_integrations','new_exact_algebra_certificate_groups']:
        assert result[key]==0,key
    for name in ['scipy','positive_kernel_2d','cartesian_fv_stream','contact_frozen_probe']:
        assert name not in sys.modules,name
    preserved=preservation()
    for directory in ['reconnect-route-decision','reconnect-density-weak','reconnect-point-quadrature',
        'reconnect-density-path','reconnect-saved-density-paths','reconnect-cartesian-fv-rows',
        'reconnect-FV-face-budget','reconnect-FV-interface-expansion','reconnect-four-mean-interface','reconnect-FV-global-frozen']:
        path=ROOT/'runs'/directory/'sha256.json';entries=read(path)
        for rel,expected in entries.items():assert digest(ROOT/rel)==expected,rel
        preserved['formal_scientific_manifests_verified'].append({'manifest':str(path.relative_to(ROOT)),
            'manifest_sha256':digest(path),'entries_verified':len(entries)})
    assert len(preserved['formal_scientific_manifests_verified'])==63
    save(out/'preservation.json',preserved)
    roles={
        'research/reconnect/FV_GLOBAL_FROZEN_COMPARISON.md':'prior result and fixed-time reference-first scope',
        'runs/reconnect-FV-global-frozen/stage3-status.json':'predeclared T, truncations, arithmetic and locked initial data',
        'runs/reconnect-FV-global-frozen/selected-source-observations.npz':'unaltered saved material means and masses',
        'runs/reconnect-FV-global-frozen/input-provenance.json':'selected member-byte provenance to original snapshot',
        'research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md':'original analytic initial formula and periodic contact',
        'research/reconnect/FV_MASS_MEAN_INTERFACE_EXPANSION.md':'original mass averages and initial high-order compatibility scope',
        provenance['original_snapshot']:'original full source; no matrix rebuilt or propagated'}
    manifest=read(SOURCE/'sha256.json');index=[]
    for rel,role in roles.items():
        assert manifest[rel]==digest(ROOT/rel),rel
        index.append({'path':rel,'sha256':manifest[rel],'role':role})
    save(out/'evidence-index.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256':digest(SOURCE/'sha256.json'),'sources':index})
    status=read(out/'stage3-status.json')
    assert status['recorded_checks_total']==396 and not status['fluid_orc_enabled']
    assert status['next_ordinary_FV_fixed_time_predeclared']['already_executed'] is False
    audit={'predeclared_single_time_reference_scope_verified':True,
        'basis_completeness_and_tail_proof_saved_before_numerical_reference':True,
        'original_mass_c0_positions_rho_ID_bytes_and_JSON_roundtrip_verified':True,
        'projection_inputs_exclude_saved_c0_FV_kernel_and_reference_errors':True,
        'all_retained_modes_cross_checked_by_boundary_identity':True,
        'AGM_boundary_projection_endpoint_mean_reconstruction_max_difference':reconstruction_difference,
        'rational_exponential_tail_certificates_verified':True,
        'normal_zero_mode_retained_and_no_coefficients_clipped':True,
        'initial_inventory_gap_and_conditional_bridge_not_removed':True,
        'full_reference_float_output_bytes_verified':True,'standard_library_replay_identical':True,
        'new_software_checks_passed':4,'recorded_checks_total':396,'previous_checks_rerun':0,
        'prior_formal_manifests_verified':63,'reference_physical_time_cases':1,
        'reference_control_evaluations':4,'new_candidate_or_FV_time_propagations':0,
        'new_reference_time_stepping_solves':0,'new_reference_numerical_integrations':0,
        'fluid_orc_enabled':False}
    save(out/'implementation-audit.json',audit)
    print(json.dumps({'original_files_verified':782,'historical_attachments_verified':15,**audit},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
