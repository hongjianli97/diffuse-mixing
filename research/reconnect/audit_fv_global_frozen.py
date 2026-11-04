"""Source-byte, independent all-cell gather, and full preservation audit."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import numpy as np

from audit_route_decision import ROOT, digest, read, save, preservation
from fv_global_frozen_study import SOURCE, record, replay


def main(output):
    out=Path(output).resolve()
    assert not (out/'sha256.json').exists(),'never rewrite sealed evidence'
    config=read(out/'config.json');source_status=SOURCE/'stage3-status.json'
    assert config['protocol']==read(source_status)['next_ordinary_FV_global_frozen_predeclared']
    assert config['protocol_source_sha256']==digest(source_status)
    assert not config['protocol']['already_executed']
    provenance=read(out/'input-provenance.json');snapshot=ROOT/provenance['source_snapshot']
    assert digest(snapshot)==provenance['source_snapshot_sha256']
    assert digest(out/'selected-source-observations.npz')==provenance['selected_NPZ_sha256']
    packet=read(out/'frozen-inputs.json');arrays=packet['saved_arrays']
    with np.load(snapshot,allow_pickle=False) as original,np.load(out/'selected-source-observations.npz',allow_pickle=False) as selected:
        for key,expected in provenance['selected_source_member_records'].items():
            raw=original[key][0] if key in ['positions','rho'] else original[key]
            assert record(raw)==record(selected[key])==expected,key
            encoded=np.asarray(arrays[key],dtype=raw.dtype)
            assert record(encoded)==expected,key
            assert raw.tobytes()==selected[key].tobytes()==encoded.tobytes(),key
        data,ptr=original['L_data'],original['L_indptr']
        for key,value in [('L_data',data),('L_indptr',ptr)]:
            assert record(value)==provenance['old_CSR_member_records_used_only_for_row_norm'][key]
        row_abs=np.add.reduceat(np.abs(data),ptr[:-1])/original['mass']
        assert np.asarray(packet['derived_controls']['kernel_absolute_row_sum_per_mass']).tobytes()==row_abs.tobytes()
    for name,expected in provenance['receipt_SHA256'].items():
        assert digest(out/name)==expected,name
    receipts={
        'source-stencil.json':'runs/reconnect-FV-interface-expansion/source-stencil.json',
        'source-two-row-comparison.json':'runs/reconnect-FV-face-budget/source-main-c0-comparison.json',
        'source-local-fluxes.json':'runs/reconnect-FV-face-budget/source-main-c0-local-fluxes.json',
        'source-contact-results.json':'runs/reconnect-contact-frozen/results.json'}
    for name,path in receipts.items():
        assert (out/name).read_bytes()==(ROOT/path).read_bytes(),path
    replay(out)
    exact=read(out/'exact-action.json');result=read(out/'results.json')
    # Independent exact CELL gather, using phase factors rather than the
    # streaming face resistance implementation. No global matrix is assembled.
    n=128;D=F(config['protocol']['D_binary_input_exact'])
    c=list(map(F,arrays['c0']));m=list(map(F,arrays['mass']));rho=arrays['rho']
    assert arrays['ids']==list(range(n*n))
    assert arrays['positions']==[[(i+.5)/n,(j+.5)/n] for j in range(n) for i in range(n)]
    assert rho==[1. if i<n//2 else 4. for j in range(n) for i in range(n)]
    rates=list(map(F,exact['rates_per_s']));mass_rates=list(map(F,exact['mass_rates_kg_per_s']))
    for j in range(n):
        for i in range(n):
            k=j*n+i
            other=[j*n+(i-1)%n,j*n+(i+1)%n,((j-1)%n)*n+i,((j+1)%n)*n+i]
            flux=F(0)
            for v in other:
                factor=F(8,5) if rho[k]!=rho[v] else F(rho[k])
                flux+=D*factor*(c[v]-c[k])
            assert flux==mass_rates[k] and flux/m[k]==rates[k],k
    assert sum(mass_rates,F(0))==0
    assert sum((a*b for a,b in zip(c,mass_rates)),F(0))+F(result['exact_FV_projected_dissipation_kg_per_s'])==0
    error_square=sum((m[i]*(rates[i]-F(arrays['reference_rate'][i]))**2 for i in range(n*n)),F(0))
    assert error_square==F(exact['global_error_square_from_exact_binary_FV_minus_saved_reference'])
    assert result['main_FV_face_law_evaluations']==32768
    assert result['additional_face_locations_beyond_saved_local_stencil']==32761
    assert result['saved_kernel_global_error_minus_original_report']==0
    assert result['FV_two_row_control_over_original_global_norm']==packet['source_controls']['two_row_comparison']['FV_two_row_error_contribution_over_global_norm']
    assert result['original_global_PDE_rate_norm']==packet['source_controls']['two_row_comparison']['original_all_cell_mass_weighted_PDE_rate_norm']
    for region in ['near_contact','far_from_contact']:
        assert result['regional_comparison_with_original_masks'][region]['cells']==packet['source_controls']['contact_results']['regional_comparison'][region]['cells']
    tests=read(out/'tests.json')
    assert tests==config['software_checks'] and tests['successful'] and tests['checks_run']==4
    assert result['recorded_checks_total']==388+tests['checks_run']==392
    for key in ['new_point_kernel_builds','new_point_candidate_graphs','new_global_matrices','new_reference_integrations',
                'new_candidate_transient_runs','new_reference_transient_runs','LP_solves','new_exact_algebra_certificate_groups']:
        assert result[key]==0,key
    assert not result['four_mean_correction_installed'] and not result['fluid_orc_enabled']
    assert result['stage3_physical_acceptance']=='incomplete' and not result['finite_node_candidate_admitted']
    for name in ['scipy','positive_kernel_2d','contact_frozen_probe','cartesian_fv_rows_study']:
        assert name not in sys.modules,name
    preserved=preservation()
    for directory in ['reconnect-route-decision','reconnect-density-weak','reconnect-point-quadrature',
        'reconnect-density-path','reconnect-saved-density-paths','reconnect-cartesian-fv-rows',
        'reconnect-FV-face-budget','reconnect-FV-interface-expansion','reconnect-four-mean-interface']:
        path=ROOT/'runs'/directory/'sha256.json';entries=read(path)
        for rel,expected in entries.items():assert digest(ROOT/rel)==expected,rel
        preserved['formal_scientific_manifests_verified'].append({'manifest':str(path.relative_to(ROOT)),
            'manifest_sha256':digest(path),'entries_verified':len(entries)})
    assert len(preserved['formal_scientific_manifests_verified'])==62
    save(out/'preservation.json',preserved)
    roles={
        'research/reconnect/FOUR_MEAN_INTERFACE_ADMISSIBILITY.md':'restricted rejection and next full-domain scope',
        'runs/reconnect-four-mean-interface/stage3-status.json':'predeclared one-case full-domain frozen comparison',
        'research/reconnect/CARTESIAN_FV_TWO_ROW_CONTROL.md':'limited prior two-row conclusion',
        'runs/reconnect-FV-interface-expansion/source-stencil.json':'seven original g controls and geometry prior',
        'runs/reconnect-FV-face-budget/source-main-c0-comparison.json':'original two-row rates and fixed global denominator',
        'runs/reconnect-FV-face-budget/source-main-c0-local-fluxes.json':'eight original exact outward contributions',
        'research/reconnect/PERIODIC_DENSITY_CONTACT_FROZEN.md':'original contact problem, masks and unadmitted kernel transient',
        'runs/reconnect-contact-frozen/results.json':'saved full-kernel error and reference checks',
        provenance['source_snapshot']:'original input and reference vectors, matrix not rebuilt'}
    manifest=read(SOURCE/'sha256.json');index=[]
    for rel,role in roles.items():
        assert manifest[rel]==digest(ROOT/rel),rel
        index.append({'path':rel,'sha256':manifest[rel],'role':role})
    save(out/'evidence-index.json',{'source_manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256':digest(SOURCE/'sha256.json'),'sources':index})
    status=read(out/'stage3-status.json')
    assert status['recorded_checks_total']==392 and not status['fluid_orc_enabled']
    assert not status['next_same_contact_finite_time_reference_predeclared']['already_executed']
    audit={'predeclared_one_case_full_domain_scope_verified':True,
        'selected_original_member_bytes_and_dtypes_and_JSON_roundtrip_verified':True,
        'old_CSR_read_only_for_uncertainty_row_norm_not_rebuilt_or_applied':True,
        'all_16384_exact_rates_match_independent_phase_factor_cell_gather':True,
        'all_seven_original_g_two_rates_eight_flux_contributions_match':True,
        'exact_global_mass_and_projected_energy_identities_verified':True,
        'original_full_PDE_denominator_and_interface_masks_retained':True,
        'saved_full_kernel_error_reproduced_without_kernel_rebuild':True,
        'standard_library_full_input_replay_identical':True,
        'new_FV_global_frozen_actions':1,'main_FV_shared_face_evaluations':32768,
        'new_software_checks_passed':4,'recorded_checks_total':392,'previous_checks_rerun':0,
        'prior_formal_manifests_verified':62,'new_point_kernel_builds':0,
        'new_global_matrices':0,'new_reference_integrations':0,'fluid_orc_enabled':False}
    save(out/'implementation-audit.json',audit)
    print(json.dumps({'original_files_verified':782,'historical_attachments_verified':15,**audit},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
