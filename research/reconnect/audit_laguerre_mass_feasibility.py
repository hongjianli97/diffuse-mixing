"""Audit Laguerre theory scope, standalone algebra and unchanged historical sources."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import ast
import json
import sys

from audit_route_decision import ROOT,read,save,digest,preservation
from laguerre_mass_feasibility import SOURCE,THEORY,ZERO_FIELDS,BLOCKED,replay


def main(directory,standalone):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists();replay(out)
    cfg=read(out/'config.json');parent=read(SOURCE/'sha256.json')
    assert cfg['parent_manifest_entries_verified']==len(parent)==1723
    assert cfg['parent_manifest_sha256']==digest(SOURCE/'sha256.json')
    assert cfg['parent_theory_scope_sha256']==digest(SOURCE/'next-theory-scope.json')
    assert (out/'theory-at-definition.md').read_bytes()==THEORY.read_bytes()
    receipt=read(out/'definition-receipt.json')
    assert receipt['analytic_formulas_and_scope_saved_before_algebra']
    assert receipt['theory_sha256_before_algebra']==digest(THEORY)
    assert receipt['contract_sha256_before_algebra']==digest(out/'feasibility-contract.json')
    certs=read(out/'algebra-certificates.json');assert read(standalone)==certs
    assert certs['verified_exact_algebra_groups']==8 and len({v['name'] for v in certs['groups']})==8
    assert all(v['exact_algebra_passed'] for v in certs['groups'])
    # Independently use the geometric endpoints, not the polynomial expansion.
    w=certs['witness'];left,right=F(w['cell_0_left']),F(w['cell_0_right'])
    assert right-left==F(w['target_mass_0'])==F(1,4)
    assert 1-(right-left)==F(w['target_mass_1'])==F(3,4)
    assert right<0 and F(w['right_face_signed_distance_0'])==right
    assert F(w['right_face_signed_distance_1'])==F(1,4)-right>0
    assert F(w['seed_0_competitor_score'])==F(1,16)-F(w['weight_difference'])<0
    assert F(w['minimum_mass_0_with_closed_seed_inclusion'])-F(w['target_mass_0'])==F(w['unavoidable_mass_0_residual'])==F(1,12)
    c=read(out/'feasibility-contract.json');s=read(out/'stage3-status.json')
    assert c['existence_proved_in_report'] and not c['weight_uniqueness_in_general_claimed']
    assert c['cannot_force_both_exact_mass_and_seed_interiority_for_all_inputs']
    assert not c['all_seed_interiority_automatic'] and not c['all_positive_face_distances_automatic']
    assert c['absolute_value_of_negative_face_distance_forbidden']
    assert not c['old_half_distance_resistance_valid_for_new_cells']
    assert not c['generic_2D_cut_density_face_conductance_justified']
    assert c['weighted_moments_use_fixed_density_integrals'] and not c['true_geometry_or_mean_identified']
    assert not c['scalar_or_reference_response_used_for_geometry_weights']
    for key in ZERO_FIELDS:assert s[key]==0,key
    assert s['recorded_checks_total']==428 and s['new_exact_algebra_certificate_groups']==8
    assert not s['general_Laguerre_TPFA_backend_selected'] and not s['new_numerical_gate_selected']
    for obj in [c,s]:
        assert not obj['finite_node_candidate_admitted'] and not obj['contact_transient_physically_admitted']
        assert obj['stage3_physical_acceptance']=='incomplete' and not obj['fluid_orc_enabled']
    text=THEORY.read_text()
    for marker in ['Phi','m_{min}','1/12','t_i','rho_hat','容量差','不是PDE误差下界']:
        assert marker in text,marker
    tree=ast.parse((ROOT/'research/reconnect/laguerre_mass_feasibility.py').read_text())
    imports={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    imports.update(a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names)
    assert imports <= {'fractions','pathlib','argparse','json','shutil','sys','audit_route_decision'}
    for name in BLOCKED:assert name not in sys.modules,name
    preserved=preservation();known={v['manifest'] for v in preserved['formal_scientific_manifests_verified']}
    prior=read(SOURCE/'preservation.json')['formal_scientific_manifests_verified']+[
        {'manifest':str((SOURCE/'sha256.json').relative_to(ROOT)),'manifest_sha256':digest(SOURCE/'sha256.json'),
            'entries_verified':len(parent)}]
    for item in prior:
        if item['manifest'] in known:continue
        path=ROOT/item['manifest'];assert digest(path)==item['manifest_sha256']
        entries=read(path);assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(ROOT/rel)==h,(item['manifest'],rel)
        preserved['formal_scientific_manifests_verified'].append(item);known.add(item['manifest'])
    assert len(known)==76
    save(out/'preservation.json',preserved)
    audit={'definition_receipt_and_parent_theory_scope_verified':True,
        'eight_new_algebra_groups_equal_standalone_python_minus_S_output':True,
        'mass_match_seed_exclusion_and_positive_distance_constraint_residual_verified':True,
        'existence_proof_not_misrepresented_as_seed_interiority_or_PDE_accuracy':True,
        'fixed_density_cut_cell_moments_and_unjustified_2D_flux_gaps_retained':True,
        'no_negative_distance_absolute_value_mass_or_density_normalization':True,
        'geometric_witness_not_counted_as_flow_case_or_runtime_partition_solver':True,
        **{k:0 for k in ZERO_FIELDS},'new_exact_algebra_certificate_groups':8,'recorded_checks_total':428,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':76,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'theory-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--standalone',required=True);args=p.parse_args()
    main(args.directory,args.standalone)
