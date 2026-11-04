"""Audit the new theory contract, standalone algebra record and preserved history."""
from pathlib import Path
import argparse
import ast
import json
import sys

from audit_route_decision import ROOT,read,save,digest,preservation
from orthogonal_mean_closure import SOURCE,THEORY,ZERO_FIELDS,BLOCKED,replay


def main(directory,standalone):
    out=Path(directory).resolve();assert not (out/'sha256.json').exists();replay(out)
    cfg=read(out/'config.json');parent=read(SOURCE/'sha256.json')
    assert cfg['parent_manifest_entries_verified']==len(parent)==1704
    assert cfg['parent_manifest_sha256']==digest(SOURCE/'sha256.json')
    assert (out/'theory-at-definition.md').read_bytes()==THEORY.read_bytes()
    receipt=read(out/'definition-receipt.json')
    assert receipt['formula_and_assumptions_saved_before_certificate_evaluation']
    assert receipt['theory_sha256_before_algebra']==digest(THEORY)
    assert receipt['contract_sha256_before_algebra']==digest(out/'closure-contract.json')
    certs=read(out/'algebra-certificates.json')
    assert read(standalone)==certs and certs['verified_exact_algebra_groups']==7
    assert len({v['name'] for v in certs['groups']})==7
    assert all(v['exact_algebra_passed'] for v in certs['groups'])
    c=read(out/'closure-contract.json');s=read(out/'stage3-status.json')
    assert c['state']=='fixed_true_material_cell_mass_average'
    assert c['scope']=='theory_only_static_periodic_isotropic_diffusion'
    assert c['true_mean_not_relabelled_surrogate_mean']
    assert c['coefficient_formula_does_not_use_scalar_state']
    assert set(c['allowed_inputs']).isdisjoint(c['forbidden_inputs'])
    assert 'no normalization' in c['mass_compatibility']
    assert c['observation_bound_requires_unobservable_true_geometry_assumptions']
    assert len(c['transient_budget'])==4
    assert not c['general_allowed_point_inputs_supported'] and not c['full_transient_error_estimate_proved']
    assert not c['candidate_weights_implemented_or_evaluated'] and not c['next_numerical_gate_selected']
    assert not c['new_production_backend_admitted']
    for key in ZERO_FIELDS:assert s[key]==0,key
    assert s['recorded_checks_total']==428 and s['new_exact_algebra_certificate_groups']==7
    assert not s['candidate_graph_implemented'] and not s['new_numerical_gate_selected']
    assert not s['finite_node_candidate_admitted'] and not s['contact_transient_physically_admitted']
    for obj in [c,s]:assert obj['stage3_physical_acceptance']=='incomplete' and not obj['fluid_orc_enabled']
    text=THEORY.read_text()
    for marker in ['epsilon_c','epsilon_2','协方差','W_1','(S_C^hA_C-A_CS_{hat})','逐单元','Laguerre']:
        assert marker in text,marker
    tree=ast.parse((ROOT/'research/reconnect/orthogonal_mean_closure.py').read_text())
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
    assert len(known)==75
    save(out/'preservation.json',preserved)
    audit={'new_formula_and_assumptions_saved_before_algebra_verified':True,
        'seven_algebra_groups_equal_standalone_python_minus_S_output':True,
        'algebra_is_not_claimed_as_automated_geometry_or_PDE_proof':True,
        'mass_compatibility_and_true_surrogate_mean_distinction_explicit':True,
        'centroid_and_covariance_defect_terms_retained':True,
        'general_point_input_obstruction_retained_no_normalization_or_admission':True,
        'four_transient_terms_and_unproved_evolution_gaps_explicit':True,
        'no_geometry_kernel_graph_solver_or_old_proof_program_imported':True,
        **{k:0 for k in ZERO_FIELDS},'new_exact_algebra_certificate_groups':7,'recorded_checks_total':428,
        'original_files_verified':782,'historical_attachments_verified':15,'prior_formal_manifests_verified':75,
        'finite_node_candidate_admitted':False,'stage3_physical_acceptance':'incomplete','fluid_orc_enabled':False}
    save(out/'theory-audit.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--standalone',required=True);args=p.parse_args()
    main(args.directory,args.standalone)
