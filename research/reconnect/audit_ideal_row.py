"""One ideal row: raw numerical audit and prior byte preservation; no graph."""
from pathlib import Path
import argparse,json
import hashlib
from fractions import Fraction
import numpy as np
from ideal_row_response import aggregate_row

_hash_cache={}
def digest(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if key not in _hash_cache:
        with path.open('rb') as f:_hash_cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
    return _hash_cache[key]


def main(output):
    root=Path(__file__).resolve().parents[2];out=Path(output).resolve();source=root/'runs/reconnect-mass-projection'
    old=json.loads((source/'preservation.json').read_text());originals=json.loads(Path('/workspace/original-handoff-sha256.json').read_text())
    for rel,h in originals.items():assert digest(root/rel)==h,rel
    attachments=json.loads(Path('/workspace/setup-ltn/attachment-manifest.json').read_text())
    for item in attachments:
        p=Path('/workspace')/item['path'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    formal=old['formal_scientific_manifests_verified']+[{'manifest':str((source/'sha256.json').relative_to(root)),
        'manifest_sha256':digest(source/'sha256.json'),'entries_verified':len(json.loads((source/'sha256.json').read_text()))}]
    for item in formal:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries_verified']
        for rel,h in entries.items():assert digest(root/rel)==h,(item['manifest'],rel)
    historical=old['historical_diagnostic_manifests_not_claimed_as_current_hash_matches']
    for item in historical:
        p=root/item['manifest'];assert digest(p)==item['manifest_sha256'];entries=json.loads(p.read_text());assert len(entries)==item['entries']
        differences=[{'path':rel,'saved':h,'current':digest(root/rel)} for rel,h in entries.items() if digest(root/rel)!=h]
        assert sorted(differences,key=lambda v:v['path'])==sorted(item['historical_paths_differing_from_later_formal_versions'],key=lambda v:v['path'])
    cfg=json.loads((out/'config.json').read_text());r=json.loads((out/'results.json').read_text())
    protocol=json.loads((source/'stage3-status.json').read_text())['next_single_row_response_predeclared']
    assert cfg['protocol']==protocol and cfg['protocol_source_sha256']==digest(source/'stage3-status.json')
    data=root/cfg['original_input_source'];assert digest(data)==cfg['original_input_sha256']
    with np.load(out/'ideal-row.npz') as z,np.load(data) as original:
        for key in ['mass','c0','ids','material_label_centers','rho0_point','reference_rate','kernel_rate']:
            np.testing.assert_array_equal(z[key],original[key])
        target=protocol['N_per_direction']//4-1;ell=1/protocol['N_per_direction'];h=protocol['h_m'];D=protocol['D_m2_per_s']
        center=float(z['material_label_centers'][target,0]);mass=float(z['mass'][target]);rho=float(z['rho0_point'][target])
        assert cfg['target_index']==target==31 and cfg['target_id']==int(z['ids'][target])
        assert cfg['target_center_a_m']==center==63/256 and cfg['saved_original_mass_kg']==mass==rho*ell**2
        assert cfg['unchanged_original_c0_mass_average']==float(z['c0'][target]) and rho==1. and min(center,.5-center)-ell/2>2*h
        for name,quad in zip(['coarse','fine'],protocol['fixed_quadrature_levels']):
            arrays,metrics=aggregate_row(ell,h,D,rho,mass,center,quad['normal_order'],quad['transverse_order'])
            for key,value in arrays.items():np.testing.assert_array_equal(z[name+'_'+key],value)
            assert metrics==r['rules'][name]
            assert not metrics['raw_kernel_or_response_renormalized'] and metrics['edge_weights_created']==0
        np.testing.assert_array_equal(z['panel_eta_coarse_minus_fine'],z['coarse_panel_eta']-z['fine_panel_eta'])
    bound_source=json.loads((source/'proof-certificates.json').read_text())['certificate_groups']['fixed_original_ratio_strict_excess_bounds']
    lower=float(Fraction(bound_source['eta_strict_lower']['exact']));upper=float(Fraction(bound_source['eta_strict_upper']['exact']))
    assert r['strict_analytic_eta_bounds']=={'lower':lower,'upper':upper}
    for name in ['coarse','fine']:assert lower<r['rules'][name]['eta']<upper
    for key,value in r['coarse_fine_absolute_differences'].items():assert abs(r['rules']['coarse'][key]-r['rules']['fine'][key])==value
    assert r['fine_eta_percent']==100*r['rules']['fine']['eta']
    assert r['fine_predicted_rate_minus_PDE_per_s']==r['rules']['fine']['effective_rate_exact_moment_prediction_per_s']-D
    assert r['best_raw_rule']=='fine' and r['aggregate_row_rule_evaluations']==2
    assert r['edge_weights_created']==r['new_candidate_graphs']==r['new_candidate_transient_runs']==r['new_reference_transient_runs']==0
    assert r['original_c0_and_point_candidate_and_PDE_unchanged'] and not r['fluid_orc_enabled'] and not r['finite_node_candidate_admitted']
    tests=json.loads((out/'tests.json').read_text());assert tests['exit_code']==0 and tests['checks_passed']==4 and tests['recorded_total']==356
    checks={'predeclared_one_row_two_rules_verified':True,'original_mass_c0_ids_saved_PDE_and_candidate_unchanged':True,
        'all_raw_quadrature_phase_records_and_panel_contributions_recomputed':True,
        'independent_direct_material_phase_check_and_original_mass_denominator_verified':True,
        'raw_kernel_moments_and_unmodified_exact_moment_prediction_separated':True,
        'all_response_metrics_rule_differences_and_strict_bounds_recomputed':True,
        'new_checks_passed':4,'recorded_checks_total':356,'new_candidate_graphs':0,'edge_weights_created':0,
        'new_candidate_transient_runs':0,'new_reference_transient_runs':0,'fluid_orc_enabled':False}
    (out/'numerical-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    preservation={'checked_original_files':len(originals),'checked_historical_attachments':len(attachments),
        'formal_scientific_manifests_verified':formal,'all_formal_scientific_entries_unchanged':True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches':historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit':True,'historical_note':old['historical_note'],
        'old_candidate_code_and_results_unchanged':True,'mutable_index_intentionally_excluded':'docs/RESEARCH_STATUS.md'}
    (out/'preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    print(json.dumps({'original_files_verified':len(originals),'historical_attachments_verified':len(attachments),
        'prior_formal_manifests_verified':len(formal),'unchanged_historical_diagnostic_manifests':len(historical),**checks},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
