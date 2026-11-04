"""Read-only scientific provenance and route-contract audit; no scientific reruns.

Uses the standard library only. Numerical checks and proof counts remain unchanged.
The full historical workspace is required; this is not a standalone ZIP auditor.
"""
from pathlib import Path
import argparse
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/reconnect-cross-scalar'
CURRENT = ROOT / 'runs/reconnect-route-decision'
PROTOCOL = 'next_sealed_evidence_route_decision_predeclared'
ZERO_FIELDS = [
    'new_numerical_or_flow_tests', 'new_exact_algebra_certificate_groups',
    'new_candidate_graphs', 'new_edge_weights', 'new_operator_quadrature_evaluations',
    'new_candidate_transient_runs', 'new_reference_transient_runs', 'LP_solves',
]
EXTRA_SOURCES = {
    'research/reconnect/MASS_AVERAGE_PROJECTION_WEAK_BUDGET.md': 'ideal material-average projection scope and sufficient scale condition',
    'research/reconnect/ORACLE_INTERFACE_SECOND_ORDER.md': 'known curvature obstruction even with full density',
    'research/reconnect/ORACLE_WEAK_INTERFACE_CONSISTENCY.md': 'conditional continuous weak result; not finite-node admission',
    'research/reconnect/ORACLE_PATH_RESISTANCE_CONTROL.md': 'extra-information oracle; frozen-state improvement only',
    'research/reconnect/IDEAL_MATERIAL_ROW_RESPONSE.md': 'fixed-ratio material-average closure defect',
    'research/reconnect/SAVED_POINT_ROW_COMPARISON.md': 'signed error cancellation; no causal allocation',
    'research/reconnect/SAVED_INTERFACE_PAIR_PROBE.md': 'original two material means and their local responses',
    'research/rebuild/CONTACT_TRANSMISSION.md': 'ordinary one-dimensional FV baseline and experimental alternative',
    'runs/reconnect-cross-scalar/proof-certificates.json': 'read-only shared scalar incompatibility certificate',
    'runs/rebuild-stage3-mass-cone/results.json': 'fixed-stencil moment-system scope',
    'runs/rebuild-stage3-shared-exchange/results.json': 'fixed shared-system scope, not a universal symmetry no-go',
    'runs/rebuild-stage3-density-geometry/results.json': 'finite-input geometry witness and its kinematic scope',
    'runs/rebuild-stage3-hybrid-face/results.json': 'negative raw offdiagonal of this particular hybrid baseline',
    'runs/reconnect-mainline/results.json': 'graph algebra versus backend admission',
    'runs/reconnect-interface-pair/results.json': 'saved two-row control, no new graph',
}
_cache = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _cache:
        with path.open('rb') as f:
            _cache[key] = hashlib.file_digest(f, 'sha256').hexdigest()
    return _cache[key]


def save(path, obj):
    with Path(path).open('x') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def preservation():
    old = read(SOURCE / 'preservation.json')
    originals = read('/workspace/original-handoff-sha256.json')
    attachments = read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals) == 782 and len(attachments) == 15
    for rel, expected in originals.items():
        assert digest(ROOT / rel) == expected, rel
    for item in attachments:
        p = Path('/workspace') / item['path']
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], str(p)
    formal = old['formal_scientific_manifests_verified'] + [{
        'manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'manifest_sha256': digest(SOURCE / 'sha256.json'),
        'entries_verified': len(read(SOURCE / 'sha256.json')),
    }]
    assert len(formal) == 53
    for item in formal:
        p = ROOT / item['manifest']
        assert digest(p) == item['manifest_sha256'], item['manifest']
        entries = read(p)
        assert len(entries) == item['entries_verified']
        for rel, expected in entries.items():
            assert digest(ROOT / rel) == expected, (item['manifest'], rel)
    historical = old['historical_diagnostic_manifests_not_claimed_as_current_hash_matches']
    assert len(historical) == 4
    for item in historical:
        p = ROOT / item['manifest']
        assert digest(p) == item['manifest_sha256']
        entries = read(p)
        assert len(entries) == item['entries']
        differences = [{'path': rel, 'saved': h, 'current': digest(ROOT / rel)}
                       for rel, h in entries.items() if digest(ROOT / rel) != h]
        assert sorted(differences, key=lambda x: x['path']) == sorted(
            item['historical_paths_differing_from_later_formal_versions'], key=lambda x: x['path'])
    return {
        'checked_original_files': len(originals),
        'checked_historical_attachments': len(attachments),
        'formal_scientific_manifests_verified': formal,
        'all_formal_scientific_entries_unchanged': True,
        'historical_diagnostic_manifests_not_claimed_as_current_hash_matches': historical,
        'historical_nonformal_path_differences_unchanged_from_source_audit': True,
        'historical_note': old['historical_note'],
        'old_candidate_code_and_results_unchanged': True,
        'mutable_index_intentionally_excluded': 'docs/RESEARCH_STATUS.md',
    }


def main(output):
    out = Path(output).resolve()
    assert not (out / 'sha256.json').exists(), 'never rewrite sealed evidence'
    out.mkdir(parents=True, exist_ok=True)
    for name in ['evidence-index.json', 'review-audit.json', 'preservation.json']:
        assert not (out / name).exists(), 'use a fresh audit output'
    source_status = read(SOURCE / 'stage3-status.json')
    protocol = source_status[PROTOCOL]
    config = read(CURRENT / 'config.json')
    decision = read(CURRENT / 'route-decision.json')
    status = read(CURRENT / 'stage3-status.json')
    assert config['protocol'] == protocol
    assert config['protocol_source_sha256'] == digest(SOURCE / 'stage3-status.json')
    assert decision['state'] == 'fixed_material_cell_mass_average'
    assert decision['route_dispositions']['common_cross_phase_scalar'] == 'closed_restricted_family'
    assert decision['route_dispositions']['all_positive_exchange'] == 'not_ruled_out'
    assert decision['route_dispositions']['ordinary_central_resistance_FV'] == 'retain_limited_baseline'
    assert decision['selected_next_work'] == 'conditional_density_reconstruction_continuous_weak_perturbation_bound'
    assert decision['density_reconstruction'] == 'periodic_Euclidean_nearest_point_density_fixed_ID_ties'
    assert not decision['new_production_backend_admitted']
    assert not decision['reconstruction_partition_is_material_geometry']
    assert decision['mass_and_density_renormalization_forbidden']
    forbidden = {'true_material_faces', 'F_C', 'velocity_gradients', 'reference_scalar_or_PDE_rate'}
    assert forbidden <= set(decision['forbidden_candidate_inputs'])
    assert not forbidden.intersection(decision['allowed_candidate_inputs'])
    for obj in [decision, status]:
        assert obj['stage3_physical_acceptance'] == 'incomplete' and not obj['fluid_orc_enabled']
        assert obj['recorded_checks_total'] == 363
        assert all(obj[key] == 0 for key in ZERO_FIELDS)
    next_plan = status['next_density_closure_weak_perturbation_predeclared']
    assert next_plan['theory_only'] and not next_plan['finite_node_convergence_claimed']
    assert not next_plan['new_graphs_or_numerical_cases']
    assert next_plan['proof_completed'] is False

    # Read saved conclusions only: do not import numerical or certificate modules.
    scalar = read(SOURCE / 'proof-certificates.json')
    assert scalar['verified_exact_algebra_groups'] == 7 and not scalar['finite_node_candidate_admitted']
    assert scalar['recorded_checks_total'] == 363
    mass = read(ROOT / 'runs/rebuild-stage3-mass-cone/results.json')
    shared = read(ROOT / 'runs/rebuild-stage3-shared-exchange/results.json')
    assert not mass['physical_acceptance'] and not shared['accepted_global_operator_constructed']
    assert not shared['shared_global_conductance_witness_found']
    hybrid = read(ROOT / 'runs/rebuild-stage3-hybrid-face/results.json')
    assert hybrid['minimum_raw_offdiagonal_rate_per_s'] < 0 and not hybrid['physical_acceptance']
    mainline = read(ROOT / 'runs/reconnect-mainline/results.json')['backend_admission']
    assert not mainline['one_dimensional_ordinary_FV']['general_point_trajectory_backend']
    assert not mainline['saved_optimized_graph']['fluid_admitted']

    preserved = preservation()
    entries = read(SOURCE / 'sha256.json')
    roles = {rel: 'required by predeclared route-decision source protocol'
             for rel in protocol['source_reports']}
    roles.update(EXTRA_SOURCES)
    evidence = []
    for rel, role in roles.items():
        assert rel in entries, ('source not sealed', rel)
        assert digest(ROOT / rel) == entries[rel], rel
        evidence.append({'path': rel, 'sha256': entries[rel], 'role': role,
                         'source_read_only': True, 'scientific_recomputation': False})
    save(out / 'evidence-index.json', {
        'source_manifest': str((SOURCE / 'sha256.json').relative_to(ROOT)),
        'source_manifest_sha256': digest(SOURCE / 'sha256.json'),
        'source_manifest_entries': len(entries), 'sources': evidence,
    })
    review = {
        'predeclared_scope_and_source_hash_verified': True,
        'all_indexed_sources_previously_sealed': True,
        'source_reports_indexed': sum(item['path'].endswith('.md') for item in evidence),
        'source_JSON_records_indexed': sum(item['path'].endswith('.json') for item in evidence),
        'obstruction_scopes_kept_distinct': True,
        'conditional_input_closure_declared_before_new_candidate': True,
        'true_geometry_not_inferred_from_density_interpolation': True,
        'density_perturbation_proof_still_pending': True,
        'audit_is_provenance_and_metadata_consistency_not_new_scientific_validation': True,
        **{key: 0 for key in ZERO_FIELDS}, 'recorded_checks_total': 363,
        'old_v080_or_246_tests_recovered': False,
        'new_production_backend_admitted': False, 'fluid_orc_enabled': False,
    }
    save(out / 'preservation.json', preserved)
    save(out / 'review-audit.json', review)
    print(json.dumps({'original_files_verified': 782, 'historical_attachments_verified': 15,
                      'prior_formal_manifests_verified': 53,
                      'unchanged_historical_diagnostic_manifests': 4, **review}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
