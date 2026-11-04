"""Seal one nonproduct smooth-shear frozen FV comparison and replay its archive."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from audit_route_decision import ROOT, read, save, digest


def main():
    source = ROOT / 'runs/reconnect-smooth-auxiliary-FV-frozen'
    out = ROOT / 'runs/reconnect-shear-voronoi-FV-frozen'
    archive = ROOT / 'exports/ltn-shear-voronoi-FV-frozen.zip'
    delivery = ROOT / 'exports/ltn-shear-voronoi-FV-frozen-delivery.json'
    downloads = ROOT / 'exports/ltn-shear-voronoi-FV-frozen-downloads.md'
    seal = out / 'sha256.json'
    assert not any(p.exists() for p in (archive, delivery, downloads, seal)), 'never overwrite sealed evidence'
    execution = read(out / 'execution.json')
    audit = read(out / 'frozen-audit.json')
    audit_run = read(out / 'audit-execution.json')
    status = read(out / 'stage3-status.json')
    result = read(out / 'results.json')
    gate = read(out / 'next-gate-spec.json')
    assert execution['tests_exit_code'] == execution['study_exit_code'] == audit_run['audit_exit_code'] == 0
    assert execution['first_attempts_passed'] and audit_run['audit_first_attempt_passed']
    assert execution['resource_caps_passed']
    assert audit['prior_formal_manifests_verified'] == 79
    assert audit['original_files_verified'] == 782 and audit['historical_attachments_verified'] == 15
    assert status['recorded_checks_total'] == 439 and status['previous_checks_rerun'] == 0
    assert result['comparison_decision'] == 'FV_better_than_saved_kernel_resolvably'
    assert not gate['already_executed'] and gate['expected_new_midpoint_graphs'] == 12
    assert gate['ordered_interval_counts'] == [4, 8] and gate['shock_extension_deferred']
    assert not audit_run['future_graphs_or_propagation_executed']
    assert not status['general_smooth_flow_admitted'] and not status['fluid_orc_enabled']
    assert status['stage3_physical_acceptance'] == 'incomplete'
    for packet_key in ('builder_input_packet', 'comparison_packet'):
        assert digest(ROOT / gate[packet_key]) == gate[packet_key + '_sha256']

    prior = read(source / 'sha256.json')
    assert len(prior) == 1803
    for rel, h in prior.items():
        assert digest(ROOT / rel) == h, rel
    files = ['research/reconnect/' + name for name in (
        'periodic_voronoi_protocol.json', 'PERIODIC_VORONOI_CONSTRUCTION.md',
        'periodic_voronoi_fv.py', 'test_periodic_voronoi_fv.py',
        'shear_voronoi_fv_study.py', 'audit_shear_voronoi_fv.py',
        'package_shear_voronoi_fv.py', 'SHEAR_VORONOI_FV_FROZEN.md')]
    source_seal = str((source / 'sha256.json').relative_to(ROOT))
    entries = dict(prior)
    entries[source_seal] = digest(source / 'sha256.json')
    for rel in files:
        assert rel not in entries
        entries[rel] = digest(ROOT / rel)
    for p in out.iterdir():
        assert p.is_file(), p
        rel = str(p.relative_to(ROOT))
        assert rel not in entries
        entries[rel] = digest(p)
    save(seal, dict(sorted(entries.items())))

    selected = set(files)
    selected.update(str(p.relative_to(ROOT)) for p in out.iterdir())
    selected.update((source_seal, 'research/reconnect/audit_route_decision.py',
                     'research/reconnect/auxiliary_tensor_fv.py', 'docs/RESEARCH_STATUS.md'))
    # The matched future packets carry all needed old time inputs and references.
    # Their large legacy source containers remain sealed in the original workspace.
    omitted_legacy_npz = {
        'runs/reconnect-ordered-shear/ordered-shear.npz',
        'runs/reconnect-shear-reference/shear-reference-observations.npz',
    }
    selected.update(item['path'] for item in read(out / 'evidence-index.json')['sources']
                    if item['path'] not in omitted_legacy_npz)
    manifest = {rel: digest(ROOT / rel) for rel in sorted(selected)}
    note = (
        'LTN SMOOTH NONSHOCK FIRST: one existing nonproduct N128 periodic shear frozen state.\n'
        'FAVORABLE: FV relative rate error3.9908482777912727e-4; saved kernel8.695743158237219e-3; ratio21.789210095077483.\n'
        'Normalization is the same complete saved shear PDE-rate mass norm0.1652606553362351, not an initial-fluctuation or endpoint norm.\n'
        'Original actual material means/masses, IDs, positions/rho, D.02, t.05 and periodic domain unchanged.\n'
        'Exact Fraction periodic Voronoi clipping with complete radius/bin neighbor search; no jitter, fixed-k truncation or topology repair.\n'
        '48896 shared positive conductances use ordinary central face resistance. All coefficients sealed before c0/reference access.\n'
        'Builder receives no mass/scalar/reference/true material faces/F_C/velocity gradient; auxiliary cells do not replace actual material cells.\n'
        'Independent geometry reconstruction from bisectors, all shared-face resistances and exact node-action digest verified.\n'
        'Six fixed probe IDs check every other input point halfplane. Global area/energy sums have rigorous binary enclosures.\n'
        'Exact versus independent float action mass-norm difference2.624088524174136e-16.\n'
        'Error gain0.008296658330458092 exceeds reference/arithmetic indicator1.573358805650696e-9, not a certified total PDE bound.\n'
        'Original float mass-rate residual+1.160439529227161e-19 retained; no mass/rho/area/state renormalization.\n'
        'No scalar propagation or I/H prediction this round; favorable frozen response is not general smooth-flow or Stage3 admission.\n'
        'Six new checks passed first attempt; recorded439, previous433 not rerun. Main and independent audit first attempts passed.\n'
        'Main36.9732s/335.9844MiB within600s/512MiB. Archive replay repeats this same case, not a second flow experiment.\n'
        'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python research/reconnect/shear_voronoi_fv_study.py --replay runs/reconnect-shear-voronoi-FV-frozen\n'
        'Replay needs Python3 and NumPy (recorded3.12.14/2.3.5); rebuilds exact geometry, coefficients, action and saved metrics.\n'
        'Next gate UNEXECUTED: same smooth shear real finite-time cbar/I/H, fixed4/8 intervals,12 midpoint graphs, original m and independent saved PDE reference.\n'
        'Matched future builder and comparison packets included; provenance records byte matches to original legacy containers.\n'
        'Legacy full ordered-shear and time-reference NPZ containers omitted to avoid duplication, not modified; retained in workspace cumulative seal.\n'
        'No future coefficients, propagation, new reference solves or new physical-flow definitions generated.\n'
        'Shock extension deferred per user; complete Stage3 conditions unchanged, Stage3 incomplete and fluid ORC disabled.\n'
        '782 originals,15 attachments,79 prior formal scientific manifests preserved; four known historical diagnostic differences retained.\n'
        'review-manifest.json covers included files. Cumulative scientific seal also references omitted history; full preservation audit needs original workspace.\n'
        'Mutable docs/RESEARCH_STATUS.md is a review snapshot outside the cumulative scientific seal.\n'
        'No PDF, release, commit/push, environment change or publication. Old v0.8.0/246 tests not recovered.\n'
    )
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in manifest:
            z.write(ROOT / rel, rel)
        z.writestr('review-manifest.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('DELIVERY_README.txt', note)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(manifest) | {'review-manifest.json', 'DELIVERY_README.txt'}
        for rel, h in manifest.items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == h, rel
        with tempfile.TemporaryDirectory(prefix='ltn-shear-voronoi-FV-', dir='/tmp') as directory:
            z.extractall(directory)
            completed = subprocess.run([
                sys.executable, str(Path(directory) / 'research/reconnect/shear_voronoi_fv_study.py'),
                '--replay', 'runs/reconnect-shear-voronoi-FV-frozen'], cwd=directory,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4',
                         PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=600)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            for rel, h in manifest.items():
                assert digest(Path(directory) / rel) == h, rel
    for rel, h in read(seal).items():
        assert digest(ROOT / rel) == h, rel
    assert digest(ROOT / 'docs/RESEARCH_STATUS.md') == manifest['docs/RESEARCH_STATUS.md']
    info = {
        'archive': {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)},
        'archive_CRC_and_all_included_SHA256_verified': True,
        'fresh_action_replay_exit_code': completed.returncode,
        'fresh_action_replay_log': completed.stdout,
        'fresh_replay_files_unchanged': True,
        'scientific_manifest_entries': len(entries), 'archive_entries': len(manifest),
        'source_scientific_manifest': source_seal,
        'FV_relative_rate_error': result['FV_relative_rate_error'],
        'saved_kernel_relative_rate_error': result['saved_kernel_relative_rate_error_recomputed'],
        'saved_kernel_over_FV_error_ratio': result['saved_kernel_over_FV_error_ratio'],
        'comparison_decision': result['comparison_decision'],
        'new_software_checks': 6, 'recorded_checks_total': 439, 'previous_checks_rerun': 0,
        'software_main_and_audit_first_attempts_passed': True,
        'new_actual_frozen_state_cases': 1, 'new_candidate_graphs': 1,
        'new_shared_conductances': 48896, 'independent_float_action_controls': 1,
        'new_scalar_transient_runs': 0, 'new_PDE_reference_solves': 0,
        'new_physical_flow_definitions': 0, 'next_smooth_shear_time_gate_executed': False,
        'shock_extension_deferred': True,
        'original_files_preserved': 782, 'historical_attachments_preserved': 15,
        'prior_formal_manifests_preserved': 79,
        'general_smooth_flow_admitted': False, 'finite_node_candidate_admitted': False,
        'stage3_physical_acceptance': 'incomplete', 'fluid_orc_enabled': False, 'published': False,
    }
    save(delivery, info)
    with downloads.open('x') as f:
        f.write('# 光滑二维剪切Voronoi辅助FV冻结对照\n\n'
                '[报告、原输入/参考、精确几何/电导、控制与可重放代码](ltn-shear-voronoi-FV-frozen.zip)。\n\n'
                f'SHA-256：`{info["archive"]["sha256"]}`。\n\n'
                '[范围与校验](ltn-shear-voronoi-FV-frozen-delivery.json)。全新解压同一几何与作用重放通过。\n\n'
                '本例FV变化率误差3.99085e-4，保存核8.69574e-3，误差比21.8；仅一个非张量冻结态。\n\n'
                '6项新检查首次通过，累计439。下一项同一光滑剪切实际时间对照未执行；含激波后置。Stage3 incomplete，fluid ORC关闭。\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
