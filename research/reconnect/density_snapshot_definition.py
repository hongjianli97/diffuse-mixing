"""Read-only input certification for one saved density snapshot; no producers."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gc, hashlib, math, resource, signal, time, traceback
import numpy as np
from audit_route_decision import ROOT, read, save, digest

PARENT = ROOT / 'runs/reconnect-joint-evidence'
OLD = ROOT / 'runs/reconnect-compression-density-calibration'
PROTOCOL = ROOT / 'research/reconnect/density_snapshot_definition_protocol.json'


def record(a):
    return {'shape': list(a.shape), 'dtype': str(a.dtype),
            'sha256_C_bytes': hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def inventory(case):
    N = case['N']
    with np.load(ROOT / case['trajectory'], allow_pickle=False) as z:
        ids, labels, rho0 = z['ids'], z['labels'], z['rho0_point']
    with np.load(ROOT / case['mass_file'], allow_pickle=False) as z:
        mass = z['mass']
        if N != 64:
            assert np.array_equal(z['ids'], ids)
            assert np.array_equal(z['material_label_centers'], labels)
    assert np.array_equal(ids, np.arange(N*N))
    assert np.array_equal(labels[:, 0], (ids % N + .5) / N)
    assert np.array_equal(labels[:, 1], (ids // N + .5) / N)
    assert mass.shape == rho0.shape == (N*N,)
    assert np.isfinite(mass).all() and min(mass) > 0
    assert np.isfinite(rho0).all() and min(rho0) > 0
    with np.load(ROOT / case['builder'], allow_pickle=False) as z:
        assert np.array_equal(z['ids'], ids)
        times, positions, oracle = z['k32_time'], z['k32_positions'], z['k32_rho']
        assert np.array_equal(z['k32_midpoint_time'], times[1::2])
        assert np.array_equal(z['k32_boundary_time'], times[::2])
    assert np.array_equal(times, np.linspace(0, .05, 65))
    assert float(times[63]).hex() == '0x1.9333333333334p-5'
    assert positions.shape == (65, N*N, 2) and oracle.shape == (65, N*N)
    assert np.isfinite(positions).all() and np.all((positions >= 0) & (positions < 1))
    assert np.isfinite(oracle).all() and np.min(oracle) > 0
    assert np.array_equal(positions[0], labels) and np.array_equal(oracle[0], rho0)
    # Read-only jump counts, not new continuous trajectories or saved winding fields.
    positive = np.zeros(2, dtype=np.int64)
    negative = positive.copy()
    margin = 1.; largest = 0.
    old_w = None
    if N == 64:
        with np.load(OLD / 'integer-winding.npz', allow_pickle=False) as z: old_w = z['k32']
        assert not old_w[0].any()
    for k in range(1, 65):
        d = positions[k] - positions[k-1]
        margin = min(margin, float(np.min(abs(abs(d) - .5))))
        # The margin guard makes float subtraction/branching exact for these inputs.
        assert margin > 64*np.finfo(float).eps
        jump = np.floor(d + .5).astype(np.int64)
        largest = max(largest, float(np.max(abs(d - jump))))
        positive += np.sum(jump == -1, axis=0)
        negative += np.sum(jump == 1, axis=0)
        if old_w is not None: assert np.array_equal(old_w[k]-old_w[k-1], -jump)
    assert largest < .5 and max(np.diff(times)) < .5
    idx = read(ROOT / case['graph_index'])['k32-step00']
    graph_path = ROOT / case['geometry']
    assert idx['file'] == 'k32/geometry-step00.json' and idx['sha256'] == digest(graph_path)
    g = read(graph_path)
    assert g['ids'] == ids.tolist()
    edges = set(map(tuple, g['edges']))
    expected = set()
    for i in range(N*N):
        for j in (i//N*N + (i+1) % N, (i+N) % (N*N)):
            expected.add(tuple(sorted((i, j))))
    assert len(g['edges']) == len(edges) == 2*N*N and edges == expected
    out = {'N': N, 'nodes': N*N, 'arrays': {k: record(v) for k, v in
        [('ids', ids), ('labels', labels), ('rho0', rho0), ('mass', mass),
         ('time', times), ('snapshot_positions', positions[63]), ('snapshot_oracle', oracle[63])]},
        'saved_frame': 63, 'saved_step': 31, 'snapshot_time_hex': float(times[63]).hex(),
        'nominal_time_exact': '63/1280',
        'saved_minus_nominal_exact': str(F(float(times[63]))-F(63,1280)),
        'mass_sum_fsum': math.fsum(mass), 'mass_minimum': float(min(mass)),
        'mass_maximum': float(max(mass)), 'fixed_four_neighbor_edges': len(edges),
        'initial_ring_area_exact': str(F(2,N*N)),
        'maximum_short_saved_displacement': largest, 'half_period_margin': margin,
        'positive_wrap_events_axes': positive.tolist(), 'negative_wrap_events_axes': negative.tolist(),
        'N64_saved_winding_identity': True if N == 64 else None,
        'kernel_h_exact': str(F(2,N)), 'initial_labels_and_densities_identical': True,
        'c0_concentration_reference_or_states_loaded': False}
    del g, edges, expected, positions, oracle
    gc.collect()
    return out


def preserve():
    old = read(PARENT/'preservation.json'); p = PARENT/'sha256.json'
    formal = old['formal_manifests_verified'] + [{'manifest': str(p.relative_to(ROOT)),
        'manifest_sha256': digest(p), 'entries_verified': len(read(p))}]
    assert len(formal) == 123
    for item in formal:
        p = ROOT/item['manifest']; assert digest(p) == item['manifest_sha256']
        entries = read(p); assert len(entries) == item['entries_verified']
        for name, h in entries.items(): assert digest(ROOT/name) == h, name
    originals = read('/workspace/original-handoff-sha256.json')
    attachments = read('/workspace/setup-ltn/attachment-manifest.json')
    assert len(originals) == 782 and len(attachments) == 15
    for name, h in originals.items(): assert digest(ROOT/name) == h, name
    for item in attachments:
        p = Path('/workspace')/item['path']
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256']
    for item in old['historical_diagnostic_differences_retained']:
        p = ROOT/item['manifest']; assert digest(p) == item['manifest_sha256']
        diff = [{'path': k, 'saved': v, 'current': digest(ROOT/k)}
                for k, v in read(p).items() if digest(ROOT/k) != v]
        assert sorted(diff, key=lambda x: x['path']) == sorted(
            item['historical_paths_differing_from_later_formal_versions'], key=lambda x: x['path'])
    return {'formal_manifests_verified': formal, 'originals_verified': 782, 'attachments_verified': 15,
        'historical_diagnostic_differences_retained': old['historical_diagnostic_differences_retained'],
        'all_prior_scientific_files_unchanged': True}


def main(directory):
    out = Path(directory).resolve(); out.mkdir(exist_ok=False); start = time.monotonic()
    def alarm(*_): raise TimeoutError('read-only definition120second cap')
    signal.signal(signal.SIGALRM, alarm); signal.alarm(120)
    try:
        p = read(PROTOCOL); gate = read(PARENT/'next-gate-spec.json')
        assert gate['type'] == 'read_only_same_flow_density_snapshot_protocol_and_source_audit'
        assert gate['new_physical_fields_or_solves_authorized'] == 0
        assert digest(PARENT/'next-gate-spec.json') == p['source_gate_sha256']
        for name, h in p['sources'].items(): assert digest(ROOT/name) == h, name
        save(out/'protocol.json', p); save(out/'source-gate-spec.json', gate)
        save(out/'implementation-lock.json', {'files': {name: digest(ROOT/name)
            for name in p['implementation_files']}, 'locked_before_diagnostics': True})
        rows = [inventory(case) for case in p['cases']]
        assert len({r['arrays']['time']['sha256_C_bytes'] for r in rows}) == 1
        reused = {}
        for s in ('kernel', 'ledger'):
            f = OLD/s/'k32/density-step31.npz'
            with np.load(f, allow_pickle=False) as z: a = z['rho']
            assert a.shape == (4096,) and np.isfinite(a).all() and min(a) > 0
            reused[s] = {'file': str(f.relative_to(ROOT)), 'sha256': digest(f), 'array': record(a),
                'raw_minimum': float(min(a)), 'raw_maximum': float(max(a)), 'rerun': False}
        t1 = read(ROOT/'runs/reconnect-t1-density-calibration/results.json')['density_and_symbol_records']['kernel']
        assert t1['high_precision_80_100_rounded_equal']
        save(out/'input-inventory.json', {'cases': rows, 'N64_reused_fields': reused,
            'all_clocks_identical': True, 'new_density_fields_generated': 0,
            'continuous_lift_or_density_arrays_saved': False})
        save(out/'results.json', {'source_definition_ready': True,
            'future_experiment': p['future_experiment'],
            'uniform_fixed_h_over_spacing_warning': {
                'h_over_Delta': 2, 'raw_density_saved_T1_N32': t1['raw_density_minimum'],
                'saved_T1_point_bias': t1['raw_density_vs_true_one_Linf'],
                'lattice_sum_independent_of_N_in_real_arithmetic': True,
                'same_bias_asserted_for_moving_nonuniform_case': False},
            'new_physical_fields_or_solves': 0, 'new_software_tests': 0,
            'recorded_checks_total': 508, 'old_decisions_unchanged': True,
            'stage3': 'incomplete', 'fluid_orc_enabled': False})
        save(out/'preservation.json', preserve())
        for name, h in p['sources'].items(): assert digest(ROOT/name) == h, name
        elapsed = time.monotonic()-start; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        assert elapsed < 120 and rss < 1024
        save(out/'execution.json', {'completed': True, 'elapsed_seconds': elapsed, 'peak_RSS_MiB': rss,
            'new_physical_fields_or_solves': 0, 'new_software_tests': 0, 'old_tests_rerun': 0,
            'published_or_pushed': False})
        print('Three saved inputs certified; four-field experiment specified but not run.')
    except BaseException as exc:
        save(out/'failed-execution.json', {'type': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc()}); raise
    finally: signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    main(parser.parse_args().directory)
