"""Independent checks of saved inputs and the bounded future contract, no solves."""
from pathlib import Path
from fractions import Fraction as F
import argparse, ast, hashlib, math, resource, signal, time
import numpy as np
from audit_route_decision import ROOT, read, save, digest


def audit(out):
    p = read(out/'protocol.json'); inventory = read(out/'input-inventory.json')
    result = read(out/'results.json'); implementation = read(out/'implementation-lock.json')
    for entries in (p['sources'], implementation['files']):
        for name, h in entries.items(): assert digest(ROOT/name) == h, name
    assert implementation['locked_before_diagnostics']
    # This read-only phase cannot call density/geometry producers or numerical solvers.
    for fn in ('density_snapshot_definition.py', 'audit_density_snapshot_definition.py'):
        tree = ast.parse((ROOT/'research/reconnect'/fn).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module in ('pathlib', 'fractions', 'audit_route_decision')
            if isinstance(node, ast.Import):
                assert {x.name for x in node.names} <= {'argparse', 'ast', 'gc', 'hashlib',
                    'math', 'resource', 'signal', 'time', 'traceback', 'numpy'}
    reports = []; masses = {}; previous_clock = None
    for c, row in zip(p['cases'], inventory['cases']):
        n = c['N']; assert n == row['N']
        with np.load(ROOT/c['trajectory'], allow_pickle=False) as z:
            ids, labels, r0 = z['ids'], z['labels'], z['rho0_point']
        with np.load(ROOT/c['mass_file'], allow_pickle=False) as z: mass = z['mass']
        masses[n] = mass
        with np.load(ROOT/c['builder'], allow_pickle=False) as z:
            t, x, r = z['k32_time'], z['k32_positions'], z['k32_rho']
            assert np.array_equal(z['ids'], ids)
            assert np.array_equal(z['k32_midpoint_time'], t[1::2])
            assert np.array_equal(z['k32_boundary_time'], t[::2])
        assert (t.shape, x.shape, r.shape) == ((65,), (65, n*n, 2), (65, n*n))
        assert np.array_equal(ids, np.arange(n*n)) and np.array_equal(x[0], labels)
        assert np.array_equal(r[0], r0) and np.min(r) > 0 and np.min(mass) > 0
        assert np.isfinite(x).all() and np.isfinite(r).all() and np.isfinite(mass).all()
        assert np.all((x >= 0) & (x < 1))
        assert np.array_equal(t, np.linspace(0., .05, 65))
        assert previous_clock is None or np.array_equal(t, previous_clock)
        previous_clock = t
        assert F(float(t[63])) - F(63,1280) == F(1,180143985094819840)
        assert t[63].hex() == row['snapshot_time_hex'] == '0x1.9333333333334p-5'
        assert row['saved_minus_nominal_exact'] == '1/180143985094819840'
        for name, a in [('ids', ids), ('labels', labels), ('rho0', r0), ('mass', mass),
                        ('time', t), ('snapshot_positions', x[63]), ('snapshot_oracle', r[63])]:
            assert row['arrays'][name] == {'shape': list(a.shape), 'dtype': str(a.dtype),
                'sha256_C_bytes': hashlib.sha256(a.tobytes()).hexdigest()}
        assert np.array_equal(labels.reshape(n,n,2)[0,:,0], (np.arange(n)+.5)/n)
        assert np.array_equal(labels.reshape(n,n,2)[:,0,1], (np.arange(n)+.5)/n)
        assert np.all(labels.reshape(n,n,2)[:,:,0] == labels[:n,0])
        assert np.all(labels.reshape(n,n,2)[:,:,1] == labels[::n,1,None])
        positive = [0,0]; negative = [0,0]; largest = 0.; margin = 1.
        for k in range(1,65):
            d = x[k]-x[k-1]
            margin = min(margin, float(np.min(abs(abs(d)-.5))))
            # Independent sign branches; no floor recurrence or saved future lift.
            for axis in (0,1):
                a = d[:,axis]; positive[axis] += int(np.count_nonzero(a < -.5))
                negative[axis] += int(np.count_nonzero(a > .5))
                short = np.where(a > .5, a-1, np.where(a < -.5, a+1, a))
                largest = max(largest, float(np.max(abs(short))))
                # Exact rational certificate for the closest value to the branch boundary.
                i = int(np.argmin(abs(abs(a)-.5)))
                exact = F(float(x[k,i,axis]))-F(float(x[k-1,i,axis]))
                assert abs(abs(exact)-F(1,2)) > F(1,4)
        assert margin > .25 and largest < .001
        assert (positive,negative) == (row['positive_wrap_events_axes'],row['negative_wrap_events_axes'])
        assert largest == row['maximum_short_saved_displacement'] and margin == row['half_period_margin']
        graph = read(ROOT/c['geometry']); degree = np.zeros(n*n,dtype=int)
        assert graph['ids'] == ids.tolist() and len(graph['edges']) == 2*n*n
        seen = set()
        for i,j in graph['edges']:
            assert 0 <= i < j < n*n and (i,j) not in seen; seen.add((i,j))
            dx = (j%n-i%n)%n; dy = (j//n-i//n)%n
            assert (dx in (1,n-1) and dy == 0) or (dy in (1,n-1) and dx == 0)
            degree[i] += 1; degree[j] += 1
        assert np.all(degree == 4) and row['fixed_four_neighbor_edges'] == len(seen)
        assert F(row['initial_ring_area_exact']) == F(2,n*n)
        assert F(row['kernel_h_exact']) == F(2,n)
        assert row['mass_sum_fsum'] == math.fsum(mass)
        reports.append({'N':n, 'all_IDs_frames_and_edges_verified':True,
            'maximum_short_displacement':largest, 'branch_margin':margin,
            'original_mass_array_sha256':row['arrays']['mass']['sha256_C_bytes']})
        del graph, seen, x, r
    # Parent/child mass sources and saved aggregation retain their raw roundoff.
    fine = ROOT/p['cases'][2]['mass_file']
    with np.load(fine,allow_pickle=False) as z:
        assert np.array_equal(z['parent_mass'],masses[128])
        with np.load(ROOT/p['cases'][1]['trajectory'],allow_pickle=False) as a:
            assert np.array_equal(z['parent_ids'],a['ids'])
            assert np.array_equal(z['parent_material_label_centers'],a['labels'])
    with np.load(ROOT/p['cases'][0]['mass_file'],allow_pickle=False) as z:
        assert np.array_equal(z['mass'],z['n128_mass_sum'])
    # Legacy N64 anchor images/areas are identity evidence, not recomputed fields.
    old = ROOT/'runs/reconnect-compression-density-calibration'
    anchors = read(old/'anchor-rings.json')
    with np.load(ROOT/p['cases'][0]['trajectory'],allow_pickle=False) as z: labels = z['labels']
    for i, ring in enumerate(anchors['rings']):
        v = [(F(float(labels[j,0]))+sx-F(float(labels[i,0])),
              F(float(labels[j,1]))+sy-F(float(labels[i,1]))) for j,sx,sy in ring]
        assert len(set(v)) == 4 and all(abs(a)+abs(b)==F(1,64) for a,b in v)
        area = sum((a*d-b*c for (a,b),(c,d) in zip(v,v[1:]+v[:1])),F(0))/2
        assert area == F(anchors['initial_areas_exact'][i]) == F(1,2048)
    for source, item in inventory['N64_reused_fields'].items():
        assert item['sha256'] == digest(ROOT/item['file']) and not item['rerun']
        with np.load(ROOT/item['file'],allow_pickle=False) as z: a = z['rho']
        assert a.shape == (4096,) and np.min(a)>0 and np.isfinite(a).all()
        assert hashlib.sha256(a.tobytes()).hexdigest() == item['array']['sha256_C_bytes']
        assert float(np.min(a)) == item['raw_minimum'] and float(np.max(a)) == item['raw_maximum']
    # Algebraic scaling only, without generating a new lattice density or kernel sum.
    for n in (32,64,128,256):
        delta = F(1,n); h = 2*delta
        assert delta*delta/(h*h) == F(1,4) and 2*h < F(1,2)
    t1 = read(ROOT/'runs/reconnect-t1-density-calibration/results.json')['density_and_symbol_records']['kernel']
    warning = result['uniform_fixed_h_over_spacing_warning']
    assert warning['raw_density_saved_T1_N32'] == t1['raw_density_minimum'] == 1.0004157504550328
    assert warning['saved_T1_point_bias'] == t1['raw_density_vs_true_one_Linf'] > 0
    assert not warning['same_bias_asserted_for_moving_nonuniform_case']
    future = p['future_experiment']; assert future == result['future_experiment']
    assert future['new_density_fields'] == 4 and future['new_N'] == [128,256]
    assert future['reused_N64_fields'] == 2 and future['h_rule'] == '2/N'
    assert future['saved_frame'] == 63 and future['saved_clock_hex'] == '0x1.9333333333334p-5'
    assert future['new_scalar_paths'] == future['new_geometry'] == future['new_reference_PDE'] == 0
    assert future['new_software_tests'] == 0 and not future['already_executed']
    assert not future['automatic_expansion'] and not future['ledger_beating_oracle_required']
    assert p['metrics']['original_algebra_guard'] == 1e-10
    assert p['metrics']['precision_digits'] == [80,100]
    assert not p['metrics']['evaluation_indicator_is_certified_PDE_bound']
    assert result['new_physical_fields_or_solves'] == result['new_software_tests'] == 0
    assert result['stage3'] == 'incomplete' and not result['fluid_orc_enabled']
    assert result['recorded_checks_total'] == 508
    preserved = read(out/'preservation.json')
    assert len(preserved['formal_manifests_verified']) == 123
    assert preserved['originals_verified'] == 782 and preserved['attachments_verified'] == 15
    for item in preserved['formal_manifests_verified']:
        assert digest(ROOT/item['manifest']) == item['manifest_sha256']
    ex = read(out/'execution.json')
    assert ex['completed'] and ex['elapsed_seconds'] < 120 and ex['peak_RSS_MiB'] < 1024
    return {'passed':True, 'cases':reports, 'original_mass_and_saved_N64_fields_verified':True,
        'fixed_support_scaling_and_saved_T1_warning_verified':True, 'new_fields_or_solves':0,
        'new_software_tests':0, 'recorded_checks_total':508, 'stage3':'incomplete', 'fluid_orc_enabled':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory',required=True)
    parser.add_argument('--output'); args = parser.parse_args(); start = time.monotonic()
    def alarm(*_): raise TimeoutError('read-only audit120second cap')
    signal.signal(signal.SIGALRM,alarm); signal.alarm(120)
    result = audit(Path(args.directory).resolve())
    elapsed = time.monotonic()-start; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    assert elapsed < 120 and rss < 1024
    result.update(elapsed_seconds=elapsed,peak_RSS_MiB=rss)
    if args.output: save(args.output,result)
    print(result)
