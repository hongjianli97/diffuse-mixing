"""Unchanged audit algorithms specialized to one new thickness and three time grids."""
from fractions import Fraction as F
import ast,inspect,math
import numpy as np
from scipy.sparse import csr_matrix
from audit_route_decision import ROOT,read,digest
from audit_becker_spatial import Q,norm,save,arrays,audit_actions,audit_references
from becker_spatial_core import REFS

def audit_actions_delta004(out, N, write):
    folder = out / f'n{N}'
    world = read(folder / 'pure-inputs.json')
    index = read(folder / 'graph-index.json')
    data = arrays(folder / 'material.npz')
    m, c0 = (data['mass'], data['c0'])
    mm = list(map(Q, m))
    count = 0
    maximum = 0.0
    for K in (8, 16, 32):
        path = arrays(folder / f'k{K}/trajectory-states.npz')
        run = read(folder / f'k{K}/native-results.json')
        previous = c0.copy()
        machine = tail = 0.0
        np.testing.assert_array_equal(path['states'][0], c0)
        np.testing.assert_array_equal(path['independent_states'][0], c0)
        variance = float(m @ (c0 - float(m @ c0 / m.sum())) ** 2)
        V0 = variance
        for step, row in enumerate(run['steps']):
            item = index[f'k{K}-step{step:02d}']
            packet = read(folder / item['file'])
            receipt = read(folder / f'k{K}/geometry-audit-step{step:02d}.json')
            f = world['cases'][str(K)]['frames'][2 * step + 1]
            assert digest(folder / item['file']) == item['sha256'] == row['coefficient_cache_sha256'] == receipt['cache_sha256']
            assert receipt['pure_inputs_sha256'] == digest(folder / 'pure-inputs.json') and (not receipt['mass_c0_reference_loaded'])
            assert receipt['audit_source_sha256'] == digest(ROOT / 'research/reconnect/audit_bounded_geometry.py')
            cert = receipt['certificate']
            assert cert['all_cell_world_halfplanes_verified'] and cert['all_vertices_rebuilt_from_boundary_lines_and_bisectors'] and cert['all_non_nearest_y_images_excluded']
            assert row['midpoint_time_saved'] == item['midpoint_time_saved'] == float(F(2 * step + 1, 20 * K))
            points = [tuple(map(Q, p)) for p in f['positions']]
            rho = list(map(Q, f['rho']))
            D = Q(f['D'])
            Ly = Q(f['period_y'])
            degree = [F(0)] * len(m)
            gg = []
            edges = np.array(packet['edges'], dtype=np.int64)
            for j, ((i, k), (sx, sy)) in enumerate(zip(edges, packet['image_shifts'])):
                assert sx == 0 and i < k
                vx = points[k][0] - points[i][0]
                vy = points[k][1] + sy * Ly - points[i][1]
                ends = [tuple(map(F, p)) for p in packet['face_endpoints_exact'][j]]
                dx = ends[1][0] - ends[0][0]
                dy = ends[1][1] - ends[0][1]
                if dy > Ly / 2:
                    dy -= Ly
                if dy < -Ly / 2:
                    dy += Ly
                ratio = (vx * dy - vy * dx) / (vx * vx + vy * vy)
                assert ratio > 0 and ratio == F(packet['face_over_distance_exact'][j])
                g = ratio / (1 / (2 * rho[i] * D) + 1 / (2 * rho[k] * D))
                assert g == F(packet['conductances_exact'][j]) and g > 0
                degree[i] += g
                degree[k] += g
                gg.append(g)
            Lambda = math.ceil(max((d / a for d, a in zip(degree, mm))))
            assert Lambda == row['operator']['Lambda_exact_integer']
            rows = [{} for _ in m]
            for (i, j), g in zip(edges, gg):
                assert j not in rows[i] and i not in rows[j]
                rows[i][j] = float(g / (Lambda * mm[i]))
                rows[j][i] = float(g / (Lambda * mm[j]))
            for i, (d, a) in enumerate(zip(degree, mm)):
                q = 1 - d / (Lambda * a)
                assert q >= 0
                rows[i][i] = float(q)
            ptr = [0]
            cols = []
            vals = []
            for row_values in rows:
                for j in sorted(row_values):
                    cols.append(j)
                    vals.append(row_values[j])
                ptr.append(len(cols))
            P = csr_matrix((vals, cols, ptr), shape=(len(m), len(m)))
            saved = arrays(folder / f'k{K}/action-step{step:02d}.npz')
            for key, value in [('uniformization_data', P.data), ('uniformization_indices', P.indices), ('uniformization_indptr', P.indptr), ('edges', edges)]:
                np.testing.assert_array_equal(saved[key], value)
            weights = saved['weights']
            M = len(weights) - 1
            dt = F(1, 10 * K)
            z = Lambda * dt
            bound = z ** (M + 1) / math.factorial(M + 1) / (1 - z / (M + 2))
            assert row['dt_exact'] == str(dt) and row['poisson']['z_exact'] == str(z) and (row['poisson']['M'] == M)
            assert bound == F(row['poisson']['tail_upper_exact']) <= F(1, 10 ** 16)
            ratio = z / (M + 1)
            assert ratio >= 1 or z ** M / math.factorial(M) / (1 - ratio) > F(1, 10 ** 16)
            v = previous.copy()
            w = math.exp(-float(z))
            following = w * v
            assert weights[0] == w
            for j in range(1, M + 1):
                v = P @ v
                w *= float(z) / j
                assert weights[j] == w
                following += w * v
            np.testing.assert_array_equal(following, path['independent_states'][step + 1])
            previous = following
            assert math.fsum(weights) == row['poisson']['raw_weight_sum'] and (not row['poisson']['weights_or_state_normalized'])
            maxdegree = int(np.bincount(edges.ravel(), minlength=len(m)).max())
            allowance = float(64 * np.finfo(float).eps * (maxdegree + 1) * (M + 1) * (1 + float(z)) * max(1.0, math.sqrt(float(m.sum()))) * max(1.0, float(np.max(abs(c0)))))
            machine += allowance
            tail += float(bound) * math.sqrt(float(m.sum())) * float(np.max(abs(c0)))
            assert machine == row['cumulative_machine_indicator_absolute'] and tail == row['cumulative_tail_indicator_absolute']
            delta = norm(m, path['states'][step + 1] - following)
            assert delta == row['primary_control_mass_norm_difference'] and delta <= machine + tail
            maximum = max(maximum, delta)
            A = csr_matrix((saved['symmetric_data'], saved['symmetric_indices'], saved['symmetric_indptr']), shape=tuple(saved['shape']))
            root = np.sqrt(m)
            gf = np.array(list(map(float, gg)))
            i, j = edges.T
            np.testing.assert_array_equal(gf, saved['g_float'])
            action_error = 0.0
            for u, chi in zip((path['states'][step], path['exponential_half_states'][step], path['states'][step + 1]), row['chi_start_half_end']):
                flux = gf * (u[j] - u[i])
                net = np.zeros(len(m))
                np.add.at(net, i, flux)
                np.add.at(net, j, -flux)
                err = norm(m, A @ (root * u) / root - net / m)
                assert err <= 1e-10 * max(1.0, norm(m, net / m))
                action_error = max(action_error, err)
                assert float(gf @ (u[j] - u[i]) ** 2) == chi and chi >= 0
            u = path['states'][step + 1]
            drift = float(m @ (u - c0))
            assert drift == row['raw_mass_change_from_c0'] and abs(drift) / float(m @ c0) < 1e-10
            assert u.min() >= c0.min() - 1e-10 and u.max() <= c0.max() + 1e-10
            newvariance = float(m @ (u - float(m @ u / m.sum())) ** 2)
            assert newvariance <= variance + 1e-10 * V0
            variance = newvariance
            result = {'columns': N, 'K': K, 'step': step, 'sealed_independent_all_cell_geometry_receipt_verified': True, 'exact_face_resistance_rates_minimal_tail_raw_weights_complete_sequence_verified': True, 'primary_control_mass_norm_difference': delta, 'maximum_symmetric_shared_action_difference': action_error}
            target = folder / f'k{K}/action-audit-step{step:02d}.json'
            if write:
                save(target, result)
            else:
                assert read(target) == result
            count += 1
        print(f'independent saved n{N}/K{K} full sequence verified', flush=True)
    assert count == 56
    return {'columns': N, 'saved_actions_verified': count, 'maximum_primary_control_difference': maximum, 'all_cell_proof_policy': 'verify pre-scalar independent full geometry proof receipts; independently recompute every face resistance, rate and sequence; do not rerun all-cell search'}

def audit_references_delta004(out):
    columns = {N: arrays(out / f'n{N}/column.npz') for N in (32, 128)}
    p = list(map(Q, columns[128]['mass_edges']))
    mc = list(map(Q, columns[128]['mass']))
    contract = read(out / 'reference/mesh-contract.json')
    counts = [math.ceil(v / ((p[-1] - p[0]) / 16384)) for v in mc]
    assert counts == contract['counts']
    records = []
    endpoints = {}
    for level, k in REFS:
        name = f'l{level}-k{k}'
        raw = arrays(out / f'reference/raw-{name}.npz')
        inp = arrays(out / f'reference/input-{name}.npz')
        obs = arrays(out / f'reference/observations-{name}.npz')
        guard = read(out / f'reference/guard-{name}.json')
        for key in inp:
            np.testing.assert_array_equal(inp[key], raw[key])
        e = list(map(Q, raw['mass_edges']))
        mass = list(map(Q, raw['mass']))
        c0 = list(map(Q, raw['c0']))
        c = list(map(Q, raw['c']))
        index = {v: i for i, v in enumerate(e)}
        cursor = 0
        for left, right, count in zip(p[:-1], p[1:], counts):
            n = count * 2 ** level
            for j in range(n + 1):
                assert float(left + (right - left) * j / n) == float(e[cursor + j])
            cursor += n
        assert cursor == len(mass)
        np.testing.assert_array_equal(raw['mass'], np.diff(raw['mass_edges']))
        if level in endpoints:
            np.testing.assert_array_equal(endpoints[level], raw['mass_edges'])
        else:
            endpoints[level] = raw['mass_edges']
        assert guard['steps_completed'] == guard['linear_solves'] == k and guard['positivity_policy'] == 'monitor' and (not guard['states_clipped_or_normalized'])
        assert guard['max_relative_mass_drift'] <= 1e-09 and guard['min_all_times'] >= guard['initial_min'] - 1e-10 and (guard['max_all_times'] <= guard['initial_max'] + 1e-10)
        assert np.isfinite(raw['c']).all() and raw['c'].min() >= guard['min_all_times'] and (raw['c'].max() <= guard['max_all_times'])
        prefix = [[F(0)] for _ in range(5)]
        for j, m in enumerate(mass):
            defect = abs(m - (e[j + 1] - e[j]))
            values = (m, m * c0[j], m * c[j], defect * abs(c0[j]), defect * abs(c[j]))
            for a, v in zip(prefix, values):
                a.append(a[-1] + v)
        drift = abs(prefix[2][-1] - prefix[1][-1]) / abs(prefix[1][-1])
        assert drift <= F(1, 10 ** 9)
        for N in (32,):
            target = list(map(Q, columns[N]['mass_edges']))
            tm = list(map(Q, columns[N]['mass']))
            tc = list(map(Q, columns[N]['c0']))
            ri = []
            rf = []
            gap = []
            gi = []
            gf = []
            closure = []
            for i, (left, right) in enumerate(zip(target[:-1], target[1:])):
                lo = index[left]
                hi = index[right]
                mass_sum, i_sum, f_sum, i_geo, f_geo = [a[hi] - a[lo] for a in prefix]
                assert right - left == tm[i]
                im = i_sum / tm[i]
                fm = f_sum / tm[i]
                gap.append(im - tc[i])
                ri.append(Q(obs[f'n{N}_initial_projected'][i]) - im)
                rf.append(Q(obs[f'n{N}_projected'][i]) - fm)
                gi.append(i_geo / tm[i])
                gf.append(f_geo / tm[i])
                closure.append(mass_sum - tm[i])
                assert float(im) == obs[f'n{N}_exact_initial_projected_rounded'][i] and float(fm) == obs[f'n{N}_exact_projected_rounded'][i]
            for key, values in [('exact_initial_mean_gap', gap), ('initial_aggregation_rounding', ri), ('terminal_aggregation_rounding', rf), ('initial_geometry_bound', gi), ('terminal_geometry_bound', gf)]:
                np.testing.assert_array_equal(obs[f'n{N}_{key}'], np.array(list(map(float, values))))
            records.append({'reference_level': level, 'steps': k, 'observed_columns': N, 'maximum_exact_mass_closure': float(max(map(abs, closure))), 'exact_endpoint_relative_mass_drift': float(drift), 'all_complete_child_means_verified': True})
    assert np.array_equal(endpoints[0], endpoints[1][::2])
    return records

def verify_specializations():
    for original,new in ((audit_actions,audit_actions_delta004),(audit_references,audit_references_delta004)):
        old=ast.parse(inspect.getsource(original)).body[0]
        target=ast.parse(inspect.getsource(new)).body[0];old.name=target.name
        for n in ast.walk(old):
            if original is audit_actions:
                if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='KS':n.iter=ast.parse('(8,16,32)',mode='eval').body
                if isinstance(n,ast.Assert) and ast.dump(n.test)==ast.dump(ast.parse('count==28',mode='eval').body):n.test=ast.parse('count==56',mode='eval').body
            else:
                if isinstance(n,ast.comprehension) and isinstance(n.iter,ast.Name) and n.iter.id=='NS':n.iter=ast.parse('(32,128)',mode='eval').body
                if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='NS':n.iter=ast.parse('(32,)',mode='eval').body
        assert ast.dump(old)==ast.dump(target)
