"""Necessary correctness checks for the point-only path primitive.

The independent oracle intersects all periodic-site Voronoi inequalities using
25 images per node; it does not construct a lower hull. All fixtures are small
geometry unit inputs, not new flow cases or candidate graphs.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import product
import argparse
import inspect
import json
from pathlib import Path
import sys
import unittest

from periodic_density_path import inverse_density_path


def fixtures():
    return {
        'collinear': dict(positions=[(F(1, 8), F(1, 4)), (F(3, 16), F(1, 4)), (F(3, 8), F(1, 4))],
                          ids=[10, 20, 30], rho=[1, 2, 4], periods=[1, 1], start_id=10, end_id=30),
        'off_axis': dict(positions=[(F(1, 4), F(1, 2)), (F(1, 2), F(1, 2)), (F(3, 8), F(9, 16))],
                        ids=[10, 30, 20], rho=[1, 4, 2], periods=[1, 1], start_id=10, end_id=30),
        'periodic_corner': dict(positions=[(F(15, 16), F(31, 16)), (F(1, 16), F(1, 16)), (0, 0)],
                                ids=[10, 30, 20], rho=[1, 4, 2], periods=[1, 2], start_id=10, end_id=30),
        'interval_tie': dict(positions=[(F(1, 4), F(1, 2)), (F(1, 2), F(1, 2)),
                                         (F(3, 8), F(9, 16)), (F(3, 8), F(7, 16))],
                             ids=[30, 40, 20, 10], rho=[3, 5, 2, 4], periods=[1, 1], start_id=30, end_id=40),
        'point_tie': dict(positions=[(F(1, 4), F(1, 2)), (F(1, 2), F(1, 2)), (F(3, 8), F(5, 8))],
                         ids=[10, 20, 1], rho=[1, 4, 9], periods=[1, 1], start_id=10, end_id=20),
    }


def independent_intersections(data):
    """Intersect pairwise squared-distance inequalities for each periodic site."""
    points = [tuple(map(F, point)) for point in data['positions']]
    box = tuple(map(F, data['periods']))
    locations = dict(zip(data['ids'], points))
    origin, end = locations[data['start_id']], locations[data['end_id']]
    end_images = [tuple(end[k] + shift[k] * box[k] for k in range(2))
                  for shift in product((-1, 0, 1), repeat=2)]
    lifted_end = min(end_images, key=lambda p: sum((p[k] - origin[k])**2 for k in range(2)))
    sites = []
    for point, identity, rho in zip(points, data['ids'], data['rho']):
        for shift in product(range(-2, 3), repeat=2):
            p = tuple(point[k] + shift[k] * box[k] for k in range(2))
            sites.append((p, (identity, *shift), F(rho)))
    endpoint_distances = [(sum((origin[k] - p[k])**2 for k in range(2)),
                           sum((lifted_end[k] - p[k])**2 for k in range(2)))
                          for p, _, _ in sites]
    pieces = []
    for index, (_, priority, rho) in enumerate(sites):
        low, high = F(0), F(1)
        d0, d1 = endpoint_distances[index]
        for other, (_, other_priority, _) in enumerate(sites):
            e0, e1 = endpoint_distances[other]
            a, b = d0 - e0, (d1 - e1) - (d0 - e0)
            if b == 0:
                if a > 0 or a == 0 and priority > other_priority:
                    low, high = F(1), F(0)
                    break
            elif b > 0:
                high = min(high, -a / b)
            else:
                low = max(low, -a / b)
            if low >= high:
                break
        if low < high:
            pieces.append((low, high, priority[0], rho))
    pieces.sort()
    merged = []
    for piece in pieces:
        if merged and piece[2] == merged[-1][2] and piece[0] == merged[-1][1]:
            merged[-1] = (merged[-1][0], piece[1], piece[2], piece[3])
        else:
            merged.append(piece)
    def owner(parameter):
        x = tuple(origin[k] + parameter * (lifted_end[k] - origin[k]) for k in range(2))
        return min(sites, key=lambda site: (
            sum((x[k] - site[0][k])**2 for k in range(2)), site[1]))[1][0]
    return tuple(merged), sum(((b - a) / rho for a, b, _, rho in merged), F(0)), owner


def parts(result):
    return tuple((p.start, p.stop, p.source_id, p.density) for p in result.intervals)


class PeriodicDensityPathChecks(unittest.TestCase):
    def assertOracle(self, data):
        result = inverse_density_path(**data)
        expected, mean, owner = independent_intersections(data)
        self.assertEqual(parts(result), expected)
        self.assertEqual(result.mean_inverse_density, mean)
        self.assertEqual(sum((p.stop - p.start for p in result.intervals), F(0)), 1)
        for parameter, identity in result.breakpoint_owners:
            self.assertEqual(identity, owner(parameter))
        return result

    def test_independent_unequal_collinear_piecewise_resistance(self):
        result = self.assertOracle(fixtures()['collinear'])
        self.assertEqual(parts(result), ((F(0), F(1, 8), 10, F(1)),
                                        (F(1, 8), F(5, 8), 20, F(2)),
                                        (F(5, 8), F(1), 30, F(4))))
        self.assertEqual(result.mean_inverse_density, F(15, 32))

    def test_off_axis_node_changes_path_and_matches_independent_oracle(self):
        data = fixtures()['off_axis']
        result = self.assertOracle(data)
        self.assertEqual([p.stop for p in result.intervals], [F(5, 16), F(11, 16), F(1)])
        self.assertEqual(result.mean_inverse_density, F(37, 64))
        endpoint_only = {**data, 'positions': data['positions'][:2], 'ids': data['ids'][:2], 'rho': data['rho'][:2]}
        self.assertEqual(inverse_density_path(**endpoint_only).mean_inverse_density, F(5, 8))
        self.assertNotEqual(result.mean_inverse_density, F(5, 8))

    def test_reversal_preserves_exact_mean_and_reverses_ownership(self):
        for label, data in fixtures().items():
            with self.subTest(label=label):
                forward = inverse_density_path(**data)
                reverse = inverse_density_path(**{**data, 'start_id': data['end_id'], 'end_id': data['start_id']})
                self.assertEqual(forward.mean_inverse_density, reverse.mean_inverse_density)
                self.assertEqual(forward.displacement, tuple(-value for value in reverse.displacement))
                self.assertEqual(parts(forward), tuple((1 - b, 1 - a, identity, rho)
                                                      for a, b, identity, rho in reversed(parts(reverse))))
                self.assertEqual(forward.breakpoint_owners,
                                 tuple((1 - parameter, identity) for parameter, identity in reversed(reverse.breakpoint_owners)))

    def test_periodic_corner_and_exact_wrapped_translation(self):
        data = fixtures()['periodic_corner']
        result = self.assertOracle(data)
        self.assertEqual(result.displacement, (F(1, 8), F(1, 8)))
        self.assertEqual(result.mean_inverse_density, F(9, 16))
        shift = (F(3, 16), F(5, 16))
        translated = {**data, 'positions': [tuple((F(point[k]) + shift[k]) % data['periods'][k]
                                                   for k in range(2)) for point in data['positions']]}
        self.assertEqual(result, self.assertOracle(translated))

    def test_fixed_ID_ties_permutations_and_point_only_owner(self):
        data = fixtures()['interval_tie']
        result = self.assertOracle(data)
        self.assertEqual([p.source_id for p in result.intervals], [30, 10, 40])
        self.assertEqual(result.mean_inverse_density, F(25, 96))
        permutation = [2, 0, 3, 1]
        permuted = {**data, **{key: [data[key][i] for i in permutation] for key in ['positions', 'ids', 'rho']}}
        self.assertEqual(result, self.assertOracle(permuted))
        point = self.assertOracle(fixtures()['point_tie'])
        self.assertEqual([p.source_id for p in point.intervals], [10, 20])
        self.assertEqual(point.breakpoint_owners, ((F(0), 10), (F(1, 2), 1), (F(1), 20)))
        self.assertEqual(point.mean_inverse_density, F(5, 8))
        simple = fixtures()['collinear']
        coincident = {**simple, 'positions': simple['positions'] + [simple['positions'][0]],
                      'ids': simple['ids'] + [0], 'rho': simple['rho'] + [simple['rho'][0]]}
        duplicate = self.assertOracle(coincident)
        self.assertEqual(duplicate.intervals[0].source_id, 0)
        self.assertEqual(duplicate.mean_inverse_density, F(15, 32))

    def test_density_scaling_constant_and_zero_length_limits(self):
        data = fixtures()['off_axis']
        original = inverse_density_path(**data)
        scaled = inverse_density_path(**{**data, 'rho': [F(7, 3) * value for value in data['rho']]})
        self.assertEqual(scaled.mean_inverse_density, original.mean_inverse_density / F(7, 3))
        self.assertEqual([(p.start, p.stop, p.source_id) for p in original.intervals],
                         [(p.start, p.stop, p.source_id) for p in scaled.intervals])
        constant = self.assertOracle({**data, 'rho': [7, 7, 7]})
        self.assertEqual(constant.mean_inverse_density, F(1, 7))
        zero = self.assertOracle({**data, 'start_id': 20, 'end_id': 20})
        self.assertEqual(zero.displacement, (F(0), F(0)))
        self.assertEqual(parts(zero), ((F(0), F(1), 20, F(2)),))
        self.assertEqual(zero.mean_inverse_density, F(1, 2))

    def test_invalid_inputs_rejected_without_wrapping_or_filling(self):
        base = fixtures()['collinear']
        bad = [
            ('empty', {'positions': [], 'ids': [], 'rho': []}),
            ('lengths', {'rho': [1, 2]}), ('duplicate_IDs', {'ids': [10, 10, 30]}),
            ('bool_ID', {'ids': [True, 20, 30]}), ('float_ID', {'ids': [10., 20, 30]}),
            ('zero_rho', {'rho': [0, 2, 4]}), ('negative_rho', {'rho': [-1, 2, 4]}),
            ('nan_rho', {'rho': [float('nan'), 2, 4]}), ('infinite_rho', {'rho': [1, float('inf'), 4]}),
            ('boolean_rho', {'rho': [True, 2, 4]}), ('zero_period', {'periods': [1, 0]}),
            ('period_shape', {'periods': [1, 1, 1]}), ('missing_ID', {'end_id': 999}),
            ('noninteger_endpoint_ID', {'end_id': 30.}),
            ('unwrapped', {'positions': [(1, F(1, 4)), *base['positions'][1:]]}),
            ('negative_coordinate', {'positions': [(-F(1, 8), F(1, 4)), *base['positions'][1:]]}),
            ('nonfinite_coordinate', {'positions': [(float('nan'), .25), *base['positions'][1:]]}),
            ('point_shape', {'positions': [(0, 0, 0), *base['positions'][1:]]}),
            ('conflicting_coincidence', {'positions': [base['positions'][0], base['positions'][0], base['positions'][2]]}),
            ('ambiguous_half_period', {'positions': [*base['positions'][:2], (F(5, 8), F(1, 4))]}),
            ('long_diagonal', {'positions': [*base['positions'][:2], (F(1, 2), F(5, 8))]}),
        ]
        for label, changes in bad:
            with self.subTest(label=label):
                with self.assertRaises((TypeError, ValueError)):
                    inverse_density_path(**{**base, **changes})

    def test_source_immutability_frozen_output_and_point_only_signature(self):
        base = fixtures()['collinear']
        data = {**base, 'positions': [list(point) for point in base['positions']]}
        before = deepcopy(data)
        result = inverse_density_path(**data)
        self.assertEqual(data, before)
        self.assertEqual(set(inspect.signature(inverse_density_path).parameters),
                         {'positions', 'ids', 'rho', 'periods', 'start_id', 'end_id'})
        for key in ['mass', 'concentration', 'true_material_faces', 'F_C', 'reference_PDE']:
            with self.assertRaises(TypeError):
                inverse_density_path(**data, **{key: None})
        with self.assertRaises(FrozenInstanceError):
            result.mean_inverse_density = F(99)
        data['positions'][0][0] = F(0)
        data['rho'][0] = 99
        self.assertEqual(result.mean_inverse_density, F(15, 32))
        self.assertEqual(result.intervals[0].density, 1)

    def test_exact_binary_float_input_keeps_tiny_positive_interval(self):
        eta = F(1, 2**80)
        data = dict(positions=[(0., .25), (2.**-80, .25), (.25, .25)], ids=[10, 20, 30],
                    rho=[1., 2., 4.], periods=[1., 1.], start_id=10, end_id=30)
        result = self.assertOracle(data)
        self.assertEqual(result.intervals[0].stop, 2 * eta)
        self.assertGreater(result.intervals[0].stop - result.intervals[0].start, 0)
        self.assertEqual(result.mean_inverse_density, F(3, 8) + F(3, 2) * eta)
        self.assertGreater(result.mean_inverse_density, F(3, 8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PeriodicDensityPathChecks)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    summary = {
        'checks_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'successful': result.wasSuccessful(),
        'check_names': sorted(name for name in dir(PeriodicDensityPathChecks) if name.startswith('test_')),
        'failure_details': [{'test': test.id(), 'traceback': detail} for test, detail in result.failures + result.errors],
        'scope': 'new primitive software unit checks; not flow tests or physical backend acceptance',
        'independent_oracle': 'per-site periodic Voronoi halfspace intersection with 25 images per node',
    }
    if args.json:
        with Path(args.json).open('x') as f:
            f.write(json.dumps(summary, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
