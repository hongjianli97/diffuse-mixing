"""Six new localization checks; no saved-flow paths or old test suite runs."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import inspect
import json
import sys
import unittest

from endpoint_density_path import endpoint_ball_selection, localized_inverse_density_path
from periodic_density_path import inverse_density_path
from test_periodic_density_path import fixtures


def augmented(data):
    return {**data, 'positions': list(data['positions']) + [(F(3, 4), F(data['periods'][1]) * F(3, 4)),
                                                          (F(0), F(data['periods'][1]) * F(3, 4))],
            'ids': list(data['ids']) + [800, 801], 'rho': list(data['rho']) + [3, 9]}


class EndpointDensityPathChecks(unittest.TestCase):
    def test_localization_equals_unchanged_full_primitive_on_geometry_fixtures(self):
        for label, data in fixtures().items():
            with self.subTest(label=label):
                data = augmented(data)
                result = localized_inverse_density_path(**data)
                self.assertEqual(result.path, inverse_density_path(**data))
                self.assertLess(len(result.selection.retained_indices), len(data['ids']))
                self.assertTrue({identity for _, identity in result.path.breakpoint_owners}
                                <= set(result.selection.retained_ids))

    def test_closed_boundary_and_isolated_point_ties_are_retained(self):
        data = fixtures()['point_tie']
        data = {**data, 'positions': data['positions'] + [(F(2, 5), F(7, 10)), (F(3, 4), F(1, 2))],
                'ids': data['ids'] + [77, 88], 'rho': data['rho'] + [7, 2]}
        result = localized_inverse_density_path(**data)
        selection = result.selection
        self.assertEqual(selection.radius_squared, F(1, 16))
        self.assertEqual(selection.distance2_to_start[data['ids'].index(77)], selection.radius_squared)
        self.assertIn(77, selection.retained_ids)
        self.assertIn(1, selection.retained_ids)
        self.assertNotIn(88, selection.retained_ids)  # half-period distance to a far site is allowed
        self.assertEqual(result.path, inverse_density_path(**data))
        self.assertIn((F(1, 2), 1), result.path.breakpoint_owners)

    def test_geometry_only_selection_reversal_permutation_and_immutability(self):
        data = augmented(fixtures()['interval_tie'])
        before = deepcopy(data)
        result = localized_inverse_density_path(**data)
        changed = localized_inverse_density_path(**{**data, 'rho': [F(2 * i + 1) for i in range(len(data['ids']))]})
        self.assertEqual(result.selection, changed.selection)
        reverse = localized_inverse_density_path(**{**data, 'start_id': data['end_id'], 'end_id': data['start_id']})
        self.assertEqual(result.selection.retained_indices, reverse.selection.retained_indices)
        self.assertEqual(result.selection.distance2_to_start, reverse.selection.distance2_to_end)
        self.assertEqual(result.path.mean_inverse_density, reverse.path.mean_inverse_density)
        order = list(range(len(data['ids'])))[::-1]
        permuted = {**data, **{key: [data[key][i] for i in order] for key in ['positions', 'ids', 'rho']}}
        reordered = localized_inverse_density_path(**permuted)
        self.assertEqual(set(result.selection.retained_ids), set(reordered.selection.retained_ids))
        self.assertEqual(result.path, reordered.path)
        self.assertEqual(data, before)
        with self.assertRaises(FrozenInstanceError):
            result.selection.radius_squared = F(1)
        self.assertEqual(set(inspect.signature(endpoint_ball_selection).parameters),
                         {'positions', 'ids', 'periods', 'start_id', 'end_id'})
        self.assertEqual(set(inspect.signature(localized_inverse_density_path).parameters),
                         {'positions', 'ids', 'rho', 'periods', 'start_id', 'end_id'})

    def test_zero_length_coincidence_and_tiny_positive_float_distance(self):
        data = dict(positions=[(0., .25), (0., .25), (2.**-80, .25), (.25, .25)],
                    ids=[10, 0, 20, 30], rho=[1, 1, 2, 4], periods=[1, 1], start_id=10, end_id=0)
        zero = localized_inverse_density_path(**data)
        self.assertEqual(zero.selection.radius_squared, 0)
        self.assertEqual(zero.selection.retained_ids, (10, 0))
        self.assertEqual(zero.path.intervals[0].source_id, 0)
        self.assertEqual(zero.path.mean_inverse_density, 1)
        nonzero = localized_inverse_density_path(**{**data, 'end_id': 30})
        self.assertEqual(nonzero.path, inverse_density_path(**{**data, 'end_id': 30}))
        self.assertEqual(nonzero.path.intervals[0].stop, F(1, 2**79))

    def test_invalid_excluded_density_or_geometry_cannot_be_hidden_by_filter(self):
        data = augmented(fixtures()['collinear'])
        for label, change in [
            ('negative_far_density', {'rho': data['rho'][:-1] + [-1]}),
            ('nonfinite_far_density', {'rho': data['rho'][:-1] + [float('nan')]}),
            ('missing_density', {'rho': data['rho'][:-1]}),
            ('duplicate_far_ID', {'ids': data['ids'][:-1] + [data['ids'][-2]]}),
            ('unwrapped_far_coordinate', {'positions': data['positions'][:-1] + [(1, F(3, 4))]}),
            ('unknown_endpoint', {'end_id': 999}),
            ('half_period_path', {'positions': [*data['positions'][:2], (F(5, 8), F(1, 4)), *data['positions'][3:]]}),
            ('zero_period', {'periods': [1, 0]}),
        ]:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, TypeError)):
                    localized_inverse_density_path(**{**data, **change})
        conflict = {**data, 'positions': data['positions'] + [data['positions'][-1]],
                    'ids': data['ids'] + [999], 'rho': data['rho'] + [10]}
        with self.assertRaises(ValueError):
            localized_inverse_density_path(**conflict)
        with self.assertRaises(TypeError):
            localized_inverse_density_path(**data, mass=None)

    def test_all_squared_distance_certificates_match_independent_periodic_images(self):
        for label, data in fixtures().items():
            with self.subTest(label=label):
                data = augmented(data)
                selection = endpoint_ball_selection(**{key: data[key] for key in ['positions', 'ids', 'periods', 'start_id', 'end_id']})
                lookup = dict(zip(data['ids'], data['positions']))
                expected = []
                for endpoint in [data['start_id'], data['end_id']]:
                    origin = lookup[endpoint]
                    distances = tuple(min(sum((F(p[k]) + shift[k] * data['periods'][k] - origin[k])**2
                                              for k in range(2))
                                          for shift in product(range(-2, 3), repeat=2)) for p in data['positions'])
                    expected.append(distances)
                self.assertEqual(selection.distance2_to_start, expected[0])
                self.assertEqual(selection.distance2_to_end, expected[1])
                r2 = expected[0][data['ids'].index(data['end_id'])]
                self.assertEqual(selection.radius_squared, r2)
                self.assertEqual(selection.retained_indices, tuple(i for i in range(len(data['ids']))
                                                                 if expected[0][i] <= r2 and expected[1][i] <= r2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EndpointDensityPathChecks)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    summary = {
        'checks_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'successful': result.wasSuccessful(),
        'check_names': sorted(name for name in dir(EndpointDensityPathChecks) if name.startswith('test_')),
        'failure_details': [{'test': test.id(), 'traceback': detail} for test, detail in result.failures + result.errors],
        'scope': 'new exact localization software checks only; saved paths are separate observations, not physical acceptance',
    }
    if args.json:
        with Path(args.json).open('x') as f:
            f.write(json.dumps(summary, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
