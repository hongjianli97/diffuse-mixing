"""Six necessary checks of the limited point-input Cartesian FV reference."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
import argparse
import inspect
import json
import sys
import unittest

from cartesian_material_fv_rows import build_cartesian_material_rows, apply_material_rows


def fixture(nx=4, ny=4, periods=(F(1), F(1)), split=True):
    dx, dy = periods[0] / nx, periods[1] / ny
    points = [((F(i) + F(1, 2)) * dx, (F(j) + F(1, 2)) * dy) for j in range(ny) for i in range(nx)]
    rho = [F(1) if split and i < nx // 2 else F(4) if split else F(2) for j in range(ny) for i in range(nx)]
    return dict(positions=points, ids=list(range(nx * ny)), mass=[value * dx * dy for value in rho],
                rho=rho, periods=periods, D=F(1, 50), target_ids=[nx + nx // 2 - 1, nx + nx // 2])


def centered(value, origin, length):
    result = value - origin
    if result > length / 2:
        result -= length
    elif result < -length / 2:
        result += length
    return result


class CartesianMaterialFVRowsChecks(unittest.TestCase):
    def test_rectangular_mass_mean_quadratic_matches_independent_integrated_flux(self):
        data = fixture(4, 5, (F(4), F(5, 2)), split=False)
        data['target_ids'] = [9]
        stencil = build_cartesian_material_rows(**data)
        origin = data['positions'][9]
        dx, dy = stencil.spacing
        means = {identity: sum(centered(point[k], origin[k], data['periods'][k])**2 / 2 for k in range(2))
                           + (dx**2 + dy**2) / 24 for identity, point in zip(data['ids'], data['positions'])}
        self.assertEqual(means[9], F(5, 96))
        # Each pair of opposite physical face integrals contributes rho*D*b*area.
        integrated_divergence = 2 * data['rho'][9] * data['D'] * dx * dy
        self.assertEqual(apply_material_rows(stencil, means)[9], integrated_divergence / data['mass'][9])
        self.assertEqual(apply_material_rows(stencil, means)[9], 2 * data['D'])

    def test_interface_normal_and_transverse_quadratics_and_shared_positive_edge(self):
        data = fixture()
        stencil = build_cartesian_material_rows(**data)
        dx, dy = stencil.spacing
        origin_y = data['positions'][data['target_ids'][0]][1]
        normal = {identity: (point[0] - F(1, 2))**2 / 2 + dx**2 / 24
                  for identity, point in zip(data['ids'], data['positions'])}
        transverse = {identity: centered(point[1], origin_y, F(1))**2 / 2 + dy**2 / 24
                      for identity, point in zip(data['ids'], data['positions'])}
        for values in [normal, transverse]:
            self.assertEqual(tuple(apply_material_rows(stencil, values).values()), (data['D'], data['D']))
        shared = [edge for edge in stencil.edges if (edge.left_id, edge.right_id) == tuple(data['target_ids'])]
        self.assertEqual(len(stencil.edges), 7)
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].conductance, F(4, 125))
        self.assertTrue(all(edge.conductance > 0 for edge in stencil.edges))
        left, right = stencil.rows
        g_left = next(g for _, identity, g in left.neighbors if identity == right.target_id)
        g_right = next(g for _, identity, g in right.neighbors if identity == left.target_id)
        self.assertEqual(g_left, g_right)
        self.assertEqual(g_left * (F(7, 10) - F(1, 5)) + g_right * (F(1, 5) - F(7, 10)), 0)

    def test_nonzero_transmission_flux_uses_true_cell_means_and_correct_area(self):
        data = fixture()
        stencil = build_cartesian_material_rows(**data)
        J = F(1, 1000)
        means = {identity: F(1, 2) + J * (point[0] - F(1, 2)) / (rho * data['D'])
                 for identity, point, rho in zip(data['ids'], data['positions'], data['rho'])}
        self.assertEqual(tuple(apply_material_rows(stencil, means).values()), (F(0), F(0)))
        left, right = data['target_ids']
        edge = next(edge for edge in stencil.edges if (edge.left_id, edge.right_id) == (left, right))
        self.assertEqual(edge.conductance * (means[right] - means[left]), stencil.thickness * stencil.spacing[1] * J)

    def test_state_independence_density_length_scaling_D_linearity_and_zero(self):
        data = fixture()
        stencil = build_cartesian_material_rows(**data)
        values = {identity: F(identity, 100) for identity in data['ids']}
        rate = apply_material_rows(stencil, values)
        scaled = build_cartesian_material_rows(**{**data, 'mass': [7 * v for v in data['mass']], 'rho': [7 * v for v in data['rho']]})
        self.assertEqual(apply_material_rows(scaled, values), rate)
        self.assertEqual([edge.conductance for edge in scaled.edges], [7 * edge.conductance for edge in stencil.edges])
        expanded = build_cartesian_material_rows(**{**data, 'positions': [tuple(3 * v for v in p) for p in data['positions']],
            'periods': tuple(3 * v for v in data['periods']), 'mass': [9 * v for v in data['mass']]})
        self.assertEqual(apply_material_rows(expanded, values), {identity: value / 9 for identity, value in rate.items()})
        double = build_cartesian_material_rows(**{**data, 'D': 2 * data['D']})
        self.assertEqual(apply_material_rows(double, values), {identity: 2 * value for identity, value in rate.items()})
        zero = build_cartesian_material_rows(**{**data, 'D': 0})
        self.assertTrue(all(edge.conductance == 0 for edge in zero.edges))
        self.assertEqual(tuple(apply_material_rows(zero, values).values()), (F(0), F(0)))
        self.assertEqual(apply_material_rows(stencil, {identity: F(37, 100) for identity in data['ids']}),
                         {identity: F(0) for identity in data['target_ids']})
        self.assertEqual(stencil, build_cartesian_material_rows(**data))
        self.assertNotIn('concentration', inspect.signature(build_cartesian_material_rows).parameters)

    def test_ID_permutation_periodic_translation_and_immutable_inputs(self):
        data = fixture()
        before = deepcopy(data)
        stencil = build_cartesian_material_rows(**data)
        order = list(range(len(data['ids'])))[::-1]
        permuted = {**data, **{key: [data[key][i] for i in order] for key in ['positions', 'ids', 'mass', 'rho']}}
        self.assertEqual(stencil, build_cartesian_material_rows(**permuted))
        translated = {**data, 'positions': [tuple((point[k] + F(k + 1, 7)) % data['periods'][k] for k in range(2))
                                               for point in data['positions']]}
        self.assertEqual(stencil, build_cartesian_material_rows(**translated))
        self.assertEqual(data, before)
        with self.assertRaises(FrozenInstanceError):
            stencil.diffusivity = F(99)

    def test_invalid_data_or_unstated_geometry_cannot_be_repaired(self):
        data = fixture()
        for label, changes in [
            ('mass_defect', {'mass': [data['mass'][0] + F(1, 10**30), *data['mass'][1:]]}),
            ('bad_rho', {'rho': [0, *data['rho'][1:]]}), ('negative_D', {'D': -1}),
            ('nonfinite_D', {'D': float('nan')}), ('zero_thickness', {'thickness': 0}),
            ('duplicate_ID', {'ids': [data['ids'][1], *data['ids'][1:]]}),
            ('missing_target', {'target_ids': [999]}), ('empty_targets', {'target_ids': []}),
            ('nonuniform', {'positions': [(F(1, 9), data['positions'][0][1]), *data['positions'][1:]]}),
            ('unwrapped', {'positions': [(1, data['positions'][0][1]), *data['positions'][1:]]}),
            ('missing_point', {key: data[key][:-1] for key in ['positions', 'ids', 'mass', 'rho']}),
        ]:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, TypeError)):
                    build_cartesian_material_rows(**{**data, **changes})
        with self.assertRaises(ValueError):
            build_cartesian_material_rows(**fixture(2, 4))
        for forbidden in ['true_faces', 'F_C', 'velocity_gradients', 'reference_PDE', 'concentration']:
            with self.assertRaises(TypeError):
                build_cartesian_material_rows(**data, **{forbidden: None})
        with self.assertRaises(ValueError):
            apply_material_rows(build_cartesian_material_rows(**data), {})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CartesianMaterialFVRowsChecks)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    summary = {
        'checks_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'successful': result.wasSuccessful(),
        'check_names': sorted(name for name in dir(CartesianMaterialFVRowsChecks) if name.startswith('test_')),
        'failure_details': [{'test': test.id(), 'traceback': detail} for test, detail in result.failures + result.errors],
        'scope': 'six new FV-row software checks; no physical backend acceptance or full graph',
    }
    if args.json:
        with Path(args.json).open('x') as f:
            f.write(json.dumps(summary, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
