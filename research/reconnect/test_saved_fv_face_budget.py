"""Four necessary sign, reference-mismatch and exact-roundoff budget checks."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import unittest

from saved_fv_face_budget import signed_face_budget


def fixture():
    return dict(masses=[1, 4],
        fv_outward=[{'x_minus': -F(5, 2), 'x_plus': 4, 'y_minus': -F(1, 2), 'y_plus': 6},
                    {'x_minus': -4, 'x_plus': 8, 'y_minus': -5, 'y_plus': 3}],
        reference_faces=[{'left': 2, 'right': 3, 'bottom': 1, 'top': 5},
                         {'left': 3, 'right': 7, 'bottom': 4, 'top': 2}],
        saved_reference_rates=[5, F(1, 2)], published_FV_rates=[7, F(1, 2)], published_errors=[2, 0],
        global_reference_norm=2, published_pair_error_norm=1)


class SavedFVFaceBudgetChecks(unittest.TestCase):
    def test_independent_open_two_cell_flux_divergence_signs_and_cross_term(self):
        result = signed_face_budget(**fixture())
        self.assertEqual(result['signed_rate_components']['interface'], [F(1), -F(1, 4)])
        self.assertEqual(result['signed_rate_components']['other_three_faces'], [F(1), F(1, 4)])
        self.assertEqual(result['shared_interface']['common_interface_flux_defect'], 1)
        self.assertEqual(result['shared_interface']['interface_component_mass_balance'], 0)
        self.assertEqual(result['total_pair_inventory_error'], 2)  # the two-cell region is open
        self.assertEqual(result['Gram_sum'], 1)
        self.assertEqual(result['interface_and_other_cross_term'], F(3, 8))
        self.assertEqual(result['exact_Gram_closure'], 0)

    def test_nonshared_saved_reference_is_retained_without_averaging(self):
        data = fixture()
        data['reference_faces'][1]['left'] = F(13, 4)
        result = signed_face_budget(**data)
        shared = result['shared_interface']
        self.assertTrue(shared['FV_shared_face_equal'])
        self.assertFalse(shared['reference_shared_face_equal'])
        self.assertEqual(shared['reference_left_right_values'], [F(3), F(13, 4)])
        self.assertIsNone(shared['common_interface_flux_defect'])
        self.assertFalse(shared['reference_values_averaged_or_repaired'])
        self.assertEqual(result['rows'][1]['saved_PDE_minus_reference_face_divergence'], F(1, 16))
        self.assertTrue(all(row['published_error_budget_closure'] == 0 for row in result['rows']))

    def test_tiny_exact_defect_and_float_display_cancellation_are_both_retained(self):
        delta = F(1, 2**100)
        data = fixture()
        data['fv_outward'] = [
            {'x_minus': -2, 'x_plus': 3 + delta, 'y_minus': -1, 'y_plus': 5},
            {'x_minus': -(3 + delta), 'x_plus': 7, 'y_minus': -4, 'y_plus': 2},
        ]
        data['published_FV_rates'] = [float(5 + delta), float(F(1, 2) - delta / 4)]
        data['published_errors'] = [0., 0.]
        data['published_pair_error_norm'] = 0.
        result = signed_face_budget(**data)
        self.assertEqual(result['signed_rate_components']['interface'], [delta, -delta / 4])
        self.assertEqual(result['signed_rate_components']['display_bridge'], [-delta, delta / 4])
        self.assertEqual(result['shared_interface']['common_interface_flux_defect'], delta)
        self.assertGreater(result['normalized_Gram_matrix'][0][0], 0)
        self.assertLess(result['normalized_Gram_matrix'][0][3], 0)
        self.assertEqual(result['exact_normalized_published_error_square'], 0)
        self.assertEqual(result['exact_Gram_closure'], 0)

    def test_invalid_values_rejected_and_source_dictionaries_not_mutated(self):
        data = fixture()
        before = deepcopy(data)
        signed_face_budget(**data)
        self.assertEqual(data, before)
        for label, change in [('mass_zero', {'masses': [0, 4]}), ('mass_negative', {'masses': [-1, 4]}),
                              ('norm_zero', {'global_reference_norm': 0}),
                              ('norm_nan', {'global_reference_norm': float('nan')}),
                              ('missing_face', {'reference_faces': [{'right': 3}, data['reference_faces'][1]]}),
                              ('wrong_size', {'published_errors': [1]}),
                              ('nonfinite_flux', {'fv_outward': [{'x_minus': -2, 'x_plus': float('inf'), 'y_minus': -1, 'y_plus': 5}, data['fv_outward'][1]]})]:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, TypeError)):
                    signed_face_budget(**{**data, **change})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SavedFVFaceBudgetChecks)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    summary = {
        'checks_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'successful': result.wasSuccessful(),
        'check_names': sorted(name for name in dir(SavedFVFaceBudgetChecks) if name.startswith('test_')),
        'failure_details': [{'test': test.id(), 'traceback': detail} for test, detail in result.failures + result.errors],
        'scope': 'four new bookkeeping checks, not flow tests or transport changes',
    }
    if args.json:
        with Path(args.json).open('x') as f: f.write(json.dumps(summary, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
