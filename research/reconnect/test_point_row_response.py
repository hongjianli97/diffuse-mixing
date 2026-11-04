"""Raw-row identities, including deliberate row defects, without a graph."""
import unittest
import numpy as np
from point_row_response import row_response,signed_budget


class SavedRowChecks(unittest.TestCase):
    def test_original_mass_and_quadratic_mean_give_known_symmetric_row_action(self):
        values=np.array([2.,-1.,-1.]);before=values.copy()
        a,r=row_response(values,np.array([0,1,2]),np.array([1.,0.,2.]),0,2.,.5)
        np.testing.assert_array_equal(values,before)
        self.assertLess(abs(r['raw_CSR_quadratic_response_per_s']-.5),3e-15)
        self.assertEqual(r['raw_constant_response_per_s'],0.)
        self.assertEqual(r['first_normal_moment_m_per_s'],0.)
        self.assertEqual(r['half_second_normal_moment_m2_per_s'],.5)
        self.assertAlmostEqual(a['probe_quadratic_mass_means'][0],.5+.5**2/24)

    def test_nonzero_row_defect_is_retained_in_raw_action(self):
        values=np.array([2.25,-1.,-1.]);before=values.copy()
        a,r=row_response(values,np.array([0,1,2]),np.array([1.,0.,2.]),0,2.,.5)
        np.testing.assert_array_equal(values,before)
        self.assertEqual(r['raw_constant_response_per_s'],-.125)
        self.assertLess(abs(r['centered_difference_response_per_s']-.5),3e-15)
        expected=.5-.125*(.5+.5**2/24)
        self.assertLess(abs(r['raw_CSR_quadratic_response_per_s']-expected),3e-15)
        self.assertNotEqual(r['raw_CSR_quadratic_response_per_s'],r['centered_difference_response_per_s'])
        self.assertFalse(r['row_sum_or_moments_repaired'])

    def test_signed_budget_keeps_opposite_signs_and_float_closure_residual(self):
        r=signed_budget(7/64,9/64,8/64)
        self.assertEqual(r['point_minus_ideal_per_s'],-2/64)
        self.assertEqual(r['ideal_minus_PDE_per_s'],1/64)
        self.assertEqual(r['point_minus_PDE_per_s'],-1/64)
        self.assertEqual(r['signed_identity_residual_per_s'],0.)
        a=signed_budget(.1,1e12,.2)
        self.assertNotEqual(a['signed_identity_residual_per_s'],0.)
        self.assertEqual(a['signed_identity_residual_per_s'],(.1-1e12)+(1e12-.2)-(.1-.2))

    def test_invalid_mass_or_missing_diagonal_is_rejected(self):
        with self.assertRaises(ValueError):row_response([2.,-1.,-1.],np.array([0,1,2]),[1.,0.,2.],0,0.,.5)
        with self.assertRaises(ValueError):row_response([2.,-1.,-1.],np.array([0,1,2]),[1.,0.,2.],3,2.,.5)
        with self.assertRaises(ValueError):signed_budget(.02,.021,0.)


if __name__=='__main__':unittest.main(verbosity=2)
