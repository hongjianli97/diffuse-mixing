"""Raw variable-mass parent observations, without modifying saved masses."""
import unittest
import numpy as np
from smooth_density_probe import analytic_reference
from variable_parent_observation import parent_mass_audit,observe_parent


class VariableObservationChecks(unittest.TestCase):
    def test_exact_analytic_mass_averages_and_full_PDE_restrict_to_parent(self):
        parent=analytic_reference(8,.25,.02);fine=analytic_reference(16,.25,.02)
        child_sum,delta,audit=parent_mass_audit(fine['mass'],parent['mass'])
        self.assertLess(audit['maximum_relative_child_sum_discrepancy'],2e-15)
        for key in ['state','rate','laplacian_rate','drift_rate']:
            np.testing.assert_allclose(observe_parent(fine[key],fine['mass'],parent['mass']),parent[key],atol=5e-16)

    def test_rounding_sized_mass_difference_is_reported_and_not_repaired(self):
        parent=analytic_reference(4,.25,.02);fine=analytic_reference(8,.25,.02);mf=fine['mass'].copy();mp=parent['mass'].copy()
        for _ in range(8):mf[0]=np.nextafter(mf[0],np.inf)
        saved_fine=mf.copy();saved_parent=mp.copy();child_sum,delta,audit=parent_mass_audit(mf,mp)
        self.assertFalse(audit['bitwise_equal_child_sums_and_parent_mass']);self.assertGreater(audit['non_bitwise_equal_parent_count'],0)
        observed=observe_parent(np.ones(len(mf)),mf,mp)
        np.testing.assert_array_equal(observed,child_sum/mp);np.testing.assert_array_equal(mf,saved_fine);np.testing.assert_array_equal(mp,saved_parent)
        self.assertFalse(audit['child_mass_renormalized']);self.assertFalse(audit['saved_parent_mass_modified'])

    def test_unequal_mass_component_sum_is_preserved_for_signed_observations(self):
        parent=analytic_reference(8,.25,.02);fine=analytic_reference(16,.25,.02);v=np.sin(np.arange(256)*.37)
        observed=observe_parent(v,fine['mass'],parent['mass'])
        self.assertAlmostEqual(float(parent['mass']@observed),float(fine['mass']@v),places=15)

    def test_signed_component_projection_preserves_linear_error_identity(self):
        parent=analytic_reference(4,.25,.02);fine=analytic_reference(8,.25,.02);x=np.arange(64)
        components=np.array([np.sin(.2*x),-.9*np.sin(.2*x)+.03*np.cos(.3*x),.001*np.cos(.7*x)])
        R=lambda v:observe_parent(v,fine['mass'],parent['mass'])
        np.testing.assert_allclose(R(components.sum(axis=0)),sum(R(v) for v in components),atol=3e-16)
        self.assertLess(np.sqrt(parent['mass']@(R(components.sum(axis=0))**2)),sum(np.sqrt(parent['mass']@(R(v)**2)) for v in components))

    def test_incompatible_or_malformed_masses_are_rejected_without_repair(self):
        parent=analytic_reference(4,.25,.02);fine=analytic_reference(8,.25,.02);bad=fine['mass'].copy();bad[0]*=1.1
        with self.assertRaises(ValueError):parent_mass_audit(bad,parent['mass'])
        with self.assertRaises(ValueError):parent_mass_audit(-fine['mass'],parent['mass'])
        with self.assertRaises(ValueError):observe_parent(fine['state'][:-1],fine['mass'],parent['mass'])
        with self.assertRaises(ValueError):parent_mass_audit(fine['mass'],parent['mass'][:-1])


if __name__=='__main__':unittest.main(verbosity=2)
